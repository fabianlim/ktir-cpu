# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Memory compute helpers.

Tile view construction, sub-tile access, and HBM/LX load/store
primitives used by dialect handlers in ``ktir_cpu.dialects``.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
from ..affine import AffineMap, AffineSet, BoxSet
from ..dialects.ktdp_helpers import eval_subscript_expr
from ..dtypes import bytes_per_elem as _bytes_per_elem, to_np_dtype as _to_np_dtype
from ..ir_types import Tile, TileRef, DistributedTileRef
from ..grid import CoreContext
from ..memory import HBMSimulator




class MemoryOps:
    """Tile memory helpers — view, access, load, store."""

    @staticmethod
    def tile_view(
        context: CoreContext,
        ptr: int,
        shape: Tuple[int, ...],
        strides: List[int],
        memory_space: str,
        dtype: str = "f16",
        coordinate_set: Optional[str] = None,
    ) -> TileRef:
        """Create a memory layout descriptor (TileRef).

        Builds a tile reference describing a contiguous region in HBM or LX.

        Args:
            context: Core execution context
            ptr: Base pointer
            shape: Tile shape
            strides: Memory strides
            memory_space: "HBM" or "LX"
            dtype: Data type
            coordinate_set: Verbatim affine_set string, no evaluation

        Returns:
            TileRef describing the memory layout
        """
        return TileRef(
            base_ptr=ptr,
            shape=shape,
            strides=strides,
            memory_space=memory_space,
            dtype=dtype,
            coordinate_set=coordinate_set,
        )

    @staticmethod
    def tile_access(
        context: CoreContext,
        parent_ref: TileRef,
        indices: List[int],
        access_shape: Tuple[int, ...],
        base_map: AffineMap,
    ) -> TileRef:
        """Extract a sub-tile from a parent tile reference.

        Evaluates *base_map* with *indices* to obtain the base coordinates
        in the parent memref, then computes a byte offset using the parent
        strides.  The resulting base_ptr is always within the same allocation
        as parent_ref.base_ptr — this invariant is relied upon by load/store.

        Args:
            context: Core execution context
            parent_ref: Parent tile reference (memref)
            indices: Access indices (one per base_map input dim)
            access_shape: Shape of the accessed sub-tile
            base_map: AffineMap mapping indices → base coordinates

        Returns:
            TileRef for the sub-tile
        """
        base_coords = base_map.eval(indices)
        bpe = _bytes_per_elem(parent_ref.dtype)
        offset = sum(coord * stride for coord, stride in zip(base_coords, parent_ref.strides))
        new_ptr = parent_ref.base_ptr + offset * bpe

        return TileRef(
            base_ptr=new_ptr,
            shape=access_shape,
            strides=parent_ref.strides,
            memory_space=parent_ref.memory_space,
            dtype=parent_ref.dtype
        )

    @staticmethod
    def _is_contiguous(shape: Tuple[int, ...], strides: Tuple[int, ...]) -> bool:
        """Check if a shape/strides pair describes contiguous (row-major) memory."""
        expected_stride = 1
        for dim, stride in zip(reversed(shape), reversed(strides)):
            if stride != expected_stride:
                return False
            expected_stride *= dim
        return True

    @staticmethod
    def _write_to_lx(context: CoreContext, data: np.ndarray):
        """Write data into the core-local LX scratchpad.

        Advances ``next_ptr`` so subsequent writes don't collide.
        LX capacity accounting is handled by ``CoreContext.track_lx()``
        in ``_execute_operation`` — we only reserve address space here.
        All loaded Tiles always land in LX regardless of source memory space.
        """
        size = data.nbytes
        lx_ptr = context.lx.next_ptr
        context.lx.next_ptr += size
        context.lx.next_ptr = (context.lx.next_ptr + HBMSimulator.STICK_BYTES - 1) & ~(HBMSimulator.STICK_BYTES - 1)
        context.lx.write(lx_ptr, data)

    @staticmethod
    def _flat_memory_offsets(
        base_ptr: int,
        shape: Tuple[int, ...],
        strides: List[int],
        dtype: str,
        coords: Optional[List[Tuple[int, ...]]] = None,
    ) -> Tuple[List[int], int]:
        """Linearize N-d coordinates to flat element offsets and count unique HBM sticks.

        O(n) time, O(unique_sticks) memory for the stick set. For very large
        coord sizes a more efficient implementation may be needed.

        Returns:
            (offsets, unique_sticks) — flat element offsets and number of
            distinct HBM sticks touched.
        """
        offsets = []
        sticks = set()
        bpe = _bytes_per_elem(dtype)
        for coord in (coords if coords is not None else np.ndindex(*shape)):
            o = sum(c * s for c, s in zip(coord, strides))
            offsets.append(o)
            sticks.add((base_ptr + o * bpe) // HBMSimulator.STICK_BYTES)
        return offsets, len(sticks)

    @staticmethod
    def load(
        context: CoreContext,
        tile_ref: TileRef,
        coords: Optional[List[Tuple[int, ...]]] = None,
        result_shape: Optional[Tuple[int, ...]] = None,
    ) -> Tile:
        """Load data from HBM or LX into LX and return a Tile.

        All loaded Tiles always land in LX regardless of source memory space:
        - HBM source → DMA read from HBM, write into LX scratchpad.
        - LX source  → logical copy within LX (no physical movement).

        When *coords* is given (coordinate-set path), gathers only the
        elements at those local coordinates and reshapes to *result_shape*.
        When *coords* is None, loads the full tile described by tile_ref
        (contiguous or strided).

        A single ``mem.read`` covers the entire element footprint; no
        per-element dict scans occur.

        Example — loading column 2 of a 4×4 f16 matrix (strided, coords=None)::

            # Parent 4×4 allocation at base_ptr=0x1000, values 0..15
            # tile_ref for column 2: base_ptr=0x1004, shape=(4,), strides=[4]
            #   flat offsets: [0*4, 1*4, 2*4, 3*4] = [0, 4, 8, 12]
            #   span = 13  (max offset + 1)
            #   mem.read(0x1004, 13) -> [2,3,4,5,6,7,8,9,10,11,12,13,14]
            #   gathered = flat[[0,4,8,12]] = [2, 6, 10, 14]  ✓

        Example — upper-triangular load from a 4×4 tile (coords provided)::

            # tile_ref: base_ptr=0x1000, shape=(4,4), strides=[4,1]
            # coords = [(0,0),(0,1),...,(3,3)]  — 10 upper-tri tuples
            #   flat offsets = [0*4+0, 0*4+1, ..., 3*4+3] = [0,1,2,3,5,6,7,10,11,15]
            #   span = 16
            #   mem.read(0x1000, 16) -> flat 0..15
            #   gathered = flat[[0,1,2,3,5,6,7,10,11,15]] = [0,1,2,3,5,6,7,10,11,15]

        Args:
            context: Core execution context
            tile_ref: Tile reference (memref) describing source
            coords: Optional list of local coordinate tuples to gather.
                    Each tuple is 0-based within tile_ref.shape.
            result_shape: Output shape when coords is given; defaults to
                          tile_ref.shape when coords is None.

        Returns:
            Tile value (tensor) loaded into LX
        """
        mem = context.hbm if tile_ref.memory_space == "HBM" else context.lx

        # Fast path: contiguous tile, no coord filtering — single dict-key read.
        if coords is None and MemoryOps._is_contiguous(tile_ref.shape, tile_ref.strides):
            n = int(np.prod(tile_ref.shape))
            data = mem.read(tile_ref.base_ptr, n, tile_ref.dtype).reshape(tile_ref.shape)
            MemoryOps._write_to_lx(context, data)
            bpe = _bytes_per_elem(tile_ref.dtype)
            end = tile_ref.base_ptr + n * bpe
            unique_sticks = (
                (end + HBMSimulator.STICK_BYTES - 1) // HBMSimulator.STICK_BYTES
                - tile_ref.base_ptr // HBMSimulator.STICK_BYTES
            )
            return Tile(data, tile_ref.dtype, tile_ref.shape, unique_sticks)

        # Strided or coord-set path: linearize coords, single read, numpy fancy-index.
        offsets, unique_sticks = MemoryOps._flat_memory_offsets(
            tile_ref.base_ptr, tile_ref.shape, tile_ref.strides, tile_ref.dtype, coords
        )
        span = max(offsets) + 1 if offsets else 1
        flat = mem.read(tile_ref.base_ptr, span, tile_ref.dtype)

        gathered = flat[offsets]
        out_shape = result_shape if result_shape is not None else tile_ref.shape
        data = gathered.reshape(out_shape)

        MemoryOps._write_to_lx(context, data)
        return Tile(data, tile_ref.dtype, out_shape, unique_sticks)

    @staticmethod
    def store(
        context: CoreContext,
        tile: Tile,
        tile_ref: TileRef,
        coords: Optional[List[Tuple[int, ...]]] = None,
    ):
        """Store tile data to HBM or LX.

        - HBM target → DMA write from LX to HBM.
        - LX target  → write directly to LX.

        When *coords* is given (coordinate-set path), scatters tile elements
        to those local coordinates via a read-modify-write on the allocation.
        When *coords* is None, stores the full tile (contiguous or strided).

        A single ``mem.read`` + ``mem.write`` covers the entire footprint;
        no per-element dict scans occur.

        Args:
            context: Core execution context
            tile: Tile value (tensor data) to store
            tile_ref: Tile reference (memref) describing destination
            coords: Optional list of local coordinate tuples to scatter into.
        """
        mem = context.hbm if tile_ref.memory_space == "HBM" else context.lx

        # Fast path: contiguous tile, no coord filtering — single dict-key write.
        if coords is None and MemoryOps._is_contiguous(tile_ref.shape, tile_ref.strides):
            mem.write(tile_ref.base_ptr, tile.data.flatten())
            return

        # Strided or coord-set path: read-modify-write via scatter offsets.
        offsets, _ = MemoryOps._flat_memory_offsets(
            tile_ref.base_ptr, tile_ref.shape, tile_ref.strides, tile_ref.dtype, coords
        )
        span = max(offsets) + 1 if offsets else 1
        flat = mem.read(tile_ref.base_ptr, span, tile_ref.dtype)
        flat[offsets] = tile.data.flatten()
        mem.write(tile_ref.base_ptr, flat)

    @staticmethod
    def distributed_tile_access(
        dist_ref: DistributedTileRef,
        access_shape: Tuple[int, ...],
        base_map: AffineMap,
        indices: List[int],
        access_tile_set: Optional["AffineSet"] = None,
    ) -> DistributedTileRef:
        """Prune and intersect partitions for a given access tile.

        Terms
        -----
        x             : global_base = base_map.eval(indices).  The global origin
                        of the access tile.
        A             : access_tile_set — the affine set on the access tile, in
                        local coords 0..access_shape-1.  If None, treated as the
                        full box [0, access_shape).
        x + A         : the global footprint of the access tile.
        B_i           : partition i's coordinate_set, in global coords.
        C_i           : (x + A) ∩ B_i — the global coords covered by both the
                        access tile and partition i.
        p_i           : min(B_i) — the global origin of partition i.
                        base_ptr covers local element [0, 0], i.e. global p_i.

        Algorithm
        ---------
        For each partition i:
          1. Fast prune via corner test (O(2^ndims)): if no corner of x+A is in
             B_i, the intersection C_i is empty — drop the partition.
          2. Compute C_i = (x + A) ∩ B_i and p_i = min(B_i).
          3. Return a new TileRef with coordinate_set=C_i, partition_origin=p_i.

        distributed_load then uses C_i and p_i directly:
          - load coords into partition i: C_i - p_i  (local offset within base_ptr)
          - output coords in out:         C_i - x    (local offset within access tile)

        Example — 4×4 tensor, P0 covers rows 0..1, P1 covers rows 2..3,
        A = full box (no access_tile_set)::

            # Case 1: indices=[0,0], access_shape=(4,4)  → x=(0,0)
            #   C0 = [0..1]×[0..3],  p0=(0,0)
            #   C1 = [2..3]×[0..3],  p1=(2,0)
            #   distributed_load:
            #     P0: load coords C0-p0=[0..1]×[0..3], out coords C0-x=[0..1]×[0..3]
            #     P1: load coords C1-p1=[0..1]×[0..3], out coords C1-x=[2..3]×[0..3]

            # Case 2: indices=[0,0], access_shape=(2,4)  → x=(0,0)
            #   C0 = [0..1]×[0..3],  p0=(0,0)  → P0 survives
            #   C1 = empty (x+A covers rows 0..1, B1 covers rows 2..3) → P1 pruned

            # Case 3: indices=[1,1], access_shape=(2,2)  → x=(1,1)
            #   C0 = {row 1}×[1..3],  p0=(0,0)
            #     load coords C0-p0 = {(1,1),(1,2),(1,3)}
            #     out  coords C0-x  = {(0,0),(0,1),(0,2)}
            #   C1 = {row 2}×[1..3],  p1=(2,0)
            #     load coords C1-p1 = {(0,1),(0,2),(0,3)}
            #     out  coords C1-x  = {(1,0),(1,1),(1,2)}
        """
        global_base = base_map.eval(indices)
        ndim = len(dist_ref.shape)
        x = global_base

        # Access-set footprint x + A in global coords.  When both A and every
        # B_i are BoxSet (the common case after parse-time lowering), the
        # intersection is a single O(ndim) operation; otherwise we fall back
        # to per-point membership testing.
        xA_box: Optional[BoxSet] = None
        if access_tile_set is None:
            xA_box = BoxSet(lo=tuple(x), hi=tuple(x[d] + access_shape[d] for d in range(ndim)))
        elif isinstance(access_tile_set, BoxSet):
            xA_box = access_tile_set.translate(x)

        # Slow-path membership test for "p ∈ x + A" (local coords A, global
        # footprint x + A).  Used when access_tile_set is an AffineSet.
        def _in_xA(p: Tuple[int, ...]) -> bool:
            if access_tile_set is None:
                return all(0 <= p[d] - x[d] < access_shape[d] for d in range(ndim))
            return access_tile_set.contains(tuple(p[d] - x[d] for d in range(ndim)))

        survivors: List[TileRef] = []
        for part in dist_ref.partitions:
            B_i = part.coordinate_set
            if isinstance(B_i, BoxSet) and xA_box is not None:
                # Fast path: axis-aligned intersection in O(ndim).
                C_i: "Union[BoxSet, List[Tuple[int, ...]]]" = B_i.intersect(xA_box)
                if C_i.is_empty():
                    continue
                p_i = B_i.lower_bounds()
                coordinate_set_out: "Union[BoxSet, List[Tuple[int, ...]]]" = C_i
            else:
                # Slow path: brute-force enumerate B_i and filter by x + A.
                # Triggered when B_i is an AffineSet (non-axis-aligned) or
                # when access_tile_set is an AffineSet (non-axis-aligned A).
                B_i_pts = B_i.enumerate(dist_ref.shape)
                if not B_i_pts:
                    continue
                p_i = tuple(min(pt[d] for pt in B_i_pts) for d in range(ndim))
                C_i_pts = [pt for pt in B_i_pts if _in_xA(pt)]
                if not C_i_pts:
                    continue
                coordinate_set_out = C_i_pts

            survivors.append(TileRef(
                base_ptr=part.base_ptr,
                shape=part.shape,
                strides=part.strides,
                memory_space=part.memory_space,
                dtype=part.dtype,
                coordinate_set=coordinate_set_out,
                partition_origin=p_i,
            ))

        if not survivors:
            raise ValueError(
                f"distributed_tile_access: no partition covers access region "
                f"global_base={global_base} shape={access_shape}"
            )
        return DistributedTileRef(
            partitions=survivors,
            shape=dist_ref.shape,
            dtype=dist_ref.dtype,
            global_base=global_base,
        )

    @staticmethod
    def distributed_load(
        context: CoreContext,
        dist_ref: DistributedTileRef,
        result_shape: Optional[Tuple[int, ...]] = None,
    ) -> Tile:
        """Load from a pre-resolved DistributedTileRef (output of distributed_tile_access).

        Iterates surviving partitions directly. For each partition:
        - Fast path: coordinate_set covers the full partition shape → bulk read,
          rectangular assign into the output array.
        - Slow path: partial coverage → enumerate local coords, gather, scatter.
        """
        ndim = len(dist_ref.shape)
        global_base = dist_ref.global_base or (0,) * ndim
        out_shape = result_shape if result_shape is not None else tuple(dist_ref.shape)
        out = np.zeros(out_shape, dtype=_to_np_dtype(dist_ref.dtype))
        total_unique_sticks = 0

        for part in dist_ref.partitions:
            # coordinate_set is either a BoxSet (distributed fast path) or a
            # pre-enumerated point list (slow path), set by distributed_tile_access.
            cs = part.coordinate_set
            C_i_pts = cs.enumerate() if isinstance(cs, BoxSet) else cs
            p_i = part.partition_origin
            x = global_base

            # load_coords: C_i - p_i  (local offsets within partition memory)
            # out_coords:  C_i - x    (local offsets within output array)
            load_coords = [tuple(pt[d] - p_i[d] for d in range(ndim)) for pt in C_i_pts]
            out_coords  = [tuple(pt[d] - x[d]   for d in range(ndim)) for pt in C_i_pts]

            tile = MemoryOps.load(context, part, coords=load_coords, result_shape=(len(load_coords),))
            total_unique_sticks += tile.unique_sticks or 0

            flat_out = out.reshape(-1)
            out_indices = [sum(c[d] * int(np.prod(out_shape[d+1:])) for d in range(ndim)) for c in out_coords]
            flat_out[out_indices] = tile.data.flatten()

        MemoryOps._write_to_lx(context, out)
        return Tile(out, dist_ref.dtype, out_shape, total_unique_sticks)

    @staticmethod
    def distributed_store(
        context: CoreContext,
        tile: Tile,
        dist_ref: DistributedTileRef,
    ) -> None:
        """Store to a pre-resolved DistributedTileRef (output of distributed_tile_access).

        Symmetric to load_distributed: slice the input tile per partition and
        store each slice to its partition's memory.
        """
        ndim = len(dist_ref.shape)
        global_base = dist_ref.global_base or (0,) * ndim

        for part in dist_ref.partitions:
            # coordinate_set is either a BoxSet (distributed fast path) or a
            # pre-enumerated point list (slow path), set by distributed_tile_access.
            cs = part.coordinate_set
            C_i_pts = cs.enumerate() if isinstance(cs, BoxSet) else cs
            p_i = part.partition_origin
            x = global_base

            load_coords = [tuple(pt[d] - p_i[d] for d in range(ndim)) for pt in C_i_pts]
            out_coords  = [tuple(pt[d] - x[d]   for d in range(ndim)) for pt in C_i_pts]

            flat_in = tile.data.reshape(-1)
            in_indices = [sum(c[d] * int(np.prod(tile.shape[d+1:])) for d in range(ndim)) for c in out_coords]
            part_data = np.array([flat_in[i] for i in in_indices], dtype=_to_np_dtype(part.dtype))
            part_tile = Tile(part_data, part.dtype, (len(load_coords),))
            MemoryOps.store(context, part_tile, part, coords=load_coords)

    @staticmethod
    def indirect_load(
        context: CoreContext,
        iat: "IndirectAccessTile",
        result_shape: Optional[Tuple[int, ...]] = None,
    ) -> Tile:
        """Load data using an indirect access tile (gather pattern).

        Enumerates the variable space, resolves each coordinate tuple
        (direct dims use the variable value, indirect dims look up the
        index in an index memref), then delegates to :meth:`load`.
        """
        vss = iat.variables_space_set
        vso = iat.variables_space_order

        # Enumerate all points in the variable space
        points = vss.enumerate(iat.shape)
        if vso is not None:
            points = [vso.eval(pt) for pt in points]

        # For each point, resolve the actual coordinates in the parent memref
        coords = []
        for pt in points:
            coord = []
            for sub in iat.dim_subscripts:
                if sub["kind"] == "indirect":
                    # Look up the index from the index memref
                    idx_view = iat.index_views[sub["index_view_idx"]]
                    # Compute address into the index tensor
                    idx_coords = tuple(
                        eval_subscript_expr(e, pt) for e in sub["idx_exprs"]
                    )
                    mem = context.hbm if idx_view.memory_space == "HBM" else context.lx
                    offset = sum(c * s for c, s in zip(idx_coords, idx_view.strides))
                    addr = idx_view.base_ptr + offset * _bytes_per_elem(idx_view.dtype)
                    raw = mem.read(addr, 1, idx_view.dtype)
                    coord.append(int(raw[0]))
                elif sub["kind"] == "direct":
                    coord.append(pt[sub["var_index"]])
                elif sub["kind"] == "direct_expr":
                    coord.append(eval_subscript_expr(sub["subscript"], pt))
            coords.append(tuple(coord))

        out_shape = result_shape if result_shape is not None else iat.shape
        return MemoryOps.load(context, iat.parent_ref, coords=coords, result_shape=out_shape)
