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
Core IR data types.

- Tile: Data value backed by a NumPy array (tensor in MLIR)
- TileRef: Memory layout descriptor (memref in MLIR)
- AccessTile: Coordinate access tile referencing a sub-region of a TileRef
- Operation: Single IR operation
- IRFunction / IRModule: Top-level IR containers
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .affine import AffineMap, AffineSet, BoxSet

# Type alias for TileRef.coordinate_set: parsed affine sets lower to BoxSet
# at parse time when axis-aligned; non-box sets stay as AffineSet; the
# distributed fast path stores a BoxSet (C_i = B_i ∩ (x + A)); the
# distributed slow path stores a pre-enumerated list of points.
CoordinateSet = Union[BoxSet, AffineSet, List[Tuple[int, ...]]]


@dataclass
class Tile:
    """Data value backed by a NumPy array.

    Represents an actual tensor of element data (e.g. ``tensor<1024xf16>``
    in MLIR).  Produced by load operations and consumed by compute ops.
    """
    data: np.ndarray
    dtype: str  # see dtypes.SUPPORTED_DTYPES
    shape: Tuple[int, ...]
    # Number of distinct HBM sticks touched by the load that produced this
    # tile.  Set by ``MemoryOps.load`` for all access patterns.  None only
    # for tiles produced by compute ops (not loaded from memory).
    unique_sticks: Optional[int] = None

    def copy(self) -> 'Tile':
        """Create a deep copy of this tile.

        Used by comm ops for message buffers / reduce accumulators.
        When cross-core communication is fixed (gap analysis K1/K2),
        reconsider whether unique_sticks should be recomputed for the
        target device's base_ptr — currently we propagate the source
        value since the memory layout is abstracted away from Tile.
        """
        return Tile(self.data.copy(), self.dtype, self.shape, self.unique_sticks)

    def size_bytes(self) -> int:
        """Return size in bytes."""
        return self.data.nbytes

    @property
    def coalescing_efficiency(self) -> Optional[float]:
        """Ratio of packed traffic to actual stick traffic.

        Defined as ``data.nbytes / (unique_sticks * HBMSimulator.STICK_BYTES)``.
        Ranges from ``bytes_per_elem / STICK_BYTES`` (worst: every
        element on its own stick) to ``1.0`` (best: sticks fully packed).

        Returns ``None`` when ``unique_sticks`` is not set (compute-produced tiles).
        """
        if self.unique_sticks is None or self.unique_sticks == 0:
            return None
        from .memory import HBMSimulator
        return self.data.nbytes / (self.unique_sticks * HBMSimulator.STICK_BYTES)


@dataclass
class TileRef:
    """Memory layout descriptor (does NOT contain data).

    Describes a contiguous region of memory with base pointer, shape,
    strides, and dtype (corresponds to ``memref<...>`` in MLIR).
    """
    base_ptr: int
    shape: Tuple[int, ...]
    strides: List[int]
    memory_space: str  # "HBM" or "LX"
    dtype: str = "f16"
    # coordinate_set is one of three forms, chosen by the path that built this
    # TileRef (see the CoordinateSet alias above):
    #   - BoxSet: axis-aligned set, produced by parse-time lowering and by
    #     the distributed_tile_access fast path (C_i = B_i ∩ (x + A) when all
    #     inputs are boxes).  O(ndim) structural ops.
    #   - AffineSet: non-axis-aligned or symbolic set — parse-time fallback.
    #   - List[Tuple[int,...]]: pre-enumerated C_i points from the slow path
    #     in distributed_tile_access (used when access_tile_set is an
    #     AffineSet, so C_i isn't representable as a single box).
    #   - None: full-box sentinel (load/store contiguous fast path).
    coordinate_set: Optional[CoordinateSet] = None
    partition_origin: Optional[Tuple[int, ...]] = None  # min(B_i) in global coords;
                                                        # set by distributed_tile_access,
                                                        # None on construct_memory_view

    def __post_init__(self):
        valid = ("HBM", "LX")
        if self.memory_space not in valid:
            raise ValueError(
                f"Invalid memory_space {self.memory_space!r}. Must be one of {valid}."
            )

    def size_bytes(self) -> int:
        """Calculate size in bytes."""
        from .dtypes import bytes_per_elem
        return int(np.prod(self.shape) * bytes_per_elem(self.dtype))


@dataclass
class DistributedTileRef:
    """Composite memory layout combining several per-partition TileRefs.

    Produced by ``ktdp.construct_distributed_memory_view``.  Each partition
    is a plain :class:`TileRef` carrying its own ``coordinate_set`` (in
    global coords), ``memory_space``, ``base_ptr``, and ``strides``; this
    wrapper records the global logical shape and dispatches a global
    coordinate to the first partition whose set contains it.

    The op does not allocate or move data — this is bookkeeping only.
    Load/store resolution happens in ``MemoryOps.load_distributed`` /
    ``store_distributed`` (see RFC 0682 §3.3).
    """
    partitions: List[TileRef]
    shape: Tuple[int, ...]        # global logical shape (coordinate_sets use these coords)
    dtype: str
    global_base: Optional[Tuple[int, ...]] = None  # origin of the access tile in global coords;
                                                    # set by distributed_tile_access, None on
                                                    # construct_distributed_memory_view

    def __post_init__(self):
        if not self.partitions:
            raise ValueError("DistributedTileRef requires at least one partition")
        for i, p in enumerate(self.partitions):
            if p.dtype != self.dtype:
                raise ValueError(
                    f"DistributedTileRef partition {i} dtype {p.dtype!r} "
                    f"does not match view dtype {self.dtype!r}"
                )


@dataclass
class AccessTile:
    """Coordinate access tile referencing a sub-region of a TileRef.

    Holds the affine attributes that describe which coordinates of the
    parent memref to access.  Load and store operations use these to
    find the actual memory location.

    ``parent_ref`` may be either a plain :class:`TileRef` or a
    :class:`DistributedTileRef`; routing to a specific partition is
    deferred to load/store time.
    """
    parent_ref: Union[TileRef, "DistributedTileRef"]
    shape: Tuple[int, ...]
    base_map: AffineMap                                  # always present; synthesized as identity if absent in MLIR
    coordinate_set: Optional[AffineSet] = None     # parsed access_tile_set; None if omitted
    coordinate_order: Optional[AffineMap] = None   # parsed access_tile_order; None if omitted


@dataclass
class IndirectAccessTile:
    """Indirect access tile for gather/scatter patterns.

    Each dimension of the target tensor is indexed either:
    - Directly via an intermediate variable value, or
    - Indirectly via a lookup into an index memory view.
    """
    parent_ref: TileRef                         # primary memory view (e.g. X)
    shape: Tuple[int, ...]                      # output access tile shape
    dim_subscripts: List[Dict[str, Any]]        # per-dim descriptor (kind, var_indices, etc.)
    index_views: List[TileRef]                  # index memrefs for indirect dims
    variables_space_set: AffineSet              # domain of intermediate variables
    variables_space_order: Optional[AffineMap]  # iteration order; None = default


@dataclass
class Operation:
    """Single IR operation with result, operands, attributes, and optional regions."""
    result: Optional[str]  # Result SSA value name (e.g., "%x")
    op_type: str  # Operation type (e.g., "ktdp.get_compute_tile_id", "arith.addf")
    operands: List[str]  # Operand SSA value names
    attributes: Dict[str, Any]  # Operation attributes
    result_type: Optional[str]  # Result type string
    regions: List[List['Operation']] = field(default_factory=list)  # For control flow

    def __repr__(self):
        if self.result:
            return f"{self.result} = {self.op_type}({', '.join(self.operands)})"
        else:
            return f"{self.op_type}({', '.join(self.operands)})"


def _iter_ops(operations: List['Operation']):
    """Yield all operations recursively, descending into regions."""
    for op in operations:
        yield op
        for region in op.regions:
            yield from _iter_ops(region)


@dataclass
class IRFunction:
    """IR function with arguments, body operations, and grid shape."""
    name: str
    arguments: List[Tuple[str, str]]  # [(name, type), ...]
    operations: List[Operation]
    grid: Tuple[int, int, int]  # Grid shape from function attributes
    return_type: Optional[str] = None
    tensor_sizes: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    @property
    def arg_names(self) -> List[str]:
        return [name.lstrip("%") for name, _ in self.arguments]

    def __post_init__(self):
        for op in _iter_ops(self.operations):
            if op.op_type == "ktdp.construct_memory_view" and op.operands:
                arg_name = op.operands[0].lstrip("%")
                if arg_name not in self.tensor_sizes:
                    self.tensor_sizes[arg_name] = {
                        "shape": tuple(op.attributes.get("shape", ())),
                        "dtype": op.attributes.get("dtype", "f16"),
                    }

    def __repr__(self):
        tensors = ", ".join(
            f"{name}: {info['shape']}x{info['dtype']}"
            for name, info in self.tensor_sizes.items()
        )
        return f"IRFunction({self.name}, grid={self.grid}, tensors=[{tensors}])"


@dataclass
class IRModule:
    """Top-level IR module containing functions."""
    functions: Dict[str, IRFunction] = field(default_factory=dict)
    # Named attribute aliases declared at module scope before module {}.
    # Maps "#name" -> verbatim value string (e.g. "#X_coord_set" -> "affine_set<...>").
    # Populated by the parser; used by construct handlers to resolve references.
    aliases: Dict[str, str] = field(default_factory=dict)

    def get_function(self, name: str) -> IRFunction:
        """Get function by name."""
        if name not in self.functions:
            raise ValueError(f"Function '{name}' not found in module")
        return self.functions[name]

    def add_function(self, func: IRFunction):
        """Add function to module."""
        self.functions[func.name] = func
