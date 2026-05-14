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

"""Unit tests for ktdp.construct_distributed_memory_view.

Structure
---------
- test_distributed_view_copy_rfc   : RFC §C.3 reference example (xfail, per-core LX gap)
- test_distributed_copy            : parametrized table of 2-partition copy cases

The parametrized suite covers all combinations of:
  - memory spaces: HBM/HBM, HBM/LX, LX/HBM
  - partition strides: row-major [R,1] and column-packed [1,C]
  - access shapes: full tile, partial tile (one partition pruned), sub-tile
  - access indices: zero and non-zero (sub-tile spanning both partitions)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pytest

from ktir_cpu import KTIRInterpreter
from conftest import get_test_params


# ---------------------------------------------------------------------------
# Shared memory helpers
# ---------------------------------------------------------------------------

def _write_strided(mem, base_ptr: int, block: np.ndarray, strides: List[int]):
    """Write *block* (f16) into *mem* at *base_ptr* using *strides* (element units).

    Element (i, j) lands at byte offset ``(base_ptr + (i*strides[0] + j*strides[1])) * 2``.
    Holes from non-contiguous layouts are left as zero.
    """
    assert block.dtype == np.float16
    ndim = block.ndim
    assert len(strides) == ndim
    coords = np.stack(np.meshgrid(*[np.arange(s) for s in block.shape], indexing='ij'), axis=-1)
    coords = coords.reshape(-1, ndim)
    offsets = coords @ np.array(strides, dtype=np.int64)
    span = int(offsets.max()) + 1 if offsets.size else 1
    buf = np.zeros(span, dtype=np.float16)
    buf[offsets] = block.flatten()
    mem.write(base_ptr, buf)


def _get_mem(interp, space: str):
    """Return the memory object for *space* ('HBM' or 'LX')."""
    return interp.memory.hbm if space == "HBM" else interp.memory.get_lx(0)


# ---------------------------------------------------------------------------
# MLIR builder
# ---------------------------------------------------------------------------

def _affine_set_rows(r0: int, r1: int, c0: int, c1: int) -> str:
    """Return affine_set string covering rows r0..r1, cols c0..c1 (inclusive)."""
    return (
        f"affine_set<(d0, d1) : "
        f"(d0 - {r0} >= 0, -{r0} + {r1} - d0 + {r0} >= 0, "
        f"d1 - {c0} >= 0, -{c0} + {c1} - d1 + {c0} >= 0)>"
    ).replace("(d0 - 0 >= 0", "(d0 >= 0").replace("(d1 - 0 >= 0", "(d1 >= 0")


def _set_rows(r0: int, r1: int, ncols: int) -> str:
    """affine_set covering rows [r0, r1] and all cols [0, ncols-1]."""
    row_lo = f"d0 - {r0} >= 0" if r0 > 0 else "d0 >= 0"
    row_hi = f"-d0 + {r1} >= 0"
    col_lo = "d1 >= 0"
    col_hi = f"-d1 + {ncols - 1} >= 0"
    return f"affine_set<(d0, d1) : ({row_lo}, {row_hi}, {col_lo}, {col_hi})>"


def _set_box(r0: int, r1: int, c0: int, c1: int) -> str:
    """affine_set covering [r0,r1] x [c0,c1] (all inclusive)."""
    parts = []
    parts.append(f"d0 - {r0} >= 0" if r0 > 0 else "d0 >= 0")
    parts.append(f"-d0 + {r1} >= 0")
    parts.append(f"d1 - {c0} >= 0" if c0 > 0 else "d1 >= 0")
    parts.append(f"-d1 + {c1} >= 0")
    return f"affine_set<(d0, d1) : ({', '.join(parts)})>"


@dataclass
class PartitionSpec:
    """Describes one rectangular partition of the distributed view.

    Attributes:
        rows         : (first_row, last_row) inclusive in global coords
        cols         : (first_col, last_col) inclusive in global coords
        memory_space : "HBM" or "LX"
        strides      : element strides [row_stride, col_stride]
        base_ptr     : byte address of element [0,0] of this partition
    """
    rows: Tuple[int, int]
    cols: Tuple[int, int]
    memory_space: str
    strides: List[int]
    base_ptr: int

    @property
    def nrows(self) -> int:
        return self.rows[1] - self.rows[0] + 1

    @property
    def ncols(self) -> int:
        return self.cols[1] - self.cols[0] + 1


@dataclass
class DistCopySpec:
    """Full specification for a 2-partition distributed copy test.

    The kernel loads an access tile from distributed A and stores it to
    contiguous HBM output B.  Both are 2-D.

    Attributes:
        global_shape  : logical shape of the full distributed tensor
        p0, p1        : partition specs (disjoint rectangular regions in global coords)
        access_shape  : shape of the access tile
        indices       : [row, col] base indices for construct_access_tile
        out_ptr       : byte address of the B output on HBM
        id            : short human-readable label used in pytest ids
    """
    global_shape: Tuple[int, int]
    p0: PartitionSpec
    p1: PartitionSpec
    access_shape: Tuple[int, int]
    indices: List[int]
    out_ptr: int
    id: str


def _build_mlir(spec: DistCopySpec) -> str:
    """Generate a single-function MLIR module from *spec*."""
    G = spec.global_shape
    p0, p1 = spec.p0, spec.p1
    ac = spec.access_shape
    idx = spec.indices

    p0_set = _set_box(p0.rows[0], p0.rows[1], p0.cols[0], p0.cols[1])
    p1_set = _set_box(p1.rows[0], p1.rows[1], p1.cols[0], p1.cols[1])
    ac_set = _set_box(0, ac[0] - 1, 0, ac[1] - 1)

    idx_decls = "\n".join(
        f"    %idx{i} = arith.constant {v} : index" for i, v in enumerate(idx)
    )
    idx_refs = ", ".join(f"%idx{i}" for i in range(len(idx)))

    p0_ms = f"#ktdp.spyre_memory_space<{p0.memory_space}>"
    p1_ms = f"#ktdp.spyre_memory_space<{p1.memory_space}>"

    return f"""
#P0_set   = {p0_set}
#P1_set   = {p1_set}
#ac_set   = {ac_set}
#identity = affine_map<(d0, d1) -> (d0, d1)>
module {{
  func.func @dist_copy() {{
    %c0 = arith.constant 0 : index
{idx_decls}
    %A0_addr = arith.constant {p0.base_ptr} : index
    %A1_addr = arith.constant {p1.base_ptr} : index
    %B_addr  = arith.constant {spec.out_ptr} : index

    %A0 = ktdp.construct_memory_view %A0_addr, sizes: [{p0.nrows}, {p0.ncols}], strides: [{p0.strides[0]}, {p0.strides[1]}] {{
        coordinate_set = #P0_set, memory_space = {p0_ms}
    }} : memref<{p0.nrows}x{p0.ncols}xf16>

    %A1 = ktdp.construct_memory_view %A1_addr, sizes: [{p1.nrows}, {p1.ncols}], strides: [{p1.strides[0]}, {p1.strides[1]}] {{
        coordinate_set = #P1_set, memory_space = {p1_ms}
    }} : memref<{p1.nrows}x{p1.ncols}xf16>

    %A = ktdp.construct_distributed_memory_view
        (%A0, %A1 : memref<{p0.nrows}x{p0.ncols}xf16>, memref<{p1.nrows}x{p1.ncols}xf16>)
        : memref<{G[0]}x{G[1]}xf16>

    %B = ktdp.construct_memory_view %B_addr, sizes: [{ac[0]}, {ac[1]}], strides: [{ac[1]}, 1] {{
        coordinate_set = #ac_set, memory_space = #ktdp.spyre_memory_space<HBM>
    }} : memref<{ac[0]}x{ac[1]}xf16>

    %A_at = ktdp.construct_access_tile %A[{idx_refs}] {{
        access_tile_set = #ac_set, access_tile_order = #identity
    }} : memref<{G[0]}x{G[1]}xf16> -> !ktdp.access_tile<{ac[0]}x{ac[1]}xindex>

    %B_at = ktdp.construct_access_tile %B[%c0, %c0] {{
        access_tile_set = #ac_set, access_tile_order = #identity
    }} : memref<{ac[0]}x{ac[1]}xf16> -> !ktdp.access_tile<{ac[0]}x{ac[1]}xindex>

    %data = ktdp.load %A_at : !ktdp.access_tile<{ac[0]}x{ac[1]}xindex> -> tensor<{ac[0]}x{ac[1]}xf16>
    ktdp.store %data, %B_at : tensor<{ac[0]}x{ac[1]}xf16>, !ktdp.access_tile<{ac[0]}x{ac[1]}xindex>
    return
  }}
}}
"""


# ---------------------------------------------------------------------------
# Test case table
# ---------------------------------------------------------------------------
#
# Reference tensor: np.arange(GR*GC, dtype=f16).reshape(GR, GC)
# All cases use a 4×4 global tensor unless noted.
#
# Three partition layouts are used:
#   Row-band    : P0 rows 0..r, P1 rows r+1..3, each spanning all 4 cols
#   Col-band    : P0 cols 0..c, P1 cols c+1..3, each spanning all 4 rows
#   Mixed       : P0 is a 2-row block, P1 is a 2-row×2-col block beside it
#
# Memory pointer map (non-overlapping, 4096-byte spacing):
_P0_PTR  = 0
_P1_PTR  = 4096
_OUT_PTR = 8192

_CASES: List[DistCopySpec] = [
    # -----------------------------------------------------------------------
    # Row-band partitioning
    #
    #      c0   c1   c2   c3
    # r0 [ ----  P0  ---- ]
    # r1 [ ----  P0  ---- ]   P0: rows 0..1, all cols
    # r2 [ ----  P1  ---- ]   P1: rows 2..3, all cols
    # r3 [ ----  P1  ---- ]
    # -----------------------------------------------------------------------

    # Case 1: full 4×4 access — both partitions loaded
    #
    #      c0   c1   c2   c3
    # r0 [ ░░░  P0  ░░░░░ ]   ░ = access tile
    # r1 [ ░░░  P0  ░░░░░ ]
    # r2 [ ░░░  P1  ░░░░░ ]
    # r3 [ ░░░  P1  ░░░░░ ]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 1), cols=(0, 3), memory_space="HBM", strides=[4, 1], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(2, 3), cols=(0, 3), memory_space="HBM", strides=[4, 1], base_ptr=_P1_PTR),
        access_shape=(4, 4), indices=[0, 0], out_ptr=_OUT_PTR,
        id="row_hbm_hbm_full",
    ),
    # Case 2: 2×4 access at [0,0] — P1 pruned (access doesn't reach rows 2..3)
    #
    #      c0   c1   c2   c3
    # r0 [ ░░░  P0  ░░░░░ ]   ░ = access tile
    # r1 [ ░░░  P0  ░░░░░ ]
    # r2 [ ----  P1  ---- ]   P1 pruned
    # r3 [ ----  P1  ---- ]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 1), cols=(0, 3), memory_space="HBM", strides=[4, 1], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(2, 3), cols=(0, 3), memory_space="HBM", strides=[4, 1], base_ptr=_P1_PTR),
        access_shape=(2, 4), indices=[0, 0], out_ptr=_OUT_PTR,
        id="row_hbm_hbm_partial_p1_pruned",
    ),
    # Case 3: 2×2 sub-tile at [1,1] — spans both partitions (nonzero indices)
    #
    #      c0   c1   c2   c3
    # r0 [ ----  P0  ---- ]
    # r1 [  P0 | ░░░ |    ]   ░ = access tile  global[1:3, 1:3]
    # r2 [  P1 | ░░░ |    ]
    # r3 [ ----  P1  ---- ]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 1), cols=(0, 3), memory_space="HBM", strides=[4, 1], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(2, 3), cols=(0, 3), memory_space="HBM", strides=[4, 1], base_ptr=_P1_PTR),
        access_shape=(2, 2), indices=[1, 1], out_ptr=_OUT_PTR,
        id="row_hbm_hbm_subtile_nonzero",
    ),
    # Case 4: full 4×4, P1 in LX with col-packed strides [1,4]
    #
    #      c0   c1   c2   c3
    # r0 [ ░░  P0(HBM)  ░ ]   ░ = access tile
    # r1 [ ░░  P0(HBM)  ░ ]   P0 row-major:  elem[r,c] at offset r*4+c
    # r2 [ ░░  P1(LX)   ░ ]   P1 col-packed: elem[r,c] at offset r+c*2
    # r3 [ ░░  P1(LX)   ░ ]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 1), cols=(0, 3), memory_space="HBM", strides=[4, 1], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(2, 3), cols=(0, 3), memory_space="LX",  strides=[1, 4], base_ptr=_P1_PTR),
        access_shape=(4, 4), indices=[0, 0], out_ptr=_OUT_PTR,
        id="row_hbm_lx_col_packed_full",
    ),
    # Case 5: full 4×4, P0 in LX col-packed, P1 in HBM — reversed memory spaces
    #
    #      c0   c1   c2   c3
    # r0 [ ░░  P0(LX)   ░ ]   P0 col-packed: elem[r,c] at offset r+c*2
    # r1 [ ░░  P0(LX)   ░ ]   P1 row-major:  elem[r,c] at offset r*4+c
    # r2 [ ░░  P1(HBM)  ░ ]
    # r3 [ ░░  P1(HBM)  ░ ]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 1), cols=(0, 3), memory_space="LX",  strides=[1, 4], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(2, 3), cols=(0, 3), memory_space="HBM", strides=[4, 1], base_ptr=_P1_PTR),
        access_shape=(4, 4), indices=[0, 0], out_ptr=_OUT_PTR,
        id="row_lx_hbm_col_packed_full",
    ),
    # Case 6: 2×2 sub-tile at [1,1], P1 in LX col-packed — mixed spaces + nonzero indices
    #
    #      c0   c1   c2   c3
    # r0 [ ----  P0(HBM)  ---- ]
    # r1 [  P0(HBM) | ░░░ |    ]   ░ = access tile  global[1:3, 1:3]
    # r2 [  P1(LX)  | ░░░ |    ]
    # r3 [ ----  P1(LX)   ---- ]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 1), cols=(0, 3), memory_space="HBM", strides=[4, 1], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(2, 3), cols=(0, 3), memory_space="LX",  strides=[1, 4], base_ptr=_P1_PTR),
        access_shape=(2, 2), indices=[1, 1], out_ptr=_OUT_PTR,
        id="row_hbm_lx_subtile_nonzero",
    ),
    # Case 7: unequal row bands — P0 is 1 row, P1 is 3 rows; full 4×4 access
    #
    #      c0   c1   c2   c3
    # r0 [ ░░░  P0  ░░░░░ ]   P0: 1 row
    # r1 [ ░░░  P1  ░░░░░ ]   P1: 3 rows
    # r2 [ ░░░  P1  ░░░░░ ]
    # r3 [ ░░░  P1  ░░░░░ ]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 0), cols=(0, 3), memory_space="HBM", strides=[4, 1], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(1, 3), cols=(0, 3), memory_space="HBM", strides=[4, 1], base_ptr=_P1_PTR),
        access_shape=(4, 4), indices=[0, 0], out_ptr=_OUT_PTR,
        id="row_hbm_hbm_unequal_full",
    ),
    # Case 8: unequal row bands, 2×4 at [1,0] — P0 pruned (access starts at row 1)
    #
    #      c0   c1   c2   c3
    # r0 [ ----  P0  ---- ]   P0 pruned
    # r1 [ ░░░  P1  ░░░░░ ]   ░ = access tile
    # r2 [ ░░░  P1  ░░░░░ ]
    # r3 [ ----  P1  ---- ]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 0), cols=(0, 3), memory_space="HBM", strides=[4, 1], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(1, 3), cols=(0, 3), memory_space="HBM", strides=[4, 1], base_ptr=_P1_PTR),
        access_shape=(2, 4), indices=[1, 0], out_ptr=_OUT_PTR,
        id="row_hbm_hbm_unequal_partial_p0_pruned",
    ),

    # -----------------------------------------------------------------------
    # Col-band partitioning
    #
    #      c0   c1 | c2   c3
    # r0 [  --  P0  |  --  P1  ]
    # r1 [  --  P0  |  --  P1  ]   P0: all rows, cols 0..1
    # r2 [  --  P0  |  --  P1  ]   P1: all rows, cols 2..3
    # r3 [  --  P0  |  --  P1  ]
    # -----------------------------------------------------------------------

    # Case 9: full 4×4 access — both col partitions loaded
    #
    #      c0   c1 | c2   c3
    # r0 [ ░░░  P0 | ░░░  P1 ]   ░ = access tile
    # r1 [ ░░░  P0 | ░░░  P1 ]
    # r2 [ ░░░  P0 | ░░░  P1 ]
    # r3 [ ░░░  P0 | ░░░  P1 ]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 3), cols=(0, 1), memory_space="HBM", strides=[2, 1], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(0, 3), cols=(2, 3), memory_space="HBM", strides=[2, 1], base_ptr=_P1_PTR),
        access_shape=(4, 4), indices=[0, 0], out_ptr=_OUT_PTR,
        id="col_hbm_hbm_full",
    ),
    # Case 10: 4×2 access at [0,0] — P1 pruned (access stays in left cols 0..1)
    #
    #      c0   c1 | c2   c3
    # r0 [ ░░░  P0 |  --  P1 ]   P1 pruned
    # r1 [ ░░░  P0 |  --  P1 ]
    # r2 [ ░░░  P0 |  --  P1 ]
    # r3 [ ░░░  P0 |  --  P1 ]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 3), cols=(0, 1), memory_space="HBM", strides=[2, 1], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(0, 3), cols=(2, 3), memory_space="HBM", strides=[2, 1], base_ptr=_P1_PTR),
        access_shape=(4, 2), indices=[0, 0], out_ptr=_OUT_PTR,
        id="col_hbm_hbm_partial_p1_pruned",
    ),
    # Case 11: 2×2 sub-tile at [1,1] — straddles col boundary (col 1 in P0, col 2 in P1)
    #
    #      c0   c1 | c2   c3
    # r0 [  --  P0 |  --  P1 ]
    # r1 [  P0 |░░ | ░░| P1  ]   ░ = access tile  global[1:3, 1:3]
    # r2 [  P0 |░░ | ░░| P1  ]
    # r3 [  --  P0 |  --  P1 ]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 3), cols=(0, 1), memory_space="HBM", strides=[2, 1], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(0, 3), cols=(2, 3), memory_space="HBM", strides=[2, 1], base_ptr=_P1_PTR),
        access_shape=(2, 2), indices=[1, 1], out_ptr=_OUT_PTR,
        id="col_hbm_hbm_subtile_nonzero",
    ),
    # Case 12: full 4×4, P0 HBM row-major, P1 LX col-packed — mixed spaces, col-band
    #
    #      c0     c1   |   c2     c3
    # r0 [ ░  P0(HBM) ░ | ░  P1(LX) ░ ]   P0 row-major:  elem[r,c] at offset r*2+c
    # r1 [ ░  P0(HBM) ░ | ░  P1(LX) ░ ]   P1 col-packed: elem[r,c] at offset r+c*4
    # r2 [ ░  P0(HBM) ░ | ░  P1(LX) ░ ]
    # r3 [ ░  P0(HBM) ░ | ░  P1(LX) ░ ]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 3), cols=(0, 1), memory_space="HBM", strides=[2, 1], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(0, 3), cols=(2, 3), memory_space="LX",  strides=[1, 4], base_ptr=_P1_PTR),
        access_shape=(4, 4), indices=[0, 0], out_ptr=_OUT_PTR,
        id="col_hbm_lx_col_packed_full",
    ),

    # -----------------------------------------------------------------------
    # Mixed layout: P0 is a tall left block (4 rows × 2 cols),
    #               P1 is a small bottom-right block (2 rows × 2 cols).
    # Top-right corner (rows 0..1, cols 2..3) is uncovered — access tiles
    # in these cases are designed to avoid it.
    #
    #      c0   c1 | c2   c3
    # r0 [  --  P0 |  --  -- ]
    # r1 [  --  P0 |  --  -- ]   P0: rows 0..3, cols 0..1
    # r2 [  --  P0 |  --  P1 ]   P1: rows 2..3, cols 2..3
    # r3 [  --  P0 |  --  P1 ]
    # -----------------------------------------------------------------------

    # Case 13: 4×2 access at [0,0] — spans full left block P0; P1 pruned
    #
    #      c0   c1 | c2   c3
    # r0 [ ░░░  P0 |  --  -- ]
    # r1 [ ░░░  P0 |  --  -- ]
    # r2 [ ░░░  P0 |  --  P1 ]
    # r3 [ ░░░  P0 |  --  P1 ]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 3), cols=(0, 1), memory_space="HBM", strides=[2, 1], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(2, 3), cols=(2, 3), memory_space="HBM", strides=[2, 1], base_ptr=_P1_PTR),
        access_shape=(4, 2), indices=[0, 0], out_ptr=_OUT_PTR,
        id="mixed_left_block_only",
    ),
    # Case 14: 2×2 at [2,0] — bottom-left corner, only P0 contributes; P1 pruned
    #
    #      c0   c1 | c2   c3
    # r0 [  --  P0 |  --  -- ]
    # r1 [  --  P0 |  --  -- ]
    # r2 [ ░░░  P0 |  --  P1 ]   ░ = access tile  global[2:4, 0:2]
    # r3 [ ░░░  P0 |  --  P1 ]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 3), cols=(0, 1), memory_space="HBM", strides=[2, 1], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(2, 3), cols=(2, 3), memory_space="HBM", strides=[2, 1], base_ptr=_P1_PTR),
        access_shape=(2, 2), indices=[2, 0], out_ptr=_OUT_PTR,
        id="mixed_bottom_left_only",
    ),
    # Case 15: 2×2 at [2,0], P0 in LX col-packed, P1 in HBM — mixed spaces, mixed layout
    #
    #      c0   c1 | c2   c3
    # r0 [  -- P0(LX) |  --  --     ]
    # r1 [  -- P0(LX) |  --  --     ]
    # r2 [ ░░ P0(LX) ░|  --  P1(HBM)]   ░ = access tile  global[2:4, 0:2]
    # r3 [ ░░ P0(LX) ░|  --  P1(HBM)]
    DistCopySpec(
        global_shape=(4, 4),
        p0=PartitionSpec(rows=(0, 3), cols=(0, 1), memory_space="LX",  strides=[1, 4], base_ptr=_P0_PTR),
        p1=PartitionSpec(rows=(2, 3), cols=(2, 3), memory_space="HBM", strides=[2, 1], base_ptr=_P1_PTR),
        access_shape=(2, 2), indices=[2, 0], out_ptr=_OUT_PTR,
        id="mixed_lx_left_hbm_right_bottom_only",
    ),
]


def _seed_and_run(spec: DistCopySpec) -> Tuple[np.ndarray, np.ndarray]:
    """Seed memory from spec, execute the generated kernel, return (expected, actual).

    The reference tensor is np.arange(16, dtype=f16).reshape(4,4).
    The expected slice is full[indices[0]:indices[0]+access_shape[0],
                               indices[1]:indices[1]+access_shape[1]].
    """
    full = np.arange(16, dtype=np.float16).reshape(spec.global_shape)
    mlir = _build_mlir(spec)
    interp = KTIRInterpreter()
    interp.load(mlir)
    _orig = interp._prepare_execution

    p0, p1 = spec.p0, spec.p1
    ac = spec.access_shape
    idx = spec.indices

    def _prepare_and_seed(grid_shape):
        _orig(grid_shape)
        p0_block = full[p0.rows[0]:p0.rows[1] + 1, p0.cols[0]:p0.cols[1] + 1]
        p1_block = full[p1.rows[0]:p1.rows[1] + 1, p1.cols[0]:p1.cols[1] + 1]
        p0_mem = _get_mem(interp, p0.memory_space)
        p1_mem = _get_mem(interp, p1.memory_space)
        _write_strided(p0_mem, p0.base_ptr, p0_block.copy(), p0.strides)
        _write_strided(p1_mem, p1.base_ptr, p1_block.copy(), p1.strides)
        interp.memory.hbm.write(spec.out_ptr, np.zeros(ac[0] * ac[1], dtype=np.float16))

    interp._prepare_execution = _prepare_and_seed
    interp.execute_function("dist_copy")

    r0, c0 = idx[0], idx[1]
    expected = full[r0:r0 + ac[0], c0:c0 + ac[1]]
    n_out = ac[0] * ac[1]
    actual = interp.memory.hbm.read(spec.out_ptr, n_out, "f16").reshape(ac)
    return expected, actual


# ---------------------------------------------------------------------------
# RFC example (xfail: per-core LX routing not yet implemented)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason="Interpreter gap: LX memory is not per-core; simulator ignores core=N "
           "routing so LX partitions from different cores share one scratchpad and "
           "reads return zeros instead of the seeded data.",
    strict=True,
)
@pytest.mark.parametrize("path,func_name,entry", get_test_params("distributed_view_copy"))
def test_distributed_view_copy_rfc(path, func_name, entry):
    """construct_distributed_memory_view — RFC §C.3 example file.

    A is a 192×64 logical tensor distributed across three regions:
      A_HBM (96×64,  HBM,          row-major strides [64, 1])  rows   0..95
      A_LX0 (32×64,  LX core=0, col-packed strides [1, 64])   rows  96..127
      A_LX1 (64×64,  LX core=1,  row-major strides [64, 1])   rows 128..191
    The kernel copies A into contiguous HBM output B, also 192×64.
    """
    interp = KTIRInterpreter()
    interp.load(path)
    _orig = interp._prepare_execution

    def _prepare_and_seed(grid_shape):
        _orig(grid_shape)
        hbm = interp.memory.hbm
        lx0 = interp.memory.get_lx(0)
        full = np.arange(192 * 64, dtype=np.float16).reshape(192, 64)
        hbm.write(0, full[0:96, :].flatten())
        _write_strided(lx0, 12288, full[96:128, :].copy(), strides=[1, 64])
        lx0.write(16384, full[128:192, :].flatten())
        hbm.write(24576, np.zeros(192 * 64, dtype=np.float16))

    interp._prepare_execution = _prepare_and_seed
    interp.execute_function(func_name)

    expected = np.arange(192 * 64, dtype=np.float16).reshape(192, 64)
    b = interp.memory.hbm.read(24576, 192 * 64, "f16").reshape(192, 64)
    np.testing.assert_array_equal(b, expected)


# ---------------------------------------------------------------------------
# Parametrized 2-partition copy suite
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", _CASES, ids=[c.id for c in _CASES])
def test_distributed_copy(spec: DistCopySpec):
    """2-partition distributed copy — data-correctness check.

    Generates a single-function MLIR kernel from *spec*, seeds the partitions
    in the appropriate memory spaces and strides, runs the kernel, and asserts
    that the output matches the corresponding slice of the reference tensor.
    """
    expected, actual = _seed_and_run(spec)
    np.testing.assert_array_equal(actual, expected)
