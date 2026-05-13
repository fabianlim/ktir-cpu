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
Affine map and integer-set value objects.

Types
-----
AffineMap — represents ``affine_map<(d0,...) -> (e0,...)>``.  A function
            from dim values to output expressions.  ``eval`` delegates to
            ``parser_ast``.

AffineSet — represents ``affine_set<(d0,...) : (c0 >= 0, ...)>``.  The
            general polyhedral integer set: ``{p : all c_k(p) >= 0}``.
            ``contains`` / ``enumerate`` / ``is_full`` walk the constraint
            AST.  ``enumerate(shape)`` brute-forces the bounding box and
            tests every candidate against every constraint, so it is
            O(∏ shape · n_constraints).

BoxSet —    a *special case* of ``AffineSet``: the axis-aligned integer
            hyperrectangle ``{p : lo[d] <= p[d] < hi[d]}``.  Any ``BoxSet``
            is representable as an ``AffineSet`` whose constraints are
            exactly the per-axis inequalities ``d_i - lo[i] >= 0`` and
            ``-d_i + hi[i] - 1 >= 0`` — but the box form carries the
            structure explicitly, so ``contains``, ``enumerate``,
            ``is_empty``, ``is_full``, ``lower_bounds``, ``translate``, and
            ``intersect`` are all O(ndim) with no AST walk.  Sibling to
            ``AffineSet`` the same way MLIR's concept of an axis-aligned
            ``IntegerSet`` is still an ``IntegerSet`` — naming ends in
            ``Set`` to distinguish from ``AffineMap`` (which describes
            functions, not point sets).

Relationship between AffineSet and BoxSet
-----------------------------------------
``BoxSet`` is the axis-aligned specialisation of ``AffineSet``.
Conceptually every ``BoxSet`` *is* an ``AffineSet``; in code they are
peer dataclasses under a ``Union``, NOT a class hierarchy.  The reason is
performance: fast paths must be visible at each call site via
``isinstance`` dispatch.  Forcing a shared method surface (so every call
site calls e.g. ``.enumerate``) historically produced silent slow paths
when the union degraded to the ``AffineSet`` branch.  Mixed-type
operations — e.g. ``BoxSet.intersect(aset: AffineSet)`` — raise
``TypeError``; there is no auto-promotion.

``parse_affine_set`` lowers axis-aligned, fully-pinned sets to ``BoxSet``
at parse time (see ``BoxSet.try_from_affine_set``); other sets stay as
``AffineSet``.  ``parse_affine_set_raw`` skips the lowering and is used by
tests that inspect the ``AffineSet`` AST directly.

Which op uses which class
-------------------------
- ``ktdp.construct_memory_view``:
    ``coordinate_set`` → ``BoxSet`` (parse-time lowering) or ``AffineSet``
    fallback.  Stored in ``TileRef.coordinate_set``.

- ``ktdp.construct_distributed_memory_view``:
    per-partition ``coordinate_set`` → same as above; each partition's
    ``TileRef.coordinate_set`` is normally a ``BoxSet`` after lowering.

- ``ktdp.construct_access_tile`` + ``distributed_tile_access``:
    Fast path (both ``B_i`` and ``A`` are ``BoxSet``): computes
    ``C_i = B_i ∩ (x + A)`` in O(ndim) via ``translate`` + ``intersect``,
    stores ``C_i`` as ``BoxSet`` in the survivor's ``coordinate_set``.
    Slow path (either side is ``AffineSet``): brute-force ``B_i.enumerate``
    and filter; stores the enumerated ``List[Tuple[int, ...]]``.

- ``ktdp.load`` / ``ktdp.store``:
    Call ``css.enumerate(access_tile.shape)`` where ``css`` is either
    ``BoxSet`` or ``AffineSet``.  ``BoxSet.enumerate`` accepts ``shape``
    for signature parity with ``AffineSet`` (it sanity-checks
    ``hi <= shape`` componentwise but is otherwise self-bounded).

- ``ktdp.construct_indirect_access_tile`` + ``indirect_load``:
    Same ``enumerate(shape)`` shape as above on the variables-space set.

Replacements needed at a call site when adding a new op
-------------------------------------------------------
Reading a ``coordinate_set``:
  * If you only need to iterate points: call ``css.enumerate(shape)`` — works
    uniformly on both branches.
  * If you need structural facts (lower bounds, is-empty, intersection with
    another box): ``isinstance(css, BoxSet)`` and use ``lower_bounds`` /
    ``is_empty`` / ``intersect`` directly.  Do NOT fall through to
    ``AffineSet`` for these — ``AffineSet`` has no structural form.
  * Mixed-type intersection (``BoxSet`` with ``AffineSet``) is not
    supported.  If you need it, materialise one side to a point list and
    filter via ``contains``.

Storing a ``coordinate_set``:
  * Prefer ``BoxSet`` whenever the shape is an axis-aligned box.
  * ``List[Tuple[int, ...]]`` is acceptable for pre-enumerated points
    (``distributed_tile_access`` slow path); keep this path when the source
    set is a non-axis-aligned ``AffineSet``.
  * Never store an ``AffineSet`` that *could* have been a ``BoxSet`` —
    downstream call sites can't rediscover the box structure cheaply.

Out of scope for this module
----------------------------
- Non-convex regions (unions of boxes).
- Symbolic (SSA-valued) bounds — symbolic ``AffineSet``s stay on the
  ``AffineSet`` branch; ``try_from_affine_set`` rejects them.
- Deriving axis-aligned bounds from an existing ``AffineSet``'s constraint
  AST to drop the ``shape`` argument from ``enumerate`` on both types.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    # Avoid circular import at runtime; parser_ast imports nothing from here.
    from .parser_ast import _Node


@dataclass(frozen=True)
class AffineMap:
    """Parsed affine_map<(d0,...) -> (e0,...)>.

    Attributes:
        n_dims:  number of input dimension variables (d0, d1, ...)
        exprs:   tuple of AST nodes, one per output dimension
        source:  original verbatim string (for debugging / round-trip)
    """
    n_dims: int
    exprs: Tuple["_Node", ...]
    source: str

    def eval(self, dims: Sequence[int]) -> Tuple[int, ...]:
        """Return the output tuple for the given dimension values.

        Delegates to ``parser_ast.eval_affine_map``.
        """
        from .parser_ast import eval_affine_map
        return eval_affine_map(self, dims)

    def is_identity(self) -> bool:
        """Return True if this map is the identity: output == input for all inputs.

        Used at parse time to detect trivial coordinate-order maps.  When
        ``access_tile_order`` is an identity map it has no effect on which
        memory element lands at each output position, so we set
        ``coordinate_order`` to ``None``.  This allows load/store to skip
        the per-coord ``cso.eval()`` calls and, combined with a full
        ``coordinate_set``, enables the contiguous fast path entirely.
        """
        # Quick structural check: same number of inputs and outputs, and each
        # output expression is just the corresponding input dimension variable.
        if len(self.exprs) != self.n_dims:
            return False
        # Verify by evaluation: identity map satisfies eval(dims) == dims for
        # a non-trivial probe vector.  Using [1, 2, ..., n_dims] avoids false
        # positives from constant-zero expressions.
        probe = list(range(1, self.n_dims + 1))
        return list(self.eval(probe)) == probe


@dataclass(frozen=True)
class AffineSet:
    """Parsed affine_set<(d0,...) : (c0 >= 0, ...)>.

    Attributes:
        n_dims:       number of dimension variables
        constraints:  tuple of AST nodes; each node is the LHS of ``expr >= 0``
        source:       original verbatim string (for debugging / round-trip)
    """
    n_dims: int
    constraints: Tuple["_Node", ...]
    source: str

    def contains(self, point: Sequence[int]) -> bool:
        """Return True if *point* satisfies all constraints.

        Delegates to ``parser_ast.affine_set_contains``.
        """
        from .parser_ast import affine_set_contains
        return affine_set_contains(self, point)

    def enumerate(self, shape: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Return all integer points in ``[0, shape)`` satisfying all constraints.

        Delegates to ``parser_ast.enumerate_affine_set``.
        """
        from .parser_ast import enumerate_affine_set
        return enumerate_affine_set(self, shape)

    def is_full(self, shape: Tuple[int, ...]) -> bool:
        """Return True if this set covers every coordinate in *shape*.

        Called once at parse time to detect trivial coordinate sets — i.e.
        those that enumerate the full rectangular tile in row-major order.
        When a set is full, ``coordinate_set`` is set to ``None`` so
        that load/store can take the contiguous fast path instead of building
        and iterating a coordinate list on every execution.  Without this,
        even plain rectangular tiles pay the cost of enumerating all coords
        on every load/store (e.g. 46k times for a 32-core layernorm).

        Uses a vertex check: an affine set is convex, so it covers [0, shape)
        iff it contains all 2^n_dims corners of that box.  This is O(2^n_dims)
        constraint evaluations instead of O(∏ shape).
        """
        if len(shape) != self.n_dims:
            return False

        import itertools as _it
        corners = _it.product(*((0, n - 1) for n in shape))
        return all(self.contains(pt) for pt in corners)


@dataclass(frozen=True)
class BoxSet:
    """Axis-aligned integer hyperrectangle: ``{p : lo[d] <= p[d] < hi[d]}``.

    The axis-aligned specialisation of :class:`AffineSet`: every ``BoxSet``
    could equivalently be written as an ``AffineSet`` with per-axis
    inequalities, but carrying the ``(lo, hi)`` structure explicitly makes
    every operation (``contains``, ``enumerate``, ``is_empty``, ``is_full``,
    ``lower_bounds``, ``translate``, ``intersect``) O(ndim) with no
    constraint-AST walk.  Used for partition extents (``B_i``), access tile
    sets (``A``), and their intersections (``C_i``) in
    ``distributed_tile_access``; the parser lowers axis-aligned affine sets
    to this form at parse time (see ``try_from_affine_set``).

    ``BoxSet`` and ``AffineSet`` are peer dataclasses under a ``Union``
    rather than parent/child classes — structural fast paths must be
    visible at each call site via ``isinstance`` dispatch, not hidden
    behind polymorphism.  Mixed-type operations — e.g.
    ``BoxSet.intersect(aset: AffineSet)`` — raise ``TypeError``.  See the
    module docstring for the full relationship and operation matrix.
    """
    lo: Tuple[int, ...]   # inclusive
    hi: Tuple[int, ...]   # exclusive

    def __post_init__(self) -> None:
        if len(self.lo) != len(self.hi):
            raise ValueError(
                f"BoxSet: lo/hi length mismatch: lo={self.lo} hi={self.hi}"
            )

    @property
    def n_dims(self) -> int:
        return len(self.lo)

    def contains(self, point: Sequence[int]) -> bool:
        """True iff ``lo[d] <= point[d] < hi[d]`` for every dim."""
        if len(point) != self.n_dims:
            return False
        return all(self.lo[d] <= point[d] < self.hi[d] for d in range(self.n_dims))

    def enumerate(self, shape: Optional[Tuple[int, ...]] = None) -> List[Tuple[int, ...]]:
        """Return all integer points in the box in row-major order.

        ``shape`` is accepted for signature parity with
        :meth:`AffineSet.enumerate` (which needs an external bounding box
        for its brute-force iteration).  A ``BoxSet`` is self-bounded,
        so ``shape`` only serves as a sanity check: passed values must
        upper-bound ``hi`` componentwise, otherwise the box wouldn't fit
        inside the nominal bounding box and the call site is confused
        about its own invariants.
        """
        if shape is not None:
            if len(shape) != self.n_dims:
                raise ValueError(
                    f"BoxSet.enumerate: shape ndim {len(shape)} does not "
                    f"match box ndim {self.n_dims}"
                )
            for d in range(self.n_dims):
                if self.hi[d] > shape[d]:
                    raise ValueError(
                        f"BoxSet.enumerate: hi[{d}]={self.hi[d]} exceeds "
                        f"shape[{d}]={shape[d]} — box is not contained in "
                        f"the nominal bounding box."
                    )
        return list(itertools.product(*(range(self.lo[d], self.hi[d]) for d in range(self.n_dims))))

    def is_empty(self) -> bool:
        """True iff any axis has ``hi[d] <= lo[d]`` (i.e. empty extent)."""
        return any(self.hi[d] <= self.lo[d] for d in range(self.n_dims))

    def is_full(self, shape: Tuple[int, ...]) -> bool:
        """True iff this box equals ``[0, shape)``."""
        if len(shape) != self.n_dims:
            return False
        return self.lo == (0,) * self.n_dims and tuple(self.hi) == tuple(shape)

    def lower_bounds(self) -> Tuple[int, ...]:
        """Return ``lo`` — the per-axis minimum coordinate, O(1)."""
        return self.lo

    def translate(self, offset: Sequence[int]) -> "BoxSet":
        """Return a new box shifted by *offset* along each axis."""
        if len(offset) != self.n_dims:
            raise ValueError(
                f"BoxSet.translate: offset dim mismatch: "
                f"offset={tuple(offset)} n_dims={self.n_dims}"
            )
        return BoxSet(
            lo=tuple(self.lo[d] + offset[d] for d in range(self.n_dims)),
            hi=tuple(self.hi[d] + offset[d] for d in range(self.n_dims)),
        )

    def intersect(self, other: "BoxSet") -> "BoxSet":
        """Axis-wise intersection; result may be empty (``is_empty()``)."""
        if not isinstance(other, BoxSet):
            raise TypeError(
                f"BoxSet.intersect: mixed-type intersection not supported "
                f"(other is {type(other).__name__}).  Box and AffineSet are "
                f"structural peers, not interchangeable — promote or add a "
                f"dedicated op at the call site instead."
            )
        if other.n_dims != self.n_dims:
            raise ValueError(
                f"BoxSet.intersect: n_dims mismatch {self.n_dims} vs {other.n_dims}"
            )
        return BoxSet(
            lo=tuple(max(self.lo[d], other.lo[d]) for d in range(self.n_dims)),
            hi=tuple(min(self.hi[d], other.hi[d]) for d in range(self.n_dims)),
        )

    @classmethod
    def try_from_affine_set(cls, aset: "AffineSet") -> Optional["BoxSet"]:
        """Lower an axis-aligned :class:`AffineSet` to a ``BoxSet``.

        Returns ``None`` when the set is not representable as an integer
        box.  Lowering succeeds iff every constraint has the form
        ``c * d_i + k >= 0`` with ``c ∈ {+1, -1}`` (single dim, unit coeff)
        and every axis is pinned on **both** sides (at least one ``+d_i``
        and one ``-d_i`` constraint).

        Symbolic sets (``aset.n_syms > 0``, added by PR #42) are rejected —
        symbolic bounds need a symbolic box representation, out of scope
        here.  ``getattr`` keeps this correct before and after the PR #42
        rebase.
        """
        if getattr(aset, "n_syms", 0) != 0:
            return None
        n = aset.n_dims
        # Per axis: highest lo implied by any +d_i constraint, lowest hi
        # implied by any -d_i constraint.  None = not yet pinned.
        los: List[Optional[int]] = [None] * n
        his: List[Optional[int]] = [None] * n
        for c in aset.constraints:
            lin = _constraint_to_linear(c, n)
            if lin is None:
                return None
            coeffs, const = lin
            # Must touch exactly one dim with ±1 coefficient.
            nz = [i for i, k in enumerate(coeffs) if k != 0]
            if len(nz) != 1:
                return None
            i = nz[0]
            k = coeffs[i]
            if k == 1:
                # d_i + const >= 0  →  d_i >= -const
                candidate = -const
                los[i] = candidate if los[i] is None else max(los[i], candidate)
            elif k == -1:
                # -d_i + const >= 0  →  d_i <= const  →  hi = const + 1
                candidate = const + 1
                his[i] = candidate if his[i] is None else min(his[i], candidate)
            else:
                return None
        if any(v is None for v in los) or any(v is None for v in his):
            return None
        return cls(lo=tuple(los), hi=tuple(his))  # type: ignore[arg-type]


def _constraint_to_linear(node: "_Node", n_dims: int) -> Optional[Tuple[List[int], int]]:
    """Flatten a parsed constraint AST into ``(coeffs, const)``.

    Returns ``None`` if the expression isn't a pure linear combination of
    dimension variables and integer constants (e.g. contains ``sym`` or
    ``ref`` atoms, or non-unit multiplications on dim terms).

    The constraint represents ``sum(coeffs[i] * d_i) + const >= 0``.
    """
    coeffs = [0] * n_dims
    const_box = [0]

    def walk(n: "_Node", sign: int) -> bool:
        tag = n[0]
        if tag == "const":
            const_box[0] += sign * n[1]
            return True
        if tag == "dim":
            coeffs[n[1]] += sign
            return True
        if tag == "add":
            return walk(n[1], sign) and walk(n[2], sign)
        if tag == "sub":
            return walk(n[1], sign) and walk(n[2], -sign)
        if tag == "neg":
            return walk(n[1], -sign)
        if tag == "mul":
            # ("mul", N, operand) — N is a Python int, operand is a _Node.
            coef = n[1]
            inner = n[2]
            # Only dim variables can be scaled; scaling a const collapses
            # to a const, scaling an add/sub would require recursion with a
            # scaled sign.  Handle dim directly; reject compound operands.
            if inner[0] == "dim":
                coeffs[inner[1]] += sign * coef
                return True
            if inner[0] == "const":
                const_box[0] += sign * coef * inner[1]
                return True
            return False
        # 'sym', 'ref', or anything else — not a linear combination of dims.
        return False

    if not walk(node, 1):
        return None
    return coeffs, const_box[0]
