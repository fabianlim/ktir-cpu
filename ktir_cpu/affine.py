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

These are plain data containers.  All parsing and heavy-lifting evaluation
logic lives in ``parser_ast.py``; the convenience methods below simply
delegate there.

Types
-----
AffineMap   — represents affine_map<(d0,...) -> (e0,...)>
AffineSet   — represents affine_set<(d0,...) : (c0 >= 0, ...)>
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Sequence, Tuple

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

    def shift(self, offset: Sequence[int]) -> "AffineSet":
        """Return a new AffineSet whose domain is shifted by *offset*.

        A point ``p`` is in ``self.shift(x)`` iff ``p - x`` is in ``self``.
        Implemented by wrapping contains/enumerate — no AST modification.
        """
        return _ShiftedAffineSet(self, tuple(offset))

    def intersect(self, other: "AffineSet") -> "AffineSet":
        """Return the intersection of this set with *other*.

        A point is in the intersection iff it satisfies both sets.
        Both sets must have the same n_dims.
        """
        assert self.n_dims == other.n_dims, (
            f"intersect: n_dims mismatch {self.n_dims} vs {other.n_dims}"
        )
        return _IntersectedAffineSet(self, other)

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


def _iter_box(shape: Tuple[int, ...]):
    """Yield all integer coordinate tuples in [0, shape)."""
    import itertools
    yield from itertools.product(*[range(s) for s in shape])


def box_set(shape: Tuple[int, ...]) -> "AffineSet":
    """Return an AffineSet covering the full box [0, shape) without parsing.

    A point p is in the box iff 0 <= p[d] < shape[d] for all d.
    Use .shift(origin) to get a box anchored at a global origin.
    """
    return _BoxAffineSet(shape)


class _BoxAffineSet(AffineSet):
    """AffineSet covering the full box [0, shape) — no parsing needed."""
    def __new__(cls, shape):
        return object.__new__(cls)

    def __init__(self, shape: Tuple[int, ...]):
        object.__setattr__(self, 'n_dims', len(shape))
        object.__setattr__(self, 'constraints', ())
        object.__setattr__(self, 'source', f"box{shape}")
        object.__setattr__(self, '_shape', shape)

    def contains(self, point: Sequence[int]) -> bool:
        return all(0 <= point[d] < self._shape[d] for d in range(self.n_dims))

    def enumerate(self, shape: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        return list(_iter_box(shape))


class _ShiftedAffineSet(AffineSet):
    """AffineSet shifted by a constant offset.

    ``p in shifted`` iff ``p - offset in base``.
    """

    def __new__(cls, base: AffineSet, offset: Tuple[int, ...]):
        obj = object.__new__(cls)
        return obj

    def __init__(self, base: AffineSet, offset: Tuple[int, ...]):
        object.__setattr__(self, 'n_dims', base.n_dims)
        object.__setattr__(self, 'constraints', base.constraints)
        object.__setattr__(self, 'source', f"shift({base.source}, {offset})")
        object.__setattr__(self, '_base', base)
        object.__setattr__(self, '_offset', offset)

    def contains(self, point: Sequence[int]) -> bool:
        shifted = tuple(point[d] - self._offset[d] for d in range(self.n_dims))
        return self._base.contains(shifted)

    def enumerate(self, shape: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        # A point p is in shift(base, x) iff p-x is in base.
        # Enumerate all p in [0,shape) that satisfy contains(p).
        return [p for p in _iter_box(shape) if self.contains(p)]


class _IntersectedAffineSet(AffineSet):
    """Intersection of two AffineSets with the same n_dims.

    ``p in intersection`` iff ``p in left AND p in right``.

    enumerate() optimisation: when ``left`` is a shifted box (the common case
    from distributed_tile_access where left = x+A), we know its tight bounding
    box and iterate only that sub-region rather than all of ``shape``.
    """
    def __new__(cls, left: AffineSet, right: AffineSet):
        obj = object.__new__(cls)
        return obj

    def __init__(self, left: AffineSet, right: AffineSet):
        object.__setattr__(self, 'n_dims', left.n_dims)
        object.__setattr__(self, 'constraints', left.constraints + right.constraints)
        object.__setattr__(self, 'source', f"intersect({left.source}, {right.source})")
        object.__setattr__(self, '_left', left)
        object.__setattr__(self, '_right', right)

    def contains(self, point: Sequence[int]) -> bool:
        return self._left.contains(point) and self._right.contains(point)

    def enumerate(self, shape: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        # Fast path: if left is a shifted box we know its tight bounding region
        # [offset, offset+box_shape), so enumerate only that sub-region and
        # filter by right.  This avoids iterating all of `shape` (e.g. 192×64)
        # when the access tile is small (e.g. 4×4).
        left = self._left
        if isinstance(left, _ShiftedAffineSet) and isinstance(left._base, _BoxAffineSet):
            import itertools
            offset = left._offset
            box_shape = left._base._shape
            candidates = itertools.product(
                *(range(offset[d], offset[d] + box_shape[d]) for d in range(self.n_dims))
            )
            return [p for p in candidates if self._right.contains(p)]
        return [p for p in _iter_box(shape) if self.contains(p)]
