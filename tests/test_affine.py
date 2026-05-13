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

"""Tests for affine.py — AffineMap and AffineSet value objects.

These tests verify that the convenience methods (eval, contains, enumerate)
on AffineMap and AffineSet correctly delegate to parser_ast.py.  They are
intentionally thin — the heavy evaluation logic is tested in test_ast.py.
"""

import pytest

from ktir_cpu.parser_ast import parse_affine_map, parse_affine_set


class TestAffineMapObject:

    def test_eval_delegates(self):
        m = parse_affine_map("affine_map<(d0) -> (d0)>")
        assert m.eval([7]) == (7,)

    def test_eval_non_identity(self):
        m = parse_affine_map("affine_map<(i) -> (i, 0)>")
        assert m.eval([3]) == (3, 0)

    def test_eval_wrong_dims_raises(self):
        m = parse_affine_map("affine_map<(d0, d1) -> (d0, d1)>")
        with pytest.raises(ValueError):
            m.eval([1])

    def test_source_field(self):
        s = "affine_map<(d0) -> (d0)>"
        m = parse_affine_map(s)
        assert m.source == s

    def test_frozen(self):
        m = parse_affine_map("affine_map<(d0) -> (d0)>")
        with pytest.raises((AttributeError, TypeError)):
            m.n_dims = 99  # type: ignore[misc]


class TestAffineSetObject:

    def test_contains_delegates(self):
        s = parse_affine_set("affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>")
        assert s.contains([2])
        assert not s.contains([5])

    def test_enumerate_delegates(self):
        s = parse_affine_set("affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>")
        assert s.enumerate((4,)) == [(0,), (1,), (2,), (3,)]

    def test_enumerate_wrong_shape_raises(self):
        s = parse_affine_set("affine_set<(d0, d1) : (d0 >= 0, d1 >= 0)>")
        with pytest.raises(ValueError):
            s.enumerate((4,))

    def test_source_field(self):
        src = "affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>"
        s = parse_affine_set(src)
        assert s.source == src

    def test_frozen(self):
        s = parse_affine_set("affine_set<(d0) : (d0 >= 0)>")
        with pytest.raises((AttributeError, TypeError)):
            s.n_dims = 99  # type: ignore[misc]

    def test_is_full_true(self):
        """Set covering exactly the full 2×4 box is recognised as full."""
        s = parse_affine_set("affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 3 >= 0)>")
        assert s.is_full((2, 4))

    def test_is_full_false(self):
        """Upper-triangular set (d1 >= d0) is not full — corner (3,0) is excluded."""
        s = parse_affine_set("affine_set<(d0, d1) : (d1 - d0 >= 0)>")
        assert not s.is_full((4, 4))

    def test_is_full_wrong_ndim(self):
        """Shape ndim != set n_dims always returns False."""
        s = parse_affine_set("affine_set<(d0) : (d0 >= 0)>")
        assert not s.is_full((2, 2))


class TestBoxSet:
    """box_set(shape) produces an AffineSet covering [0, shape) without parsing."""

    def test_contains(self):
        """Boundary points are included; points at or beyond shape are excluded."""
        from ktir_cpu.affine import box_set
        b = box_set((3, 4))
        assert b.contains((0, 0))      # top-left corner
        assert b.contains((2, 3))      # bottom-right corner (inclusive)
        assert not b.contains((3, 0))  # one past last row
        assert not b.contains((0, 4))  # one past last col

    def test_enumerate(self):
        """enumerate returns all integer points in [0, shape) in any order."""
        from ktir_cpu.affine import box_set
        pts = box_set((2, 3)).enumerate((2, 3))
        assert set(pts) == {(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)}

    def test_is_full(self):
        """box_set(shape).is_full(shape) is always True; mismatched shape is False."""
        from ktir_cpu.affine import box_set
        assert box_set((2, 3)).is_full((2, 3))
        assert not box_set((2, 3)).is_full((3, 3))


class TestShiftedAffineSet:
    """AffineSet.shift(offset) — p in shifted iff p - offset in base.

    Used in distributed_tile_access to map the local access tile A into global
    coords: x + A = box_set(access_shape).shift(x).
    """

    def test_shift_contains(self):
        """box [0,2)×[0,2) shifted by (1,1) covers global [1,3)×[1,3)."""
        from ktir_cpu.affine import box_set
        s = box_set((2, 2)).shift((1, 1))
        assert s.contains((1, 1))      # new top-left
        assert s.contains((2, 2))      # new bottom-right
        assert not s.contains((0, 0))  # original origin, now outside
        assert not s.contains((3, 3))  # one past new bottom-right

    def test_shift_enumerate(self):
        """enumerate returns all global coords inside the shifted box."""
        from ktir_cpu.affine import box_set
        pts = box_set((2, 2)).shift((1, 1)).enumerate((4, 4))
        assert set(pts) == {(1,1),(1,2),(2,1),(2,2)}

    def test_shift_parsed_set(self):
        """Shifting a parsed row-band set moves it to a new row range.

        Partition B0 covers rows 0..1; shifted by (2,0) it covers rows 2..3.
        This models distributing the same partition layout to a different
        row block of a larger tensor.
        """
        s = parse_affine_set("affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 3 >= 0)>")
        shifted = s.shift((2, 0))
        assert shifted.contains((2, 0))  # first row of shifted band
        assert shifted.contains((3, 3))  # last element of shifted band
        assert not shifted.contains((0, 0))  # original band, now excluded
        assert not shifted.contains((4, 0))  # one past shifted band


class TestIntersectedAffineSet:
    """AffineSet.intersect(other) — p in result iff p in both sets.

    Used in distributed_tile_access to compute C_i = (x + A) ∩ B_i:
    the global coordinates covered by both the access tile and partition i.
    """

    def test_intersect_contains(self):
        """Full 4×4 box ∩ rows-0..1 band = rows 0..1, all cols."""
        from ktir_cpu.affine import box_set
        B = parse_affine_set("affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 3 >= 0)>")
        C = box_set((4, 4)).intersect(B)
        assert C.contains((0, 0))
        assert C.contains((1, 3))
        assert not C.contains((2, 0))  # row 2 is in box but not in B

    def test_intersect_enumerate(self):
        """2×4 box ∩ rows-0..1 band enumerates all 8 points in rows 0..1."""
        from ktir_cpu.affine import box_set
        B = parse_affine_set("affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 3 >= 0)>")
        C = box_set((2, 4)).intersect(B)
        pts = C.enumerate((4, 4))
        assert set(pts) == {(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3)}

    def test_intersect_ndim_mismatch_raises(self):
        """Intersecting sets with different n_dims raises AssertionError."""
        s1 = parse_affine_set("affine_set<(d0) : (d0 >= 0)>")
        s2 = parse_affine_set("affine_set<(d0, d1) : (d0 >= 0, d1 >= 0)>")
        with pytest.raises(AssertionError):
            s1.intersect(s2)

    def test_intersect_partial_overlap(self):
        """Access tile [1,3)×[1,3) ∩ partition rows 0..1 = row 1, cols 1..2.

        Models Case 3 of distributed_tile_access: global_base=(1,1),
        access_shape=(2,2), B0 covers rows 0..1. Only row 1, cols 1..2
        are in both the access tile and partition 0.
        """
        from ktir_cpu.affine import box_set
        B0 = parse_affine_set("affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 3 >= 0)>")
        xA = box_set((2, 2)).shift((1, 1))  # global footprint [1,3)×[1,3)
        C = xA.intersect(B0)
        pts = C.enumerate((4, 4))
        assert set(pts) == {(1, 1), (1, 2)}
