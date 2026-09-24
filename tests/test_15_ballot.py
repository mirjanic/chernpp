"""
Tier 15: the ballot conjecture C(M) >= b(M) at small d, proved in report/ballot.tex.

The proofs are by hand; these tests pin every identity and inequality they use,
exactly, on large boxes, so that a change to the artifacts cannot silently
invalidate them.
"""

import unittest

import numpy as np

from chernpp import boxes, sectors
from chernpp.artifacts import load_algebra


def z3(K):
    """Z_3 = 1 + x/((1-x)(1-y)) + xy/(1-xy) on the square box of side K."""
    Z = np.zeros((K + 1, K + 1), dtype=np.int64)
    Z[0, 0] = 1
    Z[1:, :] += 1
    for k in range(1, K + 1):
        Z[k, k] += 1
    return Z


class TestDTwo(unittest.TestCase):
    def test_f2_is_one_plus_x_over_one_minus_2x(self):
        grid = boxes.exact_box(boxes.Box(2, (20,), "line"))
        self.assertEqual([int(v) for v in grid], [1] + [2 ** (k - 1) for k in range(1, 21)])


class TestDThree(unittest.TestCase):
    def test_numerator_factorises(self):
        # N_3 = (1 - x)(1 - y)(1 - xy)
        expected = {(0, 0): 1, (1, 0): -1, (0, 1): -1, (1, 2): 1, (2, 1): 1, (2, 2): -1}
        self.assertEqual(dict(load_algebra(3).numerator), expected)

    def test_f3_dominates_z3_with_excess_off_the_first_two_rows(self):
        K = 30
        diff = boxes.exact_box(boxes.Box(3, (K, K), "sq")) - z3(K)
        self.assertTrue((diff >= 0).all())
        self.assertTrue((diff[2:, :] >= 1).all())

    def test_z3_packet_sums_are_ballot_counts(self):
        K = 30
        table = boxes.chern_table(z3(K), boxes.Box(3, (K, K), "sq"))
        for m, v in table.items():
            self.assertEqual(v, sectors.ballot_count(m), m)

    def test_plane_sector_equality_at_d3(self):
        table = boxes.chern_table_exact(boxes.level_box(3, 3))
        for m in ((0, 0, 0), (1, 0, -1), (1, 1, -2)):
            self.assertEqual(table[m], sectors.ballot_count(m))


class TestDFour(unittest.TestCase):
    """The explicit positive product formula for F_4 (report/ballot.tex, Lemma 4)."""

    def test_numerator_factorises(self):
        from chernpp.polynomial import poly_mul

        def one_minus(p):
            out = {(0, 0, 0): 1}
            for k, v in p.items():
                out[k] = out.get(k, 0) - v
            return out

        factors = [
            {(1, 0, 0): 1},
            {(0, 1, 0): 1},
            {(1, 1, 0): 1},
            {(0, 0, 1): 1},
            {(0, 1, 1): 1},
            {(1, 1, 1): 1},
            {(0, 1, 1): 1, (1, 1, 1): 2},
        ]
        prod = {(0, 0, 0): 1}
        for f in factors:
            prod = poly_mul(prod, one_minus(f))
        self.assertEqual({k: v for k, v in prod.items() if v}, dict(load_algebra(4).numerator))

    def test_block_decomposition_is_an_identity(self):
        # (1-a)(1-s)(1-s-2as) = a^2 s (1-2s) + (1-2a)(1-s-as)/2 + (1-s-as)(1-2s)/2
        from fractions import Fraction

        from chernpp.polynomial import poly_add, poly_mul

        def lin(*terms):
            return {k: Fraction(v) for k, v in terms}

        lhs = poly_mul(
            poly_mul(lin(((0, 0), 1), ((1, 0), -1)), lin(((0, 0), 1), ((0, 1), -1))),
            lin(((0, 0), 1), ((0, 1), -1), ((1, 1), -2)),
        )
        u = lin(((0, 0), 1), ((0, 1), -1), ((1, 1), -1))
        rhs = poly_add(
            poly_add(
                poly_mul(lin(((2, 1), 1)), lin(((0, 0), 1), ((0, 1), -2))),
                poly_mul(lin(((0, 0), Fraction(1, 2))), poly_mul(lin(((0, 0), 1), ((1, 0), -2)), u)),
            ),
            poly_mul(lin(((0, 0), Fraction(1, 2))), poly_mul(u, lin(((0, 0), 1), ((0, 1), -2)))),
        )
        clean = lambda p: {k: v for k, v in p.items() if v}
        self.assertEqual(clean(lhs), clean(rhs))


if __name__ == "__main__":
    unittest.main()
