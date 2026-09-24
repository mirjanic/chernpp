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


try:
    import islpy  # noqa: F401

    HAVE_ISL = True
except ImportError:  # pragma: no cover
    HAVE_ISL = False


@unittest.skipUnless(HAVE_ISL, "islpy is needed for the exact set inclusions")
class TestDFourBallot(unittest.TestCase):
    """The d = 4 ballot theorem: exact inclusions plus a numerical cross-check of every support."""

    def test_exact_inclusions(self):
        from chernpp.ballot import verify_d4

        for name, ok in verify_d4().items():
            with self.subTest(check=name):
                self.assertTrue(ok)

    def test_claimed_supports_lie_in_the_true_supports(self):
        import islpy as isl

        from chernpp.ballot import FACTOR_SUPPORTS
        from chernpp.polynomial import expand_rational

        K = 10
        x1, x2, x3, s, t = (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)
        series = {
            "X": expand_rational({x1: 1}, [{x1: 2}], 3 * K),
            "S": expand_rational({s: 1}, [{s: 2}], 3 * K),
            "Y": expand_rational({(2, 1, 1): 1}, [{x1: 2}, {s: 1, t: 1}], 3 * K),
            "V1": expand_rational({(1, 1, 0): 1}, [{(1, 1, 0): 2}], 3 * K),
            "V2": expand_rational({(1, 1, 0): 1}, [{x2: 1, (1, 1, 0): 1}], 3 * K),
            "V3": expand_rational({t: 1}, [{t: 2}], 3 * K),
            "V4": expand_rational({t: 1}, [{x3: 1, t: 1}], 3 * K),
        }
        for name, ser in series.items():
            claimed = isl.Set(FACTOR_SUPPORTS[name])
            for a in range(K + 1):
                for b in range(K + 1):
                    for c in range(K + 1):
                        inside = claimed.intersect(isl.Set(f"{{ [{a},{b},{c}] }}")).is_empty() is False
                        with self.subTest(factor=name, cell=(a, b, c)):
                            self.assertEqual(inside, ser.get((a, b, c), 0) >= 1)

    def test_plane_sector_equality_at_d4(self):
        table = boxes.chern_table_exact(boxes.level_box(4, 4))
        plane = [m for m in table if max(m) <= 1]
        self.assertEqual(len(plane), 5)
        for m in plane:
            self.assertEqual(table[m], sectors.ballot_count(m))


if __name__ == "__main__":
    unittest.main()
