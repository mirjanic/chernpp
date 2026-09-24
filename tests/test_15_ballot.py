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


class TestDFiveGauged(unittest.TestCase):
    """The GBC-gauged d = 5 chamber series: complete factorisation, and the second block's certificate."""

    def test_gauged_numerator_factors_completely(self):
        from chernpp.optimisation import gauge
        from chernpp.polynomial import poly_mul

        g = gauge.setup(5, 4)
        alg = g.algebra

        def lin(**kw):
            out = {}
            for name, c in kw.items():
                e = [0] * 5
                e[int(name[1:]) - 1] = 1
                out[tuple(e)] = out.get(tuple(e), 0) + c
            return out

        gbc = poly_mul(lin(z1=2, z2=1, z5=-1), poly_mul(lin(z1=2, z2=-1), lin(z1=1, z4=1, z5=-1)))
        Q = gauge.to_z(alg.multidegree, 5, g.degree)
        for k, v in gbc.items():
            Q[k] = Q.get(k, 0) - v
        chamber = gauge.to_chamber({k: v for k, v in Q.items() if v}, 5, g.degree)
        shifted = {tuple(e[i] - g.correction[i] for i in range(4)): g.sign * c for e, c in chamber.items()}
        numerator = {k: v for k, v in poly_mul(shifted, dict(alg.vandermonde)).items() if v}

        def one_minus(terms):
            out = {(0, 0, 0, 0): 1}
            for k, v in terms:
                out[k] = out.get(k, 0) - v
            return out

        x = lambda *idx: tuple(1 if i in idx else 0 for i in range(4))
        factors = [
            [(x(0), 1)],
            [(x(1), 1)],
            [(x(0, 1), 1)],
            [(x(2), 1)],
            [(x(1, 2), 1)],
            [(x(1, 2), 2)],
            [(x(0, 1, 2), 1)],
            [(x(3), 1)],
            [(x(2, 3), 1)],
            [(x(1, 2, 3), 1)],
            [(x(0, 1, 2, 3), 1)],
            [(x(2, 3), 1), (x(0, 1, 2, 3), 2)],
            [(x(1, 2, 3), 1), (x(0, 1, 2, 3), 2)],
        ]
        product = {(0, 0, 0, 0): 1}
        for f in factors:
            product = poly_mul(product, one_minus(f))
        self.assertEqual({k: v for k, v in product.items() if v}, numerator)

    def test_block_certificate_verifies_exactly(self):
        from chernpp.ballot import verify_d5_block

        self.assertTrue(verify_d5_block())


@unittest.skipUnless(HAVE_ISL, "islpy is needed for the exact set inclusions")
class TestDFourBijection(unittest.TestCase):
    """Step 2 at d = 4: the explicit word-preserving bijection from the zero set onto T."""

    def test_a_broken_piece_is_detected(self):
        # Negative control: the checks must be able to fail.
        from chernpp import ballot

        good = ballot.BIJECTION_D4
        try:
            broken = list(good)
            broken[0] = (broken[0][0], (2, 0, 1, 3))
            ballot.BIJECTION_D4 = tuple(broken)
            self.assertFalse(all(ballot.verify_d4_bijection().values()))
        finally:
            ballot.BIJECTION_D4 = good

    def test_z4_packet_sums_are_ballot_counts(self):
        # Z_4 = P_4 - 1_Z + 1_T on a box, grouped into complete packets.
        import islpy as isl

        from chernpp.ballot import TARGETS, ZERO_SET

        K = 18
        box = boxes.Box(4, (K, K, K), "cube")
        Z4 = np.ones(box.shape, dtype=np.int64)
        for text, delta in ((ZERO_SET, -1), (TARGETS, 1)):
            s = isl.Set(text).intersect(isl.Set(f"{{ [a,b,c] : 0 <= a,b,c <= {K} }}"))
            pts = []
            s.foreach_point(
                lambda p: pts.append(
                    tuple(p.get_coordinate_val(isl.dim_type.set, k).to_python() for k in range(3))
                )
            )
            for c in pts:
                Z4[c] += delta
        for m, v in boxes.chern_table(Z4, box).items():
            self.assertEqual(v, sectors.ballot_count(m), m)


@unittest.skipUnless(HAVE_ISL, "islpy is needed for the exact set inclusions")
class TestDominantCell(unittest.TestCase):
    """d = 4 again, and d = 5: C(M) >= F(beta_max(M)) >= 2^E > d! >= b(M) for large charge."""

    def test_d4_by_the_dominant_cell(self):
        from chernpp.ballot import verify_d4_dominant

        for name, ok in verify_d4_dominant().items():
            with self.subTest(check=name):
                self.assertTrue(ok)

    def test_d5(self):
        from chernpp.ballot import verify_d5

        for name, ok in verify_d5().items():
            with self.subTest(check=name):
                self.assertTrue(ok)

    def test_d5_alternative_block_certificate(self):
        from chernpp.ballot import d5_alternative_block

        for name, ok in d5_alternative_block().items():
            with self.subTest(check=name):
                self.assertTrue(ok)

    def test_the_growth_bound_needs_its_threshold(self):
        # Negative control: below the threshold the isl cover leaves cells over.
        from chernpp import ballot

        cells = ("a", "b", "c", "e")
        region = ballot.decreasing_cone(5, cells, 6)
        rest, _ = ballot.growth_cover(ballot.D5_BLOCK_TERMS, ballot.D5_FACTORS, region, 7, cells)
        self.assertFalse(rest.is_empty())


class TestAtoms(unittest.TestCase):
    """Every atom's claimed support and 2^E bound, against the true series on a box."""

    @staticmethod
    def truth():
        from chernpp.polynomial import expand_rational

        def m(*es):
            return tuple(sum(e[i] for e in es) for i in range(len(es[0])))

        a1, a2, a3 = (1, 0, 0), (0, 1, 0), (0, 0, 1)
        s4, t4 = m(a2, a3), m(a1, a2, a3)
        x1, x2, x3, x4 = (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)
        b, r = x2, m(x3, x4)
        p, q = m(x1, b, r), m(b, r)
        A, S, P, Q = {x1: 2}, {m(x1, b): 2}, {p: 2}, {q: 2}
        U, V = {r: 1, p: 1}, {r: 1, q: 1}
        one4, one5 = (0, 0, 0), (0, 0, 0, 0)
        d4 = {
            "1": ({one4: 1}, []),
            "X": ({a1: 1}, [{a1: 2}]),
            "S": ({s4: 1}, [{s4: 2}]),
            "Y": ({m(a1, a1, s4): 1}, [{a1: 2}, {s4: 1, t4: 1}]),
            "V1": ({m(a1, a2): 1}, [{m(a1, a2): 2}]),
            "V2": ({m(a1, a2): 1}, [{a2: 1, m(a1, a2): 1}]),
            "V3": ({t4: 1}, [{t4: 2}]),
            "V4": ({t4: 1}, [{a3: 1, t4: 1}]),
        }
        d5 = {
            "1": ({one5: 1}, []),
            "x/A": ({x1: 1}, [A]),
            "xb/S": ({m(x1, b): 1}, [S]),
            "x2b/AS": ({m(x1, x1, b): 1}, [A, S]),
            "br/Q": ({q: 1}, [Q]),
            "br2/Q": ({m(q, r): 1}, [Q]),
            "xb2r/SQ": ({m(x1, b, b, r): 1}, [S, Q]),
            "x2br/AU": ({m(x1, x1, b, r): 1}, [A, U]),
            "x3b2r/ASU": ({m(x1, x1, x1, b, b, r): 1}, [A, S, U]),
            "x2b3r2/SQU": ({m(x1, x1, b, b, b, r, r): 1}, [S, Q, U]),
            "xb2r3/PV": ({m(x1, b, b, r, r, r): 1}, [P, V]),
            "br3/QV": ({m(b, r, r, r): 1}, [Q, V]),
            "V1": ({m(x1, x2): 1}, [{x2: 1, m(x1, x2): 1}]),
            "V2": ({m(x1, x2, x3): 1}, [{m(x2, x3): 1, m(x1, x2, x3): 1}]),
            "V3": ({m(x1, x2, x3): 1}, [{x3: 1, m(x1, x2, x3): 1}]),
            "V4": ({p: 1}, [{x4: 1, p: 1}]),
            "V5": ({q: 1}, [{p: 1, q: 1}]),
            "V6": ({m(x1, x2, x3): 1}, [{m(x1, x2, x3): 2}]),
        }
        return (
            (4, d4, {**ballot_mod().D4_BASE, **ballot_mod().D4_FACTORS}),
            (5, d5, {**ballot_mod().D5_BLOCK_TERMS, **ballot_mod().D5_FACTORS}),
        )

    def test_atoms_match_their_series(self):
        from chernpp.ballot import atom_series
        from chernpp.polynomial import expand_rational

        K = 9  # cells with every coordinate <= K
        for d, true, atoms in self.truth():
            for name, atom in atoms.items():
                num, dens = true[name]
                series = expand_rational(num, dens, (d - 1) * K)
                claimed = {c: v for c, v in atom_series(atom, d - 1, 2 * K).items() if max(c) <= K}
                support = {c: v for c, v in series.items() if max(c) <= K and v}
                with self.subTest(d=d, atom=name):
                    self.assertEqual(set(claimed), set(support))
                    self.assertTrue(all(support[c] >= claimed[c] for c in claimed))


def ballot_mod():
    from chernpp import ballot

    return ballot


if __name__ == "__main__":
    unittest.main()
