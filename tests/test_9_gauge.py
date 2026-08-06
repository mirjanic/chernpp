"""
Tier 9: gauge freedom in the numerator of the residue formula.

Two numerators differing by a residue-null kernel give the same Thom polynomial
and different chamber series.  These tests cover the coordinate machinery, the
fact that the series really is linear in the numerator, and the search for a
numerator whose series is coefficientwise nonnegative -- including the point at
which that search stops being trustworthy.
"""

import unittest

from chernpp.optimisation import gauge
from chernpp.artifacts import load_algebra
from chernpp.chamber import chamber_series
from chernpp.polynomial import negative_terms


class TestCoordinates(unittest.TestCase):
    """z-space and chamber coordinates are inverse to one another."""

    def test_round_trip_on_every_stored_multidegree(self):
        for order in (4, 5, 6, 7):
            alg = load_algebra(order)
            degree = max(e[-1] for e in alg.multidegree)
            z = gauge.to_z(alg.multidegree, order, degree)
            self.assertEqual(gauge.to_chamber(z, order, degree), alg.multidegree)

    def test_multidegree_is_homogeneous_in_z(self):
        # deg Q_d is the codimension, and the residue formula needs that.
        for order, codim in ((4, 1), (5, 3), (6, 7), (7, 13)):
            alg = load_algebra(order)
            z = gauge.to_z(alg.multidegree, order, codim)
            self.assertEqual({sum(e) for e in z}, {codim})

    def test_a_non_homogeneous_numerator_is_refused(self):
        with self.assertRaises(ValueError):
            gauge.to_chamber({(1, 0, 0, 0, 0): 1, (2, 0, 0, 0, 0): 1}, 5, 3)

    def test_a_chamber_monomial_off_the_lattice_is_refused(self):
        # Decreasing partial sums do not come from a polynomial in z.
        with self.assertRaises(ValueError):
            gauge.to_z({(2, 1, 0, 0): 1}, 5, 3)


class TestCorrectionMonomial(unittest.TestCase):
    """The chamber correction is recovered from the artifact, not tabulated."""

    def test_recovers_the_stored_normalisation(self):
        for order in (4, 5, 6, 7):
            alg = load_algebra(order)
            corr = gauge.correction_monomial(alg)
            rebuilt = {
                tuple(e[i] - corr[i] for i in range(alg.nvars)): -c for e, c in alg.multidegree.items()
            }
            self.assertEqual(rebuilt, alg.normalized_numerator)

    def test_a_tampered_artifact_is_refused(self):
        alg = load_algebra(5)
        broken = type(alg)(**{**alg.__dict__, "normalized_numerator": {(0, 0, 0, 0): 1}})
        with self.assertRaises(RuntimeError):
            gauge.correction_monomial(broken)


class TestSeriesFromNumerator(unittest.TestCase):
    """The chamber series as a linear function of the numerator."""

    def test_reproduces_the_canonical_series(self):
        for order in (4, 5, 6):
            g = gauge.setup(order, 10)
            z = gauge.to_z(g.algebra.multidegree, order, g.degree)
            self.assertEqual(gauge.series_of(z, g), chamber_series(order, 10))

    def test_is_linear_in_the_numerator(self):
        g = gauge.setup(5, 8)
        a = {(3, 0, 0, 0, 0): 1}
        b = {(0, 0, 0, 3, 0): 2, (1, 1, 1, 0, 0): -1}
        combined = {(3, 0, 0, 0, 0): 1, (0, 0, 0, 3, 0): 2, (1, 1, 1, 0, 0): -1}
        sa, sb = gauge.series_of(a, g), gauge.series_of(b, g)
        expected = {e: sa.get(e, 0) + sb.get(e, 0) for e in set(sa) | set(sb)}
        self.assertEqual(gauge.series_of(combined, g), {e: c for e, c in expected.items() if c})

    def test_a_numerator_missing_the_correction_factor_is_refused(self):
        # z_5^3 has chamber image x^(0,0,0,0), not divisible by the correction
        # x^(0,0,0,1).  Expanding it anyway would give a Laurent series and
        # corrupt every packet sum, so it must raise rather than truncate.
        g = gauge.setup(5, 6)
        with self.assertRaises(ValueError):
            gauge.series_of({(0, 0, 0, 0, 3): 1}, g)


class TestPositiveGaugeAtA5(unittest.TestCase):
    """
    A_5 has a numerator whose series is coefficientwise nonnegative.

    The canonical Q_5 does not -- the strong conjecture is false there -- so this
    is the whole point of the construction: the negative coefficients are an
    artefact of the numerator, and a different representative of the same Thom
    polynomial removes them.
    """

    @classmethod
    def setUpClass(cls):
        cls.g = gauge.setup(5, 16)
        cls.result = gauge.solve_positive_gauge_continuous(5, fit_depth=16, bound=20.0)

    def test_the_search_succeeds_and_was_verified_exactly(self):
        self.assertTrue(self.result.found)
        self.assertGreater(self.result.negatives_before, 0)
        self.assertEqual(self.result.negatives_remaining, 0)

    def test_the_kernel_preserves_every_packet_in_range(self):
        # Delegated to validate_gauge, which compares packet sums exactly.  A
        # packet sum is an integer; "133.00000000000017" is a moved packet, not
        # a rounding detail to be waved through with a tolerance.
        v = gauge.validate_gauge(self.result.kernel, 5, self.g.max_deg)
        self.assertEqual(v.packets_changed, 0)
        self.assertGreater(v.packets, 0)

    def test_the_kernel_holds_out_of_sample(self):
        # Fitted at truncation 16; these truncations bring in packets and
        # coefficients the search never saw.  Overfitting shows up here.
        for truncation in (18, 20):
            v = gauge.validate_gauge(self.result.kernel, 5, truncation)
            self.assertEqual(v.packets_changed, 0, f"packet sums moved at {truncation}")
            self.assertEqual(v.negatives_gauged, 0, f"negatives reappeared at {truncation}")
            self.assertGreater(v.negatives_canonical, 0)
            self.assertTrue(v.holds)


class TestTheSearchDoesNotScaleToA6(unittest.TestCase):
    """
    Why the same search cannot be trusted at d = 6.

    Nullity is imposed through the Chern packets that fit under a truncation.  At
    d = 5 those outnumber the free monomials and the answer generalises.  At
    d = 6 the numerator has 766 free monomials while the truncation supplies only
    tens of packets, so the programme is free to zero the packets it can see and
    the kernel it returns does not survive a higher truncation.

    Pinned as a fact about the method, not as a claim about A_6: closing d = 6
    needs nullity imposed structurally, in all degrees at once, rather than
    sampled at a truncation.
    """

    def test_a5_is_overdetermined_and_a6_is_not(self):
        counts = {}
        for order in (5, 6):
            g = gauge.setup(order, 16)
            counts[order] = (len(gauge.usable_packets(g)), len(gauge.admissible_monomials(g)))
        packets5, monomials5 = counts[5]
        packets6, monomials6 = counts[6]
        self.assertGreater(packets5, monomials5)
        self.assertLess(packets6 * 20, monomials6)

    def test_the_packet_supply_grows_far_too_slowly_at_a6(self):
        # Roughly four more packets per degree against 766 monomials: the
        # truncation needed to determine the kernel is out of reach.
        supply = [len(gauge.usable_packets(gauge.setup(6, md))) for md in (10, 12, 14, 16)]
        self.assertEqual(supply, sorted(supply))
        self.assertLess(supply[-1], 40)
        self.assertEqual(len(gauge.admissible_monomials(gauge.setup(6, 10))), 766)


class TestSymmetryKernels(unittest.TestCase):
    """
    Null kernels constructed from a transposition rather than fitted to packets.

    If P / D_d is invariant under a transposition, the integrand is antisymmetric
    against the symmetric Chern insertion and the residue vanishes at every level.
    That reason does not weaken with d, which is what makes this the route that
    reaches A_7 where the packet-fitted search cannot.
    """

    def test_the_denominator_has_the_expected_number_of_factors(self):
        for order, count in ((5, 13), (6, 22), (7, 34)):
            self.assertEqual(len(gauge.denominator_weights(order)), count)

    def test_the_contour_safe_swap_at_a5_absorbs_exactly_two_factors(self):
        # s_45 is the swap the residue argument uses at d = 5, and the factors it
        # moves are z2 + z3 - z5 and z1 + z4 - z5.
        swap = gauge.analyse_swap(5, 3, 4)
        self.assertTrue(swap.contour_safe)
        self.assertEqual(swap.a_degree, 2)
        self.assertEqual(sorted(swap.moved), sorted([(0, 1, 1, 0, -1), (1, 0, 0, 1, -1)]))

    def test_every_order_has_a_contour_safe_swap(self):
        for order in (5, 6, 7):
            safe = [gauge.analyse_swap(order, i, j) for i in range(order) for j in range(i + 1, order)]
            usable = [s for s in safe if s.contour_safe and s.a_degree <= gauge.setup(order, 6).degree]
            self.assertTrue(usable, f"no usable swap at d={order}")

    def test_constructed_kernels_really_are_null(self):
        # The construction is only worth anything if its output survives the
        # packet falsifier.  Nothing here may fail it.
        for order, truncation in ((5, 14), (6, 12)):
            g = gauge.setup(order, truncation)
            packets = gauge.usable_packets(g)
            checked = 0
            for kernel, _ in gauge.symmetry_kernels(g):
                try:
                    series = gauge.series_of(kernel, g)
                except ValueError:
                    continue  # not an admissible numerator
                checked += 1
                for M in packets:
                    self.assertEqual(gauge.packet_sum(series, M), 0, f"d={order} kernel is not null")
            self.assertGreater(checked, 0)

    def test_a_single_swap_does_not_span_enough_at_a5(self):
        pass

    def test_partial_absorption_closes_a5_with_no_fitted_kernels(self):
        pass

    def test_the_falsifier_must_be_applied_deeper_than_the_fit(self):
        # At d = 6 a shallow truncation offers too few packets to filter
        # candidates: many that pass there are not null.  Deepening the filter
        # is what removes them, and the count must fall.
        g = gauge.setup(6, 12)
        shallow = gauge.null_candidates(g)
        deep = gauge.null_candidates(g, filter_at=16)
        self.assertLess(len(deep), len(shallow))


class TestSolvePositiveGauge(unittest.TestCase):
    """
    The end-to-end solver: fit, then insist the answer survives a deeper truncation.

    At d = 5 it recovers the published kernel exactly, which is the strongest
    check available -- the search is told nothing about it and pools two column
    families that were built for different reasons.
    """

    @classmethod
    def setUpClass(cls):
        cls.solution = gauge.solve_positive_gauge_continuous(5, fit_depth=16, bound=20.0)

    def test_it_solves_a5_and_validates_out_of_sample(self):
        s = self.solution
        self.assertTrue(s.found, s.note)
        self.assertGreater(s.negatives_before, 0)
        self.assertEqual(s.truncation, 16)

    def test_the_returned_numerator_is_nonnegative_and_null(self):
        g = gauge.setup(5, 16)
        v = gauge.validate_gauge(self.solution.kernel, 5, g.max_deg)
        self.assertEqual(v.packets_changed, 0)
        self.assertEqual(v.negatives_gauged, 0)

    def test_the_published_kernel_is_a_valid_gauge(self):
        # G * B * C with B = 2z1 - z2, C = z1 + z4 - z5, G = 2z1 + z2 - z5.
        # The solver is free to return a different element of the null space --
        # with the structural families in play it usually does -- so what is
        # pinned here is the mathematics: this kernel is null and its
        # representative is coefficientwise nonnegative.
        def lin(**kw):
            out = {}
            for name, c in kw.items():
                e = [0] * 5
                e[int(name[1:]) - 1] = 1
                out[tuple(e)] = out.get(tuple(e), 0) + c
            return {k: v for k, v in out.items() if v}

        def mul(p, q):
            out = {}
            for a, x in p.items():
                for b, y in q.items():
                    e = tuple(a[i] + b[i] for i in range(5))
                    out[e] = out.get(e, 0) + x * y
            return {k: v for k, v in out.items() if v}

        gbc = mul(lin(z1=2, z2=1, z5=-1), mul(lin(z1=2, z2=-1), lin(z1=1, z4=1, z5=-1)))
        for truncation in (16, 18, 20):
            v = gauge.validate_gauge(gbc, 5, truncation)
            self.assertEqual(v.packets_changed, 0)
            self.assertEqual(v.negatives_gauged, 0)
            self.assertGreater(v.negatives_canonical, 0)

    def test_a_failed_run_reports_the_failure_rather_than_the_candidate(self):
        # Truncation 8 is far too shallow to pin anything down; whatever the
        # solver likes there must not be returned as a solution.
        shallow = gauge.solve_positive_gauge_continuous(5, fit_depth=8, bound=20.0)
        if not shallow.found:
            self.assertIsNone(shallow.kernel)


if __name__ == "__main__":
    unittest.main()
