"""
Tier 4: the chamber series, the two positivity conjectures, and the reductions.

The centrepiece is :class:`TestCrossValidation`, which computes the same Chern
coefficients along two entirely independent paths -- a truncated series product
in exact Python integers, and an XLA fixed-point iteration on an integer grid.
"""

import unittest

from chernpp.artifacts import load_algebra
from chernpp.chamber import (
    ballot_orderings,
    chamber_series,
    chern_coefficient,
    monomial,
    paired_defects,
    sorted_negatives,
    tau,
    unpaired_tail_defects,
)
from chernpp.chern import chern_coefficients
from chernpp.lorentzian import check_strong_log_concavity, extract_log_concavity_sequence
from chernpp.polynomial import is_nonneg, poly_mul, poly_sub


class TestStrongLaurentPositivity(unittest.TestCase):
    """True at d = 4, false from d = 5 on."""

    def test_a4_is_coefficientwise_nonnegative(self):
        self.assertTrue(is_nonneg(chamber_series(4, 14)))

    def test_a5_first_counterexample(self):
        # A_(1,1,2,1) = -1 is the negative coefficient of lowest total degree.
        negatives = sorted_negatives(chamber_series(5, 12))
        self.assertEqual(negatives[0], ((1, 1, 2, 1), -1))

    def test_a6_has_the_analogous_counterexample(self):
        negatives = sorted_negatives(chamber_series(6, 10))
        self.assertEqual(negatives[0], ((1, 1, 2, 1, 0), -1))

    def test_series_satisfies_its_defining_identity(self):
        # F * prod(1 - f_r) must return the numerator, degree by degree.
        for order, cap in ((4, 9), (5, 8)):
            algebra = load_algebra(order)
            series = chamber_series(order, cap, algebra=algebra)
            recovered = series
            for factor in algebra.denominator_factors:
                recovered = poly_sub(recovered, poly_mul(recovered, factor, cap))
            residual = poly_sub(recovered, algebra.numerator)
            with self.subTest(order=order):
                self.assertFalse({e: c for e, c in residual.items() if sum(e) <= cap})


class TestBallotCombinatorics(unittest.TestCase):
    def test_tau_is_an_involution_where_defined(self):
        for beta in ((1, 3, 2, 1), (0, 4, 1, 1), (2, 2, 0, 0)):
            partner = tau(beta)
            with self.subTest(beta=beta):
                self.assertIsNotNone(partner)
                self.assertEqual(tau(partner), beta)

    def test_tau_is_undefined_past_the_ballot_boundary(self):
        self.assertIsNone(tau((3, 1, 0, 0)))

    def test_tau_fixes_all_later_partial_sums(self):
        beta = (1, 3, 5, 2)
        self.assertEqual(tau(beta)[1:], beta[1:])

    def test_ballot_orderings_have_nonnegative_partial_sums(self):
        for _, betas in ballot_orderings((1, 1, 0, -1, -1)):
            with self.subTest(betas=betas):
                self.assertTrue(all(b >= 0 for b in betas))

    def test_ballot_orderings_are_distinct(self):
        orderings = ballot_orderings((1, 1, 0, -1, -1))
        self.assertEqual(len({a for a, _ in orderings}), len(orderings))

    def test_nonzero_sum_is_rejected(self):
        with self.assertRaises(ValueError):
            ballot_orderings((1, 1, 1))

    def test_paper_table_is_reproduced(self):
        # Ten ballot orderings of {1,1,0,-1,-1}; the A_beta sum to 10 despite
        # one of them being the -1 counterexample.
        series = chamber_series(5, 12)
        orderings = ballot_orderings((1, 1, 0, -1, -1))
        self.assertEqual(len(orderings), 10)
        values = sorted(series.get(beta, 0) for _, beta in orderings)
        self.assertEqual(values, [-1, 0, 0, 0, 0, 0, 1, 2, 2, 6])
        self.assertEqual(chern_coefficient(series, (1, 1, 0, -1, -1), 12), 10)

    def test_truncation_is_reported_not_silently_undercounted(self):
        with self.assertRaises(ValueError):
            chern_coefficient(chamber_series(5, 6), (4, 0, 0, 0, -4), 6)


class TestCrossValidation(unittest.TestCase):
    """The two independent implementations of C(M) must agree exactly."""

    def test_chern_coefficients_agree_between_series_and_grid(self):
        for order, cap, l_max in ((4, 12, 1), (5, 12, 1)):
            series = chamber_series(order, cap)
            _, grid_values = chern_coefficients(order, l_max=l_max)
            compared = 0
            for multiset, expected in grid_values.items():
                try:
                    viaseries = chern_coefficient(series, multiset, cap)
                except ValueError:
                    continue  # outside the truncated range; not a disagreement
                with self.subTest(order=order, multiset=multiset):
                    self.assertEqual(viaseries, expected)
                compared += 1
            with self.subTest(order=order):
                self.assertGreater(compared, 5, "cross-check covered too few monomials")


class TestReductions(unittest.TestCase):
    """The A_5 reduction strategy, and how far it reaches at d = 6."""

    def test_unpaired_tail_holds_for_a5_and_a6(self):
        for order, cap in ((5, 14), (6, 11)):
            with self.subTest(order=order):
                self.assertEqual(unpaired_tail_defects(chamber_series(order, cap), cap), [])

    def test_paired_inequality_holds_for_a5_and_a6(self):
        for order, cap in ((5, 14), (6, 11)):
            with self.subTest(order=order):
                self.assertEqual(paired_defects(chamber_series(order, cap), cap), [])

    def test_prefix_positivity_holds_for_a5(self):
        prefix = chamber_series(5, 14, extra_factors=[monomial(4, 0)])
        self.assertTrue(is_nonneg(prefix))

    def test_prefix_positivity_fails_for_a6(self):
        # The prefix sum runs over r <= i, so a negative coefficient at i = 0
        # passes through untouched.  A_5 has none; A_6 does.
        base = chamber_series(6, 10)
        self.assertEqual(
            [b for b, _ in sorted_negatives(base) if b[0] == 0], [(0, 2, 3, 2, 2)]
        )

        prefix = chamber_series(6, 10, extra_factors=[monomial(5, 0)])
        self.assertEqual(prefix[(0, 2, 3, 2, 2)], -1)

    def test_a5_has_no_negative_coefficient_at_i_zero(self):
        self.assertEqual(
            [b for b, _ in sorted_negatives(chamber_series(5, 14)) if b[0] == 0], []
        )


class TestLorentzian(unittest.TestCase):
    def test_positivity_is_not_explained_by_log_concavity(self):
        grid, _ = chern_coefficients(5, l_max=2)
        sequence = extract_log_concavity_sequence(grid, 2, 1, 1)
        self.assertFalse(check_strong_log_concavity(sequence))


if __name__ == "__main__":
    unittest.main()
