"""
Tier 5: denominator certificates and the obstructions to their existence.

A certificate returned by the engine is a proof, so the tests here check both
directions: that a genuine certificate verifies against the exact identity, and
that no certificate is ever manufactured for something false.
"""

import unittest
from fractions import Fraction

from chernpp.artifacts import load_algebra
from chernpp.certificates import (
    Certificate,
    certificate_subsets,
    minimum_order,
    minimum_order as least_unobstructed_order,
    projection_is_feasible,
    search_certificate,
)
from chernpp.chamber import monomial
from chernpp.polynomial import is_nonneg


class TestSubsetEnumeration(unittest.TestCase):
    def test_counts_match_binomials(self):
        from math import comb

        for order in (1, 2, 3):
            subsets = certificate_subsets(6, order)
            expected = sum(comb(6, k) for k in range(1, order + 1))
            with self.subTest(order=order):
                self.assertEqual(len(subsets), expected)

    def test_subsets_are_sorted_and_distinct(self):
        subsets = certificate_subsets(5, 3)
        self.assertEqual(len(set(subsets)), len(subsets))
        self.assertTrue(all(list(s) == sorted(s) for s in subsets))


class TestCertificateVerification(unittest.TestCase):
    def test_a4_certificate_exists_and_verifies(self):
        algebra = load_algebra(4)
        certificate = search_certificate(
            algebra.numerator,
            algebra.denominator_factors,
            algebra.nvars,
            order=4,
            max_degree=8,
            varnames=algebra.chamber_vars,
        )
        self.assertIsNotNone(certificate, "A_4 admits an order-4 certificate at degree 8")
        # Re-derive from scratch: exact identity plus nonnegativity of each part.
        self.assertTrue(certificate.is_valid(algebra.numerator, algebra.denominator_factors))
        self.assertTrue(all(is_nonneg(part) for part in certificate.parts.values()))
        self.assertFalse(certificate.residual(algebra.numerator, algebra.denominator_factors))

    def test_certificate_coefficients_are_exact_rationals(self):
        algebra = load_algebra(4)
        certificate = search_certificate(
            algebra.numerator,
            algebra.denominator_factors,
            algebra.nvars,
            order=4,
            max_degree=8,
            varnames=algebra.chamber_vars,
        )
        for part in certificate.parts.values():
            for coefficient in part.values():
                self.assertIsInstance(coefficient, (int, Fraction))

    def test_tampered_certificate_is_rejected(self):
        # Guards the verifier itself: perturbing one coefficient must break it.
        algebra = load_algebra(4)
        certificate = search_certificate(
            algebra.numerator,
            algebra.denominator_factors,
            algebra.nvars,
            order=4,
            max_degree=8,
            varnames=algebra.chamber_vars,
        )
        parts = {key: dict(value) for key, value in certificate.parts.items()}
        target = next(key for key, value in parts.items() if key and value)
        exponent = next(iter(parts[target]))
        parts[target][exponent] += 1

        tampered = Certificate(
            parts,
            algebra.nvars,
            algebra.chamber_vars,
            certificate.order,
            certificate.max_degree,
        )
        self.assertFalse(tampered.is_valid(algebra.numerator, algebra.denominator_factors))

    def test_negative_part_is_rejected(self):
        algebra = load_algebra(4)
        parts = {(): {(0, 0, 0): -1}}
        certificate = Certificate(parts, 3, algebra.chamber_vars, 0, 0)
        self.assertFalse(certificate.is_valid(algebra.numerator, algebra.denominator_factors))


class TestOrderObstructions(unittest.TestCase):
    def test_a4_minimum_order_is_exactly_four(self):
        algebra = load_algebra(4)
        # Infeasibility of the degree-<=2 projection proves that no certificate
        # of that order exists at ANY degree, since a constraint on a monomial
        # of degree T involves only the P_S coefficients of degree <= T.
        for order in (1, 2, 3):
            with self.subTest(order=order):
                self.assertFalse(
                    projection_is_feasible(
                        algebra.numerator,
                        algebra.denominator_factors,
                        algebra.nvars,
                        order,
                        probe_degree=2,
                    )
                )
        self.assertEqual(
            minimum_order(
                algebra.numerator,
                algebra.denominator_factors,
                algebra.nvars,
                probe_degree=2,
            ),
            4,
        )

    def test_deeper_probes_give_stronger_bounds(self):
        # The projection at depth T+1 refines the one at depth T, so the
        # reported least unobstructed order is non-decreasing in T.
        algebra = load_algebra(5)
        bounds = [
            least_unobstructed_order(
                algebra.numerator,
                algebra.denominator_factors,
                algebra.nvars,
                probe_degree=depth,
                max_order=5,
            )
            for depth in (1, 2)
        ]
        self.assertEqual(bounds[0], 4)
        self.assertIsNone(bounds[1], "depth-2 probe rules out every order up to 5")

    def test_a5_prefix_series_is_obstructed(self):
        # F_5/(1-a) appears coefficientwise nonnegative, yet admits no
        # certificate of low order at any degree.
        algebra = load_algebra(5)
        factors = algebra.denominator_factors + [monomial(algebra.nvars, 0)]
        self.assertIsNone(
            minimum_order(algebra.numerator, factors, algebra.nvars, probe_degree=2, max_order=5)
        )


class TestNoFalsePositives(unittest.TestCase):
    def test_no_certificate_for_a5(self):
        # F_5 has a negative coefficient, so no certificate can exist at any
        # order or degree.  The engine must never manufacture one.
        algebra = load_algebra(5)
        for order in (1, 2, 3):
            with self.subTest(order=order):
                self.assertIsNone(
                    search_certificate(
                        algebra.numerator,
                        algebra.denominator_factors,
                        algebra.nvars,
                        order=order,
                        max_degree=6,
                        varnames=algebra.chamber_vars,
                    )
                )

    def test_order_zero_requires_a_nonnegative_numerator(self):
        algebra = load_algebra(4)
        self.assertIsNone(
            search_certificate(algebra.numerator, algebra.denominator_factors, algebra.nvars, order=0)
        )

    def test_order_zero_succeeds_when_the_numerator_is_already_nonnegative(self):
        numerator = {(0, 0): 1, (1, 0): 2}
        factors = [{(1, 0): 1}, {(0, 1): 1}]
        certificate = search_certificate(numerator, factors, 2, order=0)
        self.assertIsNotNone(certificate)
        self.assertTrue(certificate.is_valid(numerator, factors))

    def test_invalid_denominator_factors_are_refused(self):
        with self.assertRaises(ValueError):
            search_certificate({(0,): 1}, [{(1,): -1}], 1, order=1)
        with self.assertRaises(ValueError):
            search_certificate({(0,): 1}, [{(0,): 1}], 1, order=1)


class TestTheClaimTheReadmeMakes(unittest.TestCase):
    """
    The published lower bound, at the order and depth it is published at.

    The rest of this tier probes to order 5, which is enough to exercise the
    machinery but one short of the claim in the README and the report. Fourteen
    seconds buys the actual statement.
    """

    def test_nothing_of_order_at_most_seven_certifies_a5(self):
        from chernpp.chamber import monomial

        algebra = load_algebra(5)
        prefix = algebra.denominator_factors + [monomial(algebra.nvars, 0)]
        for label, factors in (("F_5", algebra.denominator_factors), ("F_5/(1-a)", prefix)):
            with self.subTest(series=label):
                self.assertIsNone(
                    minimum_order(algebra.numerator, factors, algebra.nvars, probe_degree=3, max_order=7),
                    f"{label}: an order <= 7 came back unobstructed",
                )


if __name__ == "__main__":
    unittest.main()
