"""Tier 2: the mined chamber algebras and the invariants they must satisfy."""

import unittest
from math import comb

from chernpp.artifacts import available_orders, load_algebra
from chernpp.polynomial import is_nonneg, poly_mul, total_degree

#: dim N_d = #{(m, r, l) : 1 <= m <= r, m + r <= l <= d}; also the number of
#: denominator factors in the residue formula.
AMBIENT_DIMENSION = {4: 7, 5: 13, 6: 22}

#: deg Q_d = dim N_d - binomial(d, 2), forced by homogeneity of the residue formula.
MULTIDEGREE_DEGREE = {4: 1, 5: 3, 6: 7}


class TestArtifactsPresent(unittest.TestCase):
    def test_all_three_orders_are_available(self):
        self.assertEqual(available_orders(), [4, 5, 6])

    def test_missing_order_names_the_rebuild_command(self):
        with self.assertRaises(FileNotFoundError) as caught:
            load_algebra(9)
        self.assertIn("multidegree.build", str(caught.exception))


class TestSchemaInvariants(unittest.TestCase):
    def test_validate_passes_for_every_order(self):
        for order in (4, 5, 6):
            with self.subTest(order=order):
                load_algebra(order).validate()

    def test_chamber_variable_count(self):
        for order in (4, 5, 6):
            with self.subTest(order=order):
                self.assertEqual(load_algebra(order).nvars, order - 1)

    def test_denominator_factor_count_matches_ambient_dimension(self):
        for order, expected in AMBIENT_DIMENSION.items():
            with self.subTest(order=order):
                self.assertEqual(len(load_algebra(order).denominator_factors), expected)

    def test_denominator_factors_admit_a_positive_geometric_series(self):
        # Every 1/(1 - f_r) must expand with nonnegative coefficients, which
        # needs f_r nonnegative with zero constant term.  All the positivity
        # arguments in the package rest on this.
        for order in (4, 5, 6):
            algebra = load_algebra(order)
            for index, factor in enumerate(algebra.denominator_factors):
                with self.subTest(order=order, factor=index):
                    self.assertTrue(is_nonneg(factor))
                    self.assertNotIn((0,) * algebra.nvars, factor)

    def test_normalized_numerator_has_constant_term_one(self):
        # This is what makes the c_1^d coefficient of the Thom polynomial 1.
        for order in (4, 5, 6):
            algebra = load_algebra(order)
            with self.subTest(order=order):
                self.assertEqual(algebra.normalized_numerator[(0,) * algebra.nvars], 1)


class TestAssemblyConsistency(unittest.TestCase):
    def test_numerator_is_vandermonde_times_normalized_numerator(self):
        # The stored numerator must be exactly the product it claims to be.
        for order in (4, 5, 6):
            algebra = load_algebra(order)
            with self.subTest(order=order):
                self.assertEqual(
                    poly_mul(algebra.vandermonde, algebra.normalized_numerator),
                    algebra.numerator,
                )

    def test_vandermonde_has_binomial_many_factors(self):
        # prod_{m < l} (1 - z_m/z_l) has binomial(d, 2) factors, hence degree
        # binomial(d, 2) once every z_m/z_l is a chamber monomial of degree >= 1.
        for order in (4, 5, 6):
            algebra = load_algebra(order)
            with self.subTest(order=order):
                self.assertEqual(algebra.vandermonde[(0,) * algebra.nvars], 1)
                self.assertGreaterEqual(total_degree(algebra.vandermonde), comb(order, 2))

    def test_multidegree_degree_matches_homogeneity(self):
        # Q_d in chamber coordinates is a specialisation of a homogeneous
        # polynomial of degree dim N_d - binomial(d, 2).
        for order, degree in MULTIDEGREE_DEGREE.items():
            algebra = load_algebra(order)
            with self.subTest(order=order):
                self.assertEqual(AMBIENT_DIMENSION[order] - comb(order, 2), degree)
                self.assertGreaterEqual(total_degree(algebra.multidegree), degree)

    def test_denominator_factors_are_binomials_of_chamber_monomials(self):
        # Each factor is z_m/z_l + z_r/z_l, so it has one or two terms, all
        # with coefficient 1 or 2 (the two coincide when m = r).
        for order in (4, 5, 6):
            algebra = load_algebra(order)
            for index, factor in enumerate(algebra.denominator_factors):
                with self.subTest(order=order, factor=index):
                    self.assertIn(len(factor), (1, 2))
                    self.assertTrue(all(c in (1, 2) for c in factor.values()))


if __name__ == "__main__":
    unittest.main()
