"""Tier 2: the mined chamber algebras and the invariants they must satisfy."""

import unittest
from math import comb

from chernpp.artifacts import available_orders, load_algebra
from chernpp.polynomial import is_nonneg, poly_mul, total_degree

#: dim N_d = #{(m, r, l) : 1 <= m <= r, m + r <= l <= d}; also the number of
#: denominator factors in the residue formula.
AMBIENT_DIMENSION = {1: 0, 2: 1, 3: 3, 4: 7, 5: 13, 6: 22, 7: 34}

#: deg Q_d = dim N_d - binomial(d, 2), forced by homogeneity of the residue formula.
MULTIDEGREE_DEGREE = {1: 0, 2: 0, 3: 0, 4: 1, 5: 3, 6: 7, 7: 13}

#: Orders with a mined artifact.  At d <= 3 the degree is 0, so Q_d = 1.
ORDERS = (1, 2, 3, 4, 5, 6, 7)


class TestArtifactsPresent(unittest.TestCase):
    def test_every_expected_order_is_available(self):
        self.assertEqual(available_orders(), [1, 2, 3, 4, 5, 6, 7])

    def test_missing_order_names_the_rebuild_command(self):
        with self.assertRaises(FileNotFoundError) as caught:
            load_algebra(9)
        self.assertIn("multidegree.build", str(caught.exception))


class TestSchemaInvariants(unittest.TestCase):
    def test_validate_passes_for_every_order(self):
        for order in ORDERS:
            with self.subTest(order=order):
                load_algebra(order).validate()

    def test_chamber_variable_count(self):
        for order in ORDERS:
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
        for order in ORDERS:
            algebra = load_algebra(order)
            for index, factor in enumerate(algebra.denominator_factors):
                with self.subTest(order=order, factor=index):
                    self.assertTrue(is_nonneg(factor))
                    self.assertNotIn((0,) * algebra.nvars, factor)

    def test_normalized_numerator_has_constant_term_one(self):
        # This is what makes the c_1^d coefficient of the Thom polynomial 1.
        for order in ORDERS:
            algebra = load_algebra(order)
            with self.subTest(order=order):
                self.assertEqual(algebra.normalized_numerator[(0,) * algebra.nvars], 1)


class TestAssemblyConsistency(unittest.TestCase):
    def test_numerator_is_vandermonde_times_normalized_numerator(self):
        # The stored numerator must be exactly the product it claims to be.
        for order in ORDERS:
            algebra = load_algebra(order)
            with self.subTest(order=order):
                self.assertEqual(
                    poly_mul(algebra.vandermonde, algebra.normalized_numerator),
                    algebra.numerator,
                )

    def test_vandermonde_has_binomial_many_factors(self):
        # prod_{m < l} (1 - z_m/z_l) has binomial(d, 2) factors, hence degree
        # binomial(d, 2) once every z_m/z_l is a chamber monomial of degree >= 1.
        for order in ORDERS:
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
        for order in ORDERS:
            algebra = load_algebra(order)
            for index, factor in enumerate(algebra.denominator_factors):
                with self.subTest(order=order, factor=index):
                    self.assertIn(len(factor), (1, 2))
                    self.assertTrue(all(c in (1, 2) for c in factor.values()))


if __name__ == "__main__":
    unittest.main()


class TestOrbitGeometry(unittest.TestCase):
    """
    The second artifact: geometry of O_d that the residue formula does not use.

    Nothing in the pipeline reads it, so what is checked here is that it is
    internally consistent and consistent with the algebra artifact beside it.
    """

    def setUp(self):
        from chernpp.artifacts import available_geometry_orders

        self.orders = available_geometry_orders()

    def test_every_order_with_an_algebra_has_geometry(self):
        self.assertEqual(self.orders, available_orders())

    def test_validate_passes_for_every_order(self):
        from chernpp.artifacts import load_geometry

        for order in self.orders:
            with self.subTest(order=order):
                load_geometry(order).validate()

    def test_codimension_matches_the_multidegree_degree(self):
        # Same integer as MULTIDEGREE_DEGREE above, arrived at from the ideal
        # rather than from homogeneity of the residue formula.
        from chernpp.artifacts import load_geometry

        for order, expected in MULTIDEGREE_DEGREE.items():
            with self.subTest(order=order):
                self.assertEqual(load_geometry(order).codimension, expected)

    def test_ambient_dimension_matches_the_denominator_count(self):
        from chernpp.artifacts import load_geometry

        for order, expected in AMBIENT_DIMENSION.items():
            with self.subTest(order=order):
                self.assertEqual(load_geometry(order).ambient_dimension, expected)

    def test_degree_is_the_sum_of_component_multiplicities(self):
        # Each component contributes multiplicity times a product of weights,
        # and at z = 1 every weight is 1.  Independent of how the file was made.
        from chernpp.artifacts import load_geometry

        for order in self.orders:
            geometry = load_geometry(order)
            if geometry.codimension == 0:
                continue  # d <= 3: the orbit fills N_d, and there is no degeneration
            with self.subTest(order=order):
                self.assertEqual(sum(m for _, m in geometry.components), geometry.degree)

    def test_known_degrees(self):
        # 2, 6, 55, 957 -- each confirmed three ways when the artifact was written.
        from chernpp.artifacts import load_geometry

        for order, expected in ((4, 2), (5, 6), (6, 55), (7, 957)):
            with self.subTest(order=order):
                self.assertEqual(load_geometry(order).degree, expected)

    def test_only_a5_has_a_reducible_multidegree(self):
        # Q_5 = (2z_1 + z_2 - z_5) P_5 is the exception, not the pattern: Q_4 is
        # a single linear form and Q_6, Q_7 are irreducible.  Worth pinning,
        # because the d = 5 factorisation has been read as a hint about geometry.
        from chernpp.artifacts import load_geometry

        for order in (4, 6, 7):
            with self.subTest(order=order):
                self.assertTrue(load_geometry(order).factors_are_trivial)
        self.assertEqual(sorted(m for _, m in load_geometry(5).factors), [1, 1])
        self.assertEqual(sorted(total_degree(f) for f, _ in load_geometry(5).factors), [1, 2])

    def test_hilbert_numerator_starts_at_one(self):
        # The orbit closure passes through the origin with multiplicity one in
        # degree zero, so the series starts 1 + ...
        from chernpp.artifacts import load_geometry

        for order in self.orders:
            with self.subTest(order=order):
                self.assertEqual(load_geometry(order).hilbert_numerator[0], 1)

    def test_weights_are_the_residue_formula_weights(self):
        # Coordinate q^{mr}_l has weight z_m + z_r - z_l, so each row sums to 1
        # and has entries in {-1, 0, 1, 2}.
        from chernpp.artifacts import load_geometry

        for order in self.orders:
            geometry = load_geometry(order)
            for name, weight in zip(geometry.variables, geometry.weights):
                with self.subTest(order=order, variable=name):
                    self.assertEqual(sum(weight), 1)
                    self.assertTrue(all(-1 <= w <= 2 for w in weight))
