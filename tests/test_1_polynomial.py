"""Tier 1: the exact sparse polynomial layer everything else is built on."""

import unittest
from fractions import Fraction
from math import comb

from chernpp.polynomial import (
    divide_by_one_minus,
    expand_rational,
    is_nonneg,
    monomials_up_to,
    negative_terms,
    one_minus,
    poly_add,
    poly_mul,
    poly_mul_many,
    poly_one,
    poly_sub,
    poly_to_string,
    total_degree,
)


class TestRingLaws(unittest.TestCase):
    def setUp(self):
        # p = 1 + 2x - y,  q = x + 3y^2
        self.p = {(0, 0): 1, (1, 0): 2, (0, 1): -1}
        self.q = {(1, 0): 1, (0, 2): 3}
        self.r = {(0, 1): 5, (2, 1): -2}

    def test_multiplication_is_commutative_and_associative(self):
        self.assertEqual(poly_mul(self.p, self.q), poly_mul(self.q, self.p))
        self.assertEqual(
            poly_mul(poly_mul(self.p, self.q), self.r),
            poly_mul(self.p, poly_mul(self.q, self.r)),
        )

    def test_distributivity(self):
        self.assertEqual(
            poly_mul(self.p, poly_add(self.q, self.r)),
            poly_add(poly_mul(self.p, self.q), poly_mul(self.p, self.r)),
        )

    def test_identity_and_inverse(self):
        self.assertEqual(poly_mul(self.p, poly_one(2)), self.p)
        self.assertEqual(poly_sub(self.p, self.p), {})

    def test_zero_coefficients_are_never_stored(self):
        # (1 + x) - (1 + x) must be the empty dict, not {e: 0}.
        self.assertEqual(poly_sub({(0, 0): 1, (1, 0): 1}, {(0, 0): 1, (1, 0): 1}), {})
        self.assertNotIn((0, 0), poly_add({(0, 0): 3}, {(0, 0): -3}))

    def test_explicit_product(self):
        # (1 + 2x - y)(x + 3y^2) = x + 2x^2 - xy + 3y^2 + 6xy^2 - 3y^3
        self.assertEqual(
            poly_mul(self.p, self.q),
            {(1, 0): 1, (2, 0): 2, (1, 1): -1, (0, 2): 3, (1, 2): 6, (0, 3): -3},
        )


class TestTruncation(unittest.TestCase):
    def test_truncated_product_agrees_with_full_product(self):
        p = {(2, 0): 1, (1, 1): 2, (0, 3): -1}
        q = {(1, 1): 3, (0, 1): 1, (4, 0): 5}
        full = poly_mul(p, q)
        for cap in range(0, 9):
            truncated = poly_mul(p, q, max_deg=cap)
            expected = {e: c for e, c in full.items() if sum(e) <= cap}
            self.assertEqual(truncated, expected, f"truncation at {cap}")

    def test_monomial_count_matches_binomial(self):
        for nvars in (1, 3, 4):
            for deg in range(5):
                self.assertEqual(len(monomials_up_to(nvars, deg)), comb(deg + nvars, nvars))

    def test_monomials_are_distinct_and_degree_ordered(self):
        monomials = monomials_up_to(3, 4)
        self.assertEqual(len(set(monomials)), len(monomials))
        self.assertEqual([sum(m) for m in monomials], sorted(sum(m) for m in monomials))


class TestGeometricSeries(unittest.TestCase):
    def test_division_inverts_multiplication(self):
        f = {(1, 0): 1, (0, 1): 2}
        p = {(0, 0): 1, (2, 1): -4}
        cap = 7
        quotient = divide_by_one_minus(p, f, cap)
        recovered = poly_mul(quotient, one_minus(f, 2), cap)
        self.assertEqual(recovered, {e: c for e, c in p.items() if sum(e) <= cap})

    def test_geometric_series_of_a_single_variable(self):
        # 1/(1 - x) = 1 + x + x^2 + ...
        series = divide_by_one_minus(poly_one(1), {(1,): 1}, 5)
        self.assertEqual(series, {(k,): 1 for k in range(6)})

    def test_constant_term_is_rejected(self):
        with self.assertRaises(ValueError):
            divide_by_one_minus(poly_one(2), {(0, 0): 1, (1, 0): 1}, 3)

    def test_rational_expansion_is_exact_on_a_known_case(self):
        # 1 / ((1-x)(1-y)) has every coefficient equal to 1.
        series = expand_rational(poly_one(2), [{(1, 0): 1}, {(0, 1): 1}], 4)
        self.assertTrue(all(c == 1 for c in series.values()))
        self.assertEqual(len(series), comb(4 + 2, 2))


class TestPredicatesAndRendering(unittest.TestCase):
    def test_sign_predicates(self):
        self.assertTrue(is_nonneg({(0,): 0, (1,): 3}))
        self.assertFalse(is_nonneg({(1,): -1}))
        self.assertEqual(negative_terms({(0,): 2, (1,): -5}), {(1,): -5})

    def test_total_degree_of_empty_polynomial(self):
        self.assertEqual(total_degree({}), 0)

    def test_fractions_are_supported(self):
        half = {(1, 0): Fraction(1, 2)}
        self.assertEqual(poly_mul(half, half), {(2, 0): Fraction(1, 4)})
        self.assertTrue(is_nonneg(poly_mul(half, half)))

    def test_rendering_round_trips_signs(self):
        text = poly_to_string({(0, 0): 1, (1, 0): -1, (0, 2): 3}, ("a", "b"))
        self.assertIn("- a", text)
        self.assertIn("3*b^2", text)
        self.assertEqual(poly_to_string({}, ("a",)), "0")

    def test_product_of_many_is_order_independent(self):
        factors = [{(1, 0): 1, (0, 0): 1}, {(0, 1): 2, (0, 0): 1}, {(1, 1): 1, (0, 0): 1}]
        self.assertEqual(poly_mul_many(factors, 2), poly_mul_many(list(reversed(factors)), 2))


class TestArtifactGuards(unittest.TestCase):
    """
    The refusals standing between a bad artifact and everything downstream.

    Every other tier loads good artifacts and checks they validate. These are the
    paths that fire when one is not good, and until now none of them was covered:
    a guard nothing exercises is a guard nobody knows is wired up.
    """

    def algebra(self, **overrides):
        from chernpp.artifacts import ChamberAlgebra

        fields = dict(
            order=4,
            chamber_vars=("x1", "x2", "x3"),
            numerator={(0, 0, 0): 1},
            denominator_factors=[{(1, 0, 0): 1}],
            multidegree={(0, 0, 0): 1},
            normalized_numerator={(0, 0, 0): 1},
            vandermonde={(0, 0, 0): 1},
        )
        fields.update(overrides)
        return ChamberAlgebra(**fields)

    def test_a_good_artifact_validates(self):
        self.algebra().validate()

    def test_a_mod_p_artifact_is_refused(self):
        # `build.py -p 32003` writes into the same directory, and everything
        # downstream assumes exact integers.
        with self.assertRaises(ValueError) as caught:
            self.algebra(characteristic=32003).validate()
        self.assertIn("characteristic", str(caught.exception))

    def test_wrong_chamber_variable_count_is_refused(self):
        with self.assertRaises(ValueError):
            self.algebra(chamber_vars=("x1", "x2")).validate()

    def test_numerator_without_constant_term_one_is_refused(self):
        # This is what makes the c_1^d coefficient of the Thom polynomial 1.
        with self.assertRaises(ValueError):
            self.algebra(normalized_numerator={(1, 0, 0): 1}).validate()

    def test_negative_denominator_factor_is_refused(self):
        # Every positivity argument rests on 1/(1 - f_r) expanding nonnegatively.
        with self.assertRaises(ValueError):
            self.algebra(denominator_factors=[{(1, 0, 0): -1}]).validate()

    def test_denominator_factor_with_a_constant_term_is_refused(self):
        with self.assertRaises(ValueError):
            self.algebra(denominator_factors=[{(0, 0, 0): 1, (1, 0, 0): 1}]).validate()

    def test_coefficients_too_large_for_int64_are_refused_not_widened(self):
        # An object array would round-trip only with allow_pickle=True, which is
        # the property the format exists to avoid.
        from chernpp.artifacts import pack_polynomial

        with self.assertRaises(OverflowError) as caught:
            pack_polynomial({(0, 0): 2**64}, 2)
        self.assertIn("int64", str(caught.exception))

    def test_polynomials_round_trip_through_the_packed_form(self):
        from chernpp.artifacts import pack_polynomial, unpack_polynomial

        for poly in ({}, {(0, 0): 1}, {(2, 1): -7, (0, 3): 196803, (1, 1): 2}):
            with self.subTest(poly=poly):
                exponents, coefficients = pack_polynomial(poly, 2)
                self.assertEqual(unpack_polynomial(exponents, coefficients), poly)

    def test_polynomial_lists_round_trip_including_the_empty_case(self):
        from chernpp.artifacts import pack_polynomial_list, unpack_polynomial_list

        for polys in ([], [{}], [{(1, 0): 1}, {}, {(0, 2): 3}]):
            with self.subTest(polys=polys):
                packed = pack_polynomial_list(polys, 2)
                self.assertEqual(unpack_polynomial_list(*packed), polys)

    def test_values_survive_as_python_ints_not_numpy_scalars(self):
        # A numpy int64 leaking into the exact arithmetic downstream would wrap
        # silently where a Python int would not.
        from chernpp.artifacts import pack_polynomial, unpack_polynomial

        recovered = unpack_polynomial(*pack_polynomial({(1, 1): 5}, 2))
        for exponents, coefficient in recovered.items():
            self.assertIs(type(coefficient), int)
            self.assertTrue(all(type(e) is int for e in exponents))


class TestExactnessOfTheFastPath(unittest.TestCase):
    """
    The vectorised product is exact.

    A float64 path was once the default here: it rounded silently, dropped
    coefficients below 1e-10 and returned numpy floats, so a chamber series was
    only as exact as float64 happened to be.  These pin the repair.
    """

    def _python_product(self, p, q, max_deg):
        out = {}
        for e1, c1 in p.items():
            for e2, c2 in q.items():
                if sum(e1) + sum(e2) <= max_deg:
                    e = tuple(a + b for a, b in zip(e1, e2))
                    out[e] = out.get(e, 0) + c1 * c2
        return {e: c for e, c in out.items() if c}

    def test_fast_product_matches_python_and_returns_ints(self):
        import random

        rng = random.Random(7)
        for nvars in (2, 4, 6):
            p = {tuple(rng.randrange(4) for _ in range(nvars)): rng.randrange(-900, 900) for _ in range(60)}
            q = {tuple(rng.randrange(4) for _ in range(nvars)): rng.randrange(-900, 900) for _ in range(60)}
            p = {e: c for e, c in p.items() if c}
            q = {e: c for e, c in q.items() if c}
            fast = poly_mul(p, q, max_deg=7)
            self.assertEqual(fast, self._python_product(p, q, 7))
            self.assertTrue(all(type(c) is int for c in fast.values()))

    def test_huge_coefficients_fall_back_to_python_integers(self):
        big = 3**50
        p = {(0, 1): big, (1, 0): -big}
        q = {(1, 1): big, (0, 0): 1}
        self.assertEqual(poly_mul(p, q, max_deg=5), self._python_product(p, q, 5))

    def test_a_float_on_the_default_path_is_refused(self):
        with self.assertRaises(TypeError):
            poly_mul({(0,): 0.5}, {(1,): 1}, max_deg=3)

    def test_floats_are_only_used_when_asked_for(self):
        self.assertEqual(poly_mul({(0,): 0.5}, {(1,): 2.0}, max_deg=3, exact=False), {(1,): 1.0})

    def test_fractions_stay_exact(self):
        out = poly_mul({(0,): Fraction(1, 3)}, {(1,): Fraction(3, 7)}, max_deg=3)
        self.assertEqual(out, {(1,): Fraction(1, 7)})

    def test_chamber_series_and_chern_coefficient_are_exact_integers(self):
        from chernpp.chamber import chamber_series, chern_coefficient

        series = chamber_series(5, 10)
        self.assertTrue(all(type(c) is int for c in series.values()))
        value = chern_coefficient(series, (1, 1, 0, -1, -1), 10)
        self.assertIs(type(value), int)
        self.assertEqual(value, 10)


if __name__ == "__main__":
    unittest.main()
