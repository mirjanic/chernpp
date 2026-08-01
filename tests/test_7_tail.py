"""
Tier 7: the unpaired-tail series and multiplicative (Lemma 1) certificates.

The whole unpaired tail ``i > j ==> A_{i,j,...} >= 0`` collapses onto one
series in ``d - 2`` variables, ``J_d(1/2, ...)``.  For ``d = 5`` the handoff
note proves it nonnegative by exhibiting six manifestly nonnegative ratios
(its equation 9); the tests below check that our construction reproduces that
closed form exactly, and that the Lemma 1 search rediscovers the proof.
"""

import unittest
from fractions import Fraction

from chernpp import lemma1
from chernpp.artifacts import load_algebra
from chernpp.chamber import (
    tail_target,
    tail_target_factored,
    vandermonde_factors,
)
from chernpp.polynomial import (
    expand_rational,
    is_nonneg,
    one_minus,
    poly_mul,
    poly_mul_many,
    poly_sub,
)

HALF = Fraction(1, 2)

#: Equation (9) of the handoff note, in the variables (b, c, e): J_5(1/2) as a
#: product of six ratios (1 - u)/(1 - v), each nonnegative by Lemma 1.
EQ9_NUMERATORS = [
    {(0, 0, 0): Fraction(1), (0, 0, 1): Fraction(-1)},  # 1 - e
    {(0, 0, 0): Fraction(1), (0, 1, 0): Fraction(-1)},  # 1 - c
    {(0, 0, 0): Fraction(1), (0, 1, 1): Fraction(-1)},  # 1 - ce
    {(0, 0, 0): Fraction(1), (1, 0, 0): -HALF},  # 1 - b/2
    {(0, 0, 0): Fraction(1), (1, 1, 0): -HALF},  # 1 - bc/2
    {(0, 0, 0): Fraction(1), (1, 1, 1): -HALF},  # 1 - bce/2
]
EQ9_DENOMINATORS = [
    {(0, 0, 1): Fraction(1), (1, 1, 1): HALF},  # e + bce/2
    {(0, 1, 0): Fraction(1), (1, 1, 0): HALF},  # c + bc/2
    {(0, 1, 1): Fraction(1), (1, 1, 1): HALF},  # ce + bce/2
    {(1, 0, 0): Fraction(3, 2)},  # 3b/2
    {(1, 1, 0): Fraction(3, 2)},  # 3bc/2
    {(1, 1, 1): Fraction(3, 2)},  # 3bce/2
]


class TestVandermondeFactorisation(unittest.TestCase):
    def test_factor_count_is_binomial(self):
        from math import comb

        for order in (4, 5, 6):
            with self.subTest(order=order):
                self.assertEqual(len(vandermonde_factors(order)), comb(order, 2))

    def test_product_reproduces_the_stored_vandermonde(self):
        for order in (4, 5, 6):
            algebra = load_algebra(order)
            product = poly_mul_many(
                [one_minus(u, algebra.nvars) for u in vandermonde_factors(order)],
                algebra.nvars,
            )
            with self.subTest(order=order):
                self.assertEqual(product, algebra.vandermonde)


class TestTailTarget(unittest.TestCase):
    def test_factored_and_flat_forms_agree(self):
        for order in (5, 6):
            factors, residual, denominators, varnames = tail_target_factored(order)
            flat_numerator, flat_denominators, _ = tail_target(order)
            rebuilt = poly_mul(
                residual,
                poly_mul_many(
                    [one_minus(u, len(varnames)) for u in factors], len(varnames)
                ),
            )
            with self.subTest(order=order):
                self.assertEqual(rebuilt, flat_numerator)
                self.assertEqual(denominators, flat_denominators)

    def test_a5_reproduces_published_closed_form(self):
        # Our J_5(1/2, b, c, e) must equal equation (9) coefficient for
        # coefficient.  This is the check that validates the construction.
        cap = 12
        numerator, denominators, _ = tail_target(5)
        ours = expand_rational(numerator, denominators, cap)
        theirs = expand_rational(
            poly_mul_many(EQ9_NUMERATORS, 3, cap), EQ9_DENOMINATORS, cap
        )
        self.assertEqual(ours, theirs)

    def test_tail_series_is_nonnegative_in_range(self):
        for order, cap in ((5, 12), (6, 10)):
            numerator, denominators, _ = tail_target(order)
            with self.subTest(order=order):
                self.assertTrue(is_nonneg(expand_rational(numerator, denominators, cap)))


class TestLemma1Machinery(unittest.TestCase):
    def test_dominates(self):
        self.assertTrue(lemma1.dominates({(1,): 2}, {(1,): 1}))
        self.assertFalse(lemma1.dominates({(1,): 1}, {(1,): 2}))

    def test_divide_exactly_recovers_a_known_quotient(self):
        # (1 - v)(1 + v) = 1 - v^2, so dividing back must return 1 + v.
        v = {(1, 0): 1, (0, 1): 1}
        product = poly_mul(one_minus(v, 2), {(0, 0): 1, (1, 0): 1, (0, 1): 1})
        quotient = lemma1.divide_exactly(product, v, 2)
        self.assertEqual(quotient, {(0, 0): 1, (1, 0): 1, (0, 1): 1})

    def test_divide_exactly_refuses_a_non_multiple(self):
        self.assertIsNone(lemma1.divide_exactly({(0,): 1, (1,): 1}, {(1,): 3}, 1))

    def test_divide_exactly_verifies_by_multiplying_back(self):
        # Anything it returns must satisfy quotient * (1 - v) == p exactly.
        v = {(1,): 1}
        p = poly_mul(one_minus(v, 1), {(0,): 5, (2,): 3})
        quotient = lemma1.divide_exactly(p, v, 1)
        self.assertEqual(poly_mul(quotient, one_minus(v, 1)), p)


class TestUnpairedTailProof(unittest.TestCase):
    def test_a5_tail_is_proved_mechanically(self):
        # This reconstructs Proposition 3 of the handoff note: every factor is
        # paired off by Lemma 1 and the leftover numerator is exactly 1.
        factors, residual, denominators, varnames = tail_target_factored(5)
        certificate = lemma1.search(
            factors, residual, denominators, len(varnames), varnames
        )
        self.assertTrue(certificate.proved)
        self.assertEqual(certificate.leftover_numerator, {(0, 0, 0): 1})
        self.assertEqual(certificate.leftover_denominators, [])

    def test_a6_reduces_to_a_single_explicit_residual(self):
        # The same search does not close d = 6.  It leaves precisely
        #     (1 - cde - (3/2) bcde) / (1 - 2cde),
        # whose expansion has negative coefficients, so the pairing is a
        # decomposition rather than a proof.  Pinned here so that any change to
        # the matching strategy shows up immediately.
        factors, residual, denominators, varnames = tail_target_factored(6)
        certificate = lemma1.search(
            factors, residual, denominators, len(varnames), varnames
        )
        self.assertFalse(certificate.proved)
        self.assertEqual(
            certificate.leftover_numerator,
            {(0, 0, 0, 0): 1, (0, 1, 1, 1): -1, (1, 1, 1, 1): Fraction(-3, 2)},
        )
        self.assertEqual(certificate.leftover_denominators, [{(0, 1, 1, 1): 2}])

        # And Lemma 1 genuinely cannot finish it: v - u has a negative term.
        u = {e: -c for e, c in certificate.leftover_numerator.items() if sum(e) > 0}
        for v in certificate.leftover_denominators:
            self.assertFalse(is_nonneg(poly_sub(v, u)))


class TestAbsorption(unittest.TestCase):
    """The absorption criterion, a strict generalisation of Lemma 1."""

    def test_absorption_subsumes_lemma_1(self):
        # Lemma 1 is the case N = 1 - u, S = {v}, condition u <= v.
        one_minus_u = {(0,): 1, (1,): -1}
        self.assertTrue(lemma1.absorbs(one_minus_u, [{(1,): 2}], 1))
        self.assertFalse(lemma1.absorbs({(0,): 1, (1,): -2}, [{(1,): 1}], 1))

    def test_absorption_is_strictly_stronger(self):
        # N = 1 + x - 2x^2 has a negative part not of the form (1 - u), so
        # Lemma 1 does not apply, but N_+ * v dominates N_- for v = x.
        numerator = {(0,): 1, (1,): 1, (2,): -2}
        self.assertTrue(lemma1.absorbs(numerator, [{(1,): 2}], 1))

    def test_nonnegative_numerator_needs_no_denominators(self):
        self.assertEqual(lemma1.absorbing_subset({(0,): 1, (1,): 3}, [{(1,): 1}], 1), ())

    def test_sign_parts_reconstruct_the_polynomial(self):
        from chernpp.polynomial import poly_sub as sub

        p = {(0,): 2, (1,): -5, (2,): 7}
        self.assertEqual(sub(lemma1.positive_part(p), lemma1.negative_part(p)), p)

    def test_generalised_pairing_is_strictly_weaker_than_lemma_1(self):
        # (1-u)/(1-v) = 1 + (v-u)/(1-v), so the ratio is nonnegative as soon as
        # (v-u)/(1-v) is -- which absorption can decide, without needing
        # v - u >= 0 outright.
        u, v = {(1,): 1, (2,): 2}, {(1,): 2}
        self.assertFalse(lemma1.dominates(v, u))  # Lemma 1 does not apply
        self.assertTrue(lemma1.pairs_generalised(u, v, 1))
        series = expand_rational(poly_sub({(0,): 1}, u), [v], 12)
        self.assertTrue(is_nonneg(series))  # and the ratio really is nonnegative

    def test_generalised_pairing_still_rejects_the_unsound_case(self):
        self.assertFalse(lemma1.pairs_generalised({(1,): 3}, {(1,): 1}, 1))

    def test_generalised_pairing_adds_nothing_on_the_a6_tail(self):
        # Strictly stronger in general, yet on this problem it admits exactly
        # the same 44 pairs: the positive part of v - u sits in higher degree
        # than the negative part, so it cannot cover it.
        factors, _, denominators, varnames = tail_target_factored(6)
        nvars = len(varnames)
        by_lemma = {
            (i, j)
            for i, u in enumerate(factors)
            for j, v in enumerate(denominators)
            if lemma1.dominates(v, u)
        }
        generalised = {
            (i, j)
            for i, u in enumerate(factors)
            for j, v in enumerate(denominators)
            if lemma1.pairs_generalised(u, v, nvars)
        }
        self.assertTrue(by_lemma <= generalised)
        self.assertEqual(by_lemma, generalised)

    def test_unrolling_preserves_the_series(self):
        # N/(1-v) = N(1 + v + ... + v^{m-1})/(1 - v^m), an identity.
        numerator, v = {(0,): 1, (1,): -1, (3,): 2}, {(1,): 1, (2,): 1}
        base = expand_rational(numerator, [v], 10)
        for times in (1, 2, 3):
            with self.subTest(times=times):
                rolled, power = lemma1.unroll(numerator, v, times, 1)
                self.assertEqual(expand_rational(rolled, [power], 10), base)

    def test_unrolling_rejects_a_nonpositive_count(self):
        with self.assertRaises(ValueError):
            lemma1.unroll({(0,): 1}, {(1,): 1}, 0, 1)

    def test_scaling_a_denominator_would_be_unsound(self):
        # Guards the note in lemma1.py.  Absorbing against 3v rather than v
        # would "prove" (1 - 3x)/(1 - x) >= 0, which is false: the series is
        # 1 - 2x - 2x^2 - ...  Any future scaled variant must fail this case.
        numerator, v = {(0,): 1, (1,): -3}, {(1,): 1}
        self.assertFalse(lemma1.absorbs(numerator, [v], 1))
        series = expand_rational(numerator, [v], 5)
        self.assertFalse(is_nonneg(series))
        self.assertEqual(series[(1,)], -2)

    def test_a6_tail_is_not_closed_by_absorption(self):
        # The multiplicative technique that proves d = 5 does not extend: no
        # subset of the remaining denominators absorbs the A_6 remainder.
        factors, residual, denominators, varnames = tail_target_factored(6)
        backoff = lemma1.search_with_backoff(
            factors,
            residual,
            denominators,
            len(varnames),
            varnames,
            max_returned=2,
            probe_degree=11,
        )
        self.assertIsNotNone(backoff)
        self.assertIsNone(
            lemma1.absorbing_subset(
                backoff.numerator, backoff.denominators, len(varnames), max_size=3
            )
        )


class TestTables(unittest.TestCase):
    def test_table_aligns_and_includes_a_rule(self):
        from chernpp.tables import table

        text = table([["a", 1], ["bb", 22]], headers=["x", "n"])
        lines = text.splitlines()
        self.assertEqual(len(lines), 4)
        self.assertIn("-", lines[1])

    def test_reports_mention_every_order(self):
        from chernpp.tables import algebra_report, series_report

        for report in (algebra_report(), series_report(max_deg=8)):
            for order in (4, 5, 6):
                self.assertIn(f"A_{order}", report)


if __name__ == "__main__":
    unittest.main()
