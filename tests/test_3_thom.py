"""
Tier 3: the classical Thom polynomials, and the l-free reduction.

These are the strongest external check on the whole pipeline: the coefficients
below are known independently, and reproducing all of them pins down Q_d.
"""

import math
import unittest

from chernpp.chern import chern_coefficients, format_monomial, laurent_grid, thom_polynomial

#: Tp_{A_d} at relative dimension l = 0, in the classical literature.
CLASSICAL = {
    4: ["1*c_1^4", "6*c_2 * c_1^2", "2*c_2^2", "9*c_3 * c_1", "6*c_4"],
    5: [
        "1*c_1^5",
        "10*c_2 * c_1^3",
        "10*c_2^2 * c_1",
        "25*c_3 * c_1^2",
        "12*c_3 * c_2",
        "38*c_4 * c_1",
        "24*c_5",
    ],
    6: [
        "1*c_1^6",
        "15*c_2 * c_1^4",
        "30*c_2^2 * c_1^2",
        "5*c_2^3",
        "55*c_3 * c_1^3",
        "79*c_3 * c_2 * c_1",
        "17*c_3^2",
        "141*c_4 * c_1^2",
        "55*c_4 * c_2",
        "202*c_5 * c_1",
        "120*c_6",
    ],
}


class TestClassicalThomPolynomials(unittest.TestCase):
    def test_every_classical_term_is_reproduced(self):
        for order, terms in CLASSICAL.items():
            polynomial = thom_polynomial(order, l_max=0)
            for term in terms:
                with self.subTest(order=order, term=term):
                    self.assertIn(term, polynomial)

    def test_no_extra_terms(self):
        for order, terms in CLASSICAL.items():
            with self.subTest(order=order):
                produced = {t.strip() for t in thom_polynomial(order, 0).split(" + ")}
                self.assertEqual(produced, set(terms))

    def test_top_chern_class_coefficient_is_factorial(self):
        # The coefficient of c_d is (d-1)!.
        for order in (4, 5, 6):
            with self.subTest(order=order):
                self.assertIn(
                    f"{math.factorial(order - 1)}*c_{order}",
                    thom_polynomial(order, l_max=0),
                )


class TestChernMultisetStructure(unittest.TestCase):
    def test_multisets_sum_to_zero(self):
        # alpha ranges over orderings of a zero-sum multiset, by construction.
        for order in (4, 5, 6):
            _, coefficients = chern_coefficients(order, l_max=0)
            for multiset in coefficients:
                with self.subTest(order=order, multiset=multiset):
                    self.assertEqual(sum(multiset), 0)

    def test_positive_part_is_bounded_by_the_order(self):
        for order in (4, 5, 6):
            _, coefficients = chern_coefficients(order, l_max=0)
            for multiset in coefficients:
                with self.subTest(order=order, multiset=multiset):
                    self.assertLessEqual(sum(a for a in multiset if a > 0), order)

    def test_multisets_have_d_entries(self):
        for order in (4, 5, 6):
            _, coefficients = chern_coefficients(order, l_max=0)
            self.assertTrue(all(len(m) == order for m in coefficients))

    def test_ghost_monomials_are_filtered(self):
        # A multiset with an entry below -(l+1) would need a negative-index
        # Chern class, i.e. c_j = 0, so it must not appear at all.
        for l_max in (0, 2):
            _, coefficients = chern_coefficients(6, l_max=l_max)
            for multiset in coefficients:
                with self.subTest(l_max=l_max, multiset=multiset):
                    self.assertGreaterEqual(min(multiset), -(l_max + 1))

    def test_format_monomial_conventions(self):
        # c_0 = 1, so at l_max = 0 an entry of -1 contributes nothing, while an
        # entry below -(l_max + 1) needs c_{<0} = 0 and kills the monomial.
        self.assertEqual(format_monomial((0, 0, 0, 0), l_max=-1), "1")
        self.assertEqual(format_monomial((-3, 1, 1, 1), l_max=0), "0")
        self.assertEqual(format_monomial((1, 1, 0, -1, -1), l_max=0), "c_2^2 * c_1")


class TestRelativeDimensionIndependence(unittest.TestCase):
    def test_coefficient_depends_on_l_only_through_the_multiset(self):
        # The l-free reduction: C(M) is the same integer at every relative
        # dimension admitting M.  The multiset {1,1,0,-1,-1} gives 10 for A_5.
        multiset = (-1, -1, 0, 1, 1)
        for l_max in range(5):
            _, coefficients = chern_coefficients(5, l_max=l_max)
            with self.subTest(l_max=l_max):
                self.assertEqual(coefficients.get(multiset), 10)

    def test_verification_is_nested_in_l(self):
        # Raising l only admits multisets with more negative entries; every
        # multiset seen at l must still be present with the same value at l+1.
        for l_max in range(3):
            _, low = chern_coefficients(5, l_max=l_max)
            _, high = chern_coefficients(5, l_max=l_max + 1)
            for multiset, value in low.items():
                with self.subTest(l_max=l_max, multiset=multiset):
                    self.assertEqual(high.get(multiset), value)


class TestOverflowGuard(unittest.TestCase):
    def test_guard_refuses_rather_than_wrapping(self):
        # Past the int64 range a wrapped accumulator reads as a large negative
        # number, i.e. as a spurious counterexample.  It must raise instead.
        with self.assertRaises(OverflowError) as caught:
            chern_coefficients(5, l_max=9)
        self.assertIn("int64", str(caught.exception))

    def test_guard_can_be_disabled_deliberately(self):
        # Without the probe the call returns (wrapped) values instead of raising.
        grid = laurent_grid(5, l_max=9, check_overflow=False)
        self.assertEqual(grid.ndim, 4)


if __name__ == "__main__":
    unittest.main()
