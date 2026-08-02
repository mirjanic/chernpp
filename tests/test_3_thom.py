"""
Tier 3: the Thom polynomials, checked against independently known values.

Two external checks, and they are the strongest constraints on the whole
pipeline. At relative dimension zero the classical polynomials are hard-coded
below. At relative dimension one we compare against Rimányi's published tables,
loaded from ``tests/data/published_thom_polynomials.json``; those are computed
by the restriction-equation method, mathematics independent of the
Bérczi--Szenes residue formula implemented here, so agreement constrains Q_d
externally rather than self-consistently.
"""

import json
import math
import unittest
from pathlib import Path

from chernpp.chern import chern_coefficients, format_monomial, laurent_grid, thom_polynomial

PUBLISHED = Path(__file__).parent / "data" / "published_thom_polynomials.json"

#: Orders with a mined artifact.  A_7 was added once the ordered-saturation
#: strategy brought d = 7 into reach.
ORDERS = (1, 2, 3, 4, 5, 6, 7)


class TestClassicalThomPolynomials(unittest.TestCase):
    def test_top_chern_class_coefficient_is_factorial(self):
        # The coefficient of c_d is (d-1)!.
        for order in ORDERS:
            with self.subTest(order=order):
                self.assertIn(
                    f"{math.factorial(order - 1)}*c_{order}",
                    thom_polynomial(order, l_max=0),
                )


class TestChernMultisetStructure(unittest.TestCase):
    def test_multisets_sum_to_zero(self):
        # alpha ranges over orderings of a zero-sum multiset, by construction.
        for order in ORDERS:
            _, coefficients = chern_coefficients(order, l_max=0)
            for multiset in coefficients:
                with self.subTest(order=order, multiset=multiset):
                    self.assertEqual(sum(multiset), 0)

    def test_positive_part_is_bounded_by_the_order(self):
        for order in ORDERS:
            _, coefficients = chern_coefficients(order, l_max=0)
            for multiset in coefficients:
                with self.subTest(order=order, multiset=multiset):
                    self.assertLessEqual(sum(a for a in multiset if a > 0), order)

    def test_multisets_have_d_entries(self):
        for order in ORDERS:
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


def load_published():
    """The reference tables, keyed by ``(order, relative dimension)``."""
    document = json.loads(PUBLISHED.read_text())
    return {
        (table["order"], table["relative_dimension"]): {
            tuple(term["chern_indices"]): term["coefficient"] for term in table["terms"]
        }
        for table in document["tables"]
    }


def computed_table(dim, relative_dimension):
    """Our Thom polynomial as ``{sorted Chern index tuple: coefficient}``."""
    _, chern = chern_coefficients(dim=dim, l_max=relative_dimension)
    result = {}
    for multiset, coefficient in chern.items():
        indices = [relative_dimension + 1 + alpha for alpha in multiset]
        if min(indices) < 0:
            continue  # c_j = 0 for j < 0
        key = tuple(sorted(i for i in indices if i > 0))  # c_0 = 1
        result[key] = result.get(key, 0) + coefficient
    return {k: v for k, v in result.items() if v != 0}


class TestPublishedTables(unittest.TestCase):
    """Agreement with Rimányi's tables at relative dimension one."""

    def setUp(self):
        self.published = load_published()

    def test_reference_file_is_well_formed(self):
        self.assertEqual(
            {(d, l) for l in (0, 1) for d in ORDERS},
            set(self.published),
            "unexpected set of reference tables",
        )
        for key, table in self.published.items():
            with self.subTest(table=key):
                self.assertTrue(all(isinstance(v, int) for v in table.values()))
                self.assertTrue(all(tuple(sorted(m)) == m for m in table))
                # Every monomial has the codimension of the singularity.
                order, relative = key
                for monomial in table:
                    self.assertEqual(sum(monomial), order * (relative + 1))

    def test_reference_tables_have_the_expected_size(self):
        for key, expected in (
            ((1, 0), 1),
            ((2, 0), 2),
            ((3, 0), 3),
            ((4, 0), 5),
            ((5, 0), 7),
            ((6, 0), 11),
            ((7, 0), 15),
            ((1, 1), 1),
            ((2, 1), 3),
            ((3, 1), 7),
            ((4, 1), 15),
            ((5, 1), 30),
            ((6, 1), 58),
            ((7, 1), 105),
        ):
            with self.subTest(table=key):
                self.assertEqual(len(self.published[key]), expected)

    def test_every_coefficient_agrees(self):
        for (order, relative), table in sorted(self.published.items()):
            computed = computed_table(order, relative)
            with self.subTest(order=order, relative_dimension=relative):
                self.assertEqual(set(table), set(computed), "the two tables list different monomials")
                for monomial in sorted(table):
                    self.assertEqual(computed[monomial], table[monomial], f"A_{order}, {monomial}")

    def test_top_chern_coefficient(self):
        for (order, relative), table in sorted(self.published.items()):
            top = (order * (relative + 1),)
            with self.subTest(order=order):
                self.assertEqual(computed_table(order, relative)[top], table[top])


class TestExactArithmeticViaCRT(unittest.TestCase):
    """Residue arithmetic gets past the int64 ceiling the guard enforces."""

    def test_agrees_with_the_int64_path_where_that_is_valid(self):
        from chernpp.crt import chern_coefficients_exact

        for dim, l_max in ((4, 2), (5, 2)):
            _, fast = chern_coefficients(dim=dim, l_max=l_max)
            with self.subTest(dim=dim, l_max=l_max):
                self.assertEqual(chern_coefficients_exact(dim, l_max), fast)

    def test_grouping_is_independent_of_values(self):
        # The residues of one coefficient must line up across primes, so the
        # grouping has to come from the grid geometry rather than from which
        # cells happen to be nonzero.
        from chernpp.crt import grouping

        multisets, inverse, keep = grouping(3, 1)
        self.assertEqual(int(keep.sum()), inverse.size)
        self.assertTrue(all(sum(int(a) for a in m) == 0 for m in multisets))

    def test_reconstruction_is_verified_against_a_spare_prime(self):
        # Too small a prime pool must be reported, never silently wrapped.
        from chernpp.crt import chern_coefficients_exact

        with self.assertRaises(OverflowError):
            chern_coefficients_exact(5, 3, primes=(101, 103))


if __name__ == "__main__":
    unittest.main()
