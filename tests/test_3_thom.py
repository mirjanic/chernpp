"""
Tier 3: the Thom polynomials, checked against independently known values.

This is the strongest constraint on the whole pipeline. Rimányi's registry
publishes Thom polynomials at relative dimensions 0 through 5, computed by the
restriction-equation method -- mathematics independent of the Bérczi--Szenes
residue formula implemented here -- so agreement constrains Q_d externally
rather than self-consistently. The tables are scraped by
``tools/scrape_thom_polynomials.py`` into
``tests/data/published_thom_polynomials.json``.

Every A_d table the registry publishes that we have an artifact for is checked:
35 tables, spanning d = 1..7 and l = 0..5. The corpus also carries the corank-two
and other families (I, III, B, C), which nothing here computes; those are checked
for internal consistency only, and are there for the work in
:mod:`multidegree.corank2`.
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
    """
    Every reference table, keyed by ``(singularity name, relative dimension)``.

    Keyed by name rather than by order because the corpus is not only the Morin
    family: ``order`` is meaningful for A_d and null for everything else, so it
    does not identify a table.
    """
    document = json.loads(PUBLISHED.read_text())
    return {(table["singularity"], table["relative_dimension"]): table for table in document["tables"]}


def terms_of(table):
    """A table's terms as ``{sorted Chern index tuple: coefficient}``."""
    return {tuple(term["chern_indices"]): term["coefficient"] for term in table["terms"]}


def morin_tables(published):
    """
    The A_d tables we can actually check: those with a mined artifact.

    A_0 is the regular germ, with no artifact and nothing to compute; A_8 and A_9
    appear at l = 0 but are past what the Sage stage reaches.
    """
    return {
        key: table
        for key, table in published.items()
        if table["order"] in ORDERS and table["singularity"].startswith("A_")
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


class TestReferenceCorpus(unittest.TestCase):
    """Internal consistency of every scraped table, including families we cannot compute."""

    def setUp(self):
        self.published = load_published()

    def test_the_corpus_spans_six_relative_dimensions(self):
        dimensions = {relative for _, relative in self.published}
        self.assertEqual(dimensions, {0, 1, 2, 3, 4, 5})
        self.assertGreater(len(self.published), 150)

    def test_every_monomial_has_the_stated_codimension(self):
        # The Thom polynomial is homogeneous of the singularity's codimension,
        # where c_i has weight i.  This is what catches a mis-parsed exponent.
        for key, table in self.published.items():
            for monomial, coefficient in terms_of(table).items():
                with self.subTest(table=key, monomial=monomial):
                    self.assertEqual(sum(monomial), table["codimension"])
                    self.assertIsInstance(coefficient, int)
                    self.assertNotEqual(coefficient, 0)

    def test_chern_indices_are_sorted_and_positive(self):
        for key, table in self.published.items():
            for monomial in terms_of(table):
                with self.subTest(table=key, monomial=monomial):
                    self.assertEqual(list(monomial), sorted(monomial))
                    self.assertTrue(all(index >= 1 for index in monomial))

    def test_no_monomial_appears_twice_in_a_table(self):
        for key, table in self.published.items():
            monomials = [tuple(term["chern_indices"]) for term in table["terms"]]
            with self.subTest(table=key):
                self.assertEqual(len(monomials), len(set(monomials)))

    def test_morin_codimension_follows_the_closed_form(self):
        # codim A_d = d(l+1).  A cross-check on the scraper's name parsing: a
        # table mislabelled A_5 that is really A_6 fails here.
        for (name, relative), table in self.published.items():
            if table["order"] is None:
                continue
            with self.subTest(table=(name, relative)):
                self.assertEqual(table["codimension"], table["order"] * (relative + 1))

    def test_rimanyi_positivity_holds_on_the_published_tables_we_cannot_compute(self):
        # A_8 and A_9 are published at relative dimension 0 and are past what the
        # Sage stage reaches -- Q_8 needs a Groebner basis in 50 variables that
        # does not finish.  The conjecture can still be read off Rimanyi's own
        # tables there, which is evidence for it at two orders beyond ours.
        for name in ("A_8", "A_9"):
            table = self.published[(name, 0)]
            with self.subTest(singularity=name):
                self.assertTrue(
                    all(c > 0 for c in terms_of(table).values()),
                    f"{name} has a negative Chern coefficient",
                )
        self.assertEqual(len(terms_of(self.published[("A_8", 0)])), 22)
        self.assertEqual(len(terms_of(self.published[("A_9", 0)])), 30)

    def test_every_morin_table_in_the_corpus_is_chern_positive(self):
        # Rimanyi's conjecture, read across the whole scraped corpus rather than
        # only where we can recompute: every A_d table at every relative
        # dimension published.  This is not a proof of anything, but a negative
        # coefficient here would be a counterexample, and there is none.
        checked = 0
        for (name, relative), table in self.published.items():
            if not name.startswith("A_"):
                continue
            with self.subTest(singularity=name, relative_dimension=relative):
                self.assertTrue(all(c > 0 for c in terms_of(table).values()))
            checked += 1
        self.assertGreater(checked, 40)

    def test_corank_two_families_are_present_for_later_work(self):
        # Nothing here computes these; they are the reference data for the
        # corank-two question in multidegree/corank2.py.
        names = {name for name, _ in self.published}
        self.assertIn("I_2,2", names)
        self.assertIn("III_2,2", names)
        # Tp(I_2,2) at l = 0 is the Giambelli-Thom-Porteous class of Sigma^2,
        # s_{2,2} = c_2^2 - c_1 c_3.  Note the negative Chern coefficient:
        # Rimanyi's conjecture is a corank-one statement and does not extend.
        self.assertEqual(terms_of(self.published[("I_2,2", 0)]), {(1, 3): -1, (2, 2): 1})


class TestPublishedTables(unittest.TestCase):
    """Agreement with Rimányi's tables, at every relative dimension they reach."""

    def setUp(self):
        self.published = load_published()
        self.morin = morin_tables(self.published)

    def test_we_check_every_morin_table_we_have_an_artifact_for(self):
        # l = 0 publishes A_0 through A_9 and l = 5 only A_0 through A_4, so the
        # reachable set is ragged; this pins which 35 tables are actually being
        # compared, so a silently narrowed check would fail here.
        by_dimension = {}
        for (_, relative), table in self.morin.items():
            by_dimension.setdefault(relative, set()).add(table["order"])
        self.assertEqual(
            by_dimension,
            {
                0: {1, 2, 3, 4, 5, 6, 7},
                1: {1, 2, 3, 4, 5, 6, 7},
                2: {1, 2, 3, 4, 5, 6},
                3: {1, 2, 3, 4, 5, 6},
                4: {1, 2, 3, 4, 5},
                5: {1, 2, 3, 4},
            },
        )
        self.assertEqual(len(self.morin), 35)

    def test_reference_tables_have_the_expected_size(self):
        for key, expected in (
            (("A_1", 0), 1),
            (("A_2", 0), 2),
            (("A_3", 0), 3),
            (("A_4", 0), 5),
            (("A_5", 0), 7),
            (("A_6", 0), 11),
            (("A_7", 0), 15),
            (("A_4", 1), 15),
            (("A_5", 1), 30),
            (("A_6", 1), 58),
            (("A_7", 1), 105),
        ):
            with self.subTest(table=key):
                self.assertEqual(len(self.published[key]["terms"]), expected)

    def test_every_coefficient_agrees(self):
        for (name, relative), table in sorted(self.morin.items()):
            published = terms_of(table)
            computed = computed_table(table["order"], relative)
            with self.subTest(singularity=name, relative_dimension=relative):
                self.assertEqual(set(published), set(computed), "the two tables list different monomials")
                for monomial in sorted(published):
                    self.assertEqual(
                        computed[monomial], published[monomial], f"{name}, l={relative}, {monomial}"
                    )

    def test_top_chern_coefficient(self):
        for (name, relative), table in sorted(self.morin.items()):
            top = (table["codimension"],)
            with self.subTest(singularity=name, relative_dimension=relative):
                self.assertEqual(computed_table(table["order"], relative)[top], terms_of(table)[top])


class TestExactArithmeticViaCRT(unittest.TestCase):
    """Residue arithmetic gets past the int64 ceiling the guard enforces."""

    def test_agrees_with_the_int64_path_where_that_is_valid(self):
        from chernpp.crt import chern_coefficients_exact

        for dim, l_max in ((4, 2), (5, 2)):
            _, fast = chern_coefficients(dim=dim, l_max=l_max)
            with self.subTest(dim=dim, l_max=l_max):
                self.assertEqual(chern_coefficients_exact(dim, l_max), fast)

    def test_the_multi_prime_path_is_actually_exercised(self):
        # Every other CRT test lands in the regime where one prime suffices, so
        # the reconstruction itself -- the reason this module exists -- was never
        # run. A_6 at l = 6 needs three primes and reaches 2.4e19, past the int64
        # ceiling the fast path refuses at, and every value is nonnegative:
        # Rimanyi's conjecture at a relative dimension the grid cannot reach.
        from chernpp.crt import chern_coefficients_exact

        exact = chern_coefficients_exact(6, 6)
        largest = max(abs(v) for v in exact.values())
        self.assertGreater(largest, 2**63, "this should be past what int64 holds")
        self.assertTrue(all(v >= 0 for v in exact.values()))
        with self.assertRaises(OverflowError):
            chern_coefficients(dim=6, l_max=6)

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
