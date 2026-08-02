"""
Tier 6: the monomial-ideal combinatorics the multidegree is read off.

These two functions replaced a general-purpose primary decomposition, so they
carry the weight of a step that used to be Singular's problem.  A wrong answer
here does not look like a crash; it looks like a different multidegree, which is
to say like mathematics.  They are deliberately free of Sage so this tier runs
with everything else.

Also the :class:`~multidegree.Multidegree` contract, which is the boundary check
standing between a mis-assembled multidegree and a published artifact.
"""

import unittest
from itertools import combinations

from multidegree import Multidegree
from multidegree.monomial import (
    minimal_transversals,
    restrict_generators,
    standard_monomial_count,
)


def brute_force_transversals(supports, nvars):
    """Every minimal hitting set, by exhaustive search. Only usable for toy sizes."""
    hitting = []
    for size in range(nvars + 1):
        for candidate in combinations(range(nvars), size):
            chosen = frozenset(candidate)
            if all(chosen & s for s in supports):
                if not any(kept <= chosen for kept in hitting):
                    hitting.append(chosen)
    return sorted(hitting, key=lambda s: (len(s), sorted(s)))


class TestMinimalTransversals(unittest.TestCase):
    def test_single_generator_gives_one_prime_per_variable(self):
        # (x0 x1 x2) has three minimal primes, each a single variable.
        result = minimal_transversals([frozenset({0, 1, 2})], bound=3)
        self.assertEqual(result, [frozenset({0}), frozenset({1}), frozenset({2})])

    def test_disjoint_supports_force_one_index_each(self):
        supports = [frozenset({0, 1}), frozenset({2, 3})]
        result = minimal_transversals(supports, bound=2)
        self.assertEqual(len(result), 4)
        self.assertTrue(all(len(s) == 2 for s in result))

    def test_nested_support_is_absorbed(self):
        # Hitting {0} already hits {0,1}, so there is a single minimal prime.
        result = minimal_transversals([frozenset({0}), frozenset({0, 1})], bound=2)
        self.assertEqual(result, [frozenset({0})])

    def test_agrees_with_brute_force_on_random_hypergraphs(self):
        import random

        rng = random.Random(20260802)
        for trial in range(60):
            nvars = rng.randint(2, 7)
            supports = [
                frozenset(rng.sample(range(nvars), rng.randint(1, nvars))) for _ in range(rng.randint(1, 6))
            ]
            expected = brute_force_transversals(supports, nvars)
            bound = nvars
            with self.subTest(trial=trial, supports=supports):
                self.assertEqual(minimal_transversals(supports, bound), expected)

    def test_bound_keeps_everything_at_or_below_it(self):
        # Pruning must not drop a transversal of size <= bound; that is what
        # makes a codimension check against the bound meaningful in both
        # directions.
        import random

        rng = random.Random(4711)
        for trial in range(40):
            nvars = rng.randint(2, 6)
            supports = [
                frozenset(rng.sample(range(nvars), rng.randint(1, nvars))) for _ in range(rng.randint(1, 5))
            ]
            full = brute_force_transversals(supports, nvars)
            for bound in range(nvars + 1):
                expected = [s for s in full if len(s) <= bound]
                with self.subTest(trial=trial, bound=bound):
                    self.assertEqual(minimal_transversals(supports, bound), expected)

    def test_result_is_an_antichain(self):
        supports = [frozenset({0, 1, 2}), frozenset({1, 2, 3}), frozenset({0, 3})]
        result = minimal_transversals(supports, bound=4)
        for a in result:
            for b in result:
                if a is not b:
                    self.assertFalse(a < b, f"{a} is contained in {b}")

    def test_empty_support_means_no_minimal_primes(self):
        # An empty support is a unit generator: the ideal is everything.
        self.assertEqual(minimal_transversals([frozenset({0}), frozenset()], bound=3), [])

    def test_no_supports_gives_the_zero_ideal(self):
        self.assertEqual(minimal_transversals([], bound=3), [frozenset()])

    def test_negative_bound_is_refused(self):
        with self.assertRaises(ValueError):
            minimal_transversals([frozenset({0})], bound=-1)


class TestRestrictGenerators(unittest.TestCase):
    def test_variables_outside_the_prime_are_set_to_one(self):
        # x0^2 x2 restricted to {0, 1} is x0^2.
        self.assertEqual(restrict_generators([(2, 0, 1)], [0, 1]), ((2, 0),))

    def test_a_generator_supported_away_from_the_prime_is_refused(self):
        # x2 restricts to 1, so the localised ideal is the unit ideal and the
        # index set is not a transversal. Dropping the generator would return a
        # positive length for a component of length zero.
        with self.assertRaises(ValueError) as caught:
            restrict_generators([(0, 0, 1), (1, 0, 0)], [0, 1])
        self.assertIn("transversal", str(caught.exception))

    def test_divisible_generators_are_removed(self):
        # x0^2 is divisible by x0, so only x0 survives.
        self.assertEqual(restrict_generators([(1, 0), (2, 0)], [0, 1]), ((1, 0),))

    def test_result_is_canonical_and_hashable(self):
        # It is used as a cache key, so order must not depend on input order.
        first = restrict_generators([(1, 0), (0, 1)], [0, 1])
        second = restrict_generators([(0, 1), (1, 0)], [0, 1])
        self.assertEqual(first, second)
        self.assertEqual(len({first, second}), 1)

    def test_index_order_is_respected(self):
        self.assertEqual(restrict_generators([(1, 2, 0)], [1, 0]), ((2, 1),))


class TestStandardMonomialCount(unittest.TestCase):
    """
    The component multiplicities, which used to be Singular's job.

    An undercount here does not crash; it produces a different multidegree. So
    this is checked against an independent enumeration of the quotient basis,
    on every artinian monomial ideal in a small random family.
    """

    def test_powers_of_the_variables(self):
        # k[x,y]/(x^a, y^b) has basis the monomials below (a, b).
        for a, b in ((1, 1), (2, 1), (2, 3), (4, 4)):
            with self.subTest(a=a, b=b):
                self.assertEqual(standard_monomial_count([(a, 0), (0, b)], 2), a * b)

    def test_a_mixed_generator_cuts_the_box(self):
        # k[x,y]/(x^2, y^2, xy) has basis {1, x, y}: the box holds 4, xy removes 1.
        self.assertEqual(standard_monomial_count([(2, 0), (0, 2), (1, 1)], 2), 3)

    def test_the_maximal_ideal_leaves_only_the_constants(self):
        self.assertEqual(standard_monomial_count([(1, 0), (0, 1)], 2), 1)

    def test_no_variables(self):
        self.assertEqual(standard_monomial_count([], 0), 1)
        self.assertEqual(standard_monomial_count([()], 0), 0)

    def test_non_artinian_is_refused(self):
        # Nothing bounds y, so the quotient is infinite-dimensional. Reporting a
        # finite number here would be a silently wrong multiplicity.
        with self.assertRaises(ValueError) as caught:
            standard_monomial_count([(1, 0)], 2)
        self.assertIn("artinian", str(caught.exception))

    def test_matches_independent_enumeration_on_random_ideals(self):
        import random
        from itertools import product

        def brute(generators, nvars, limit=6):
            # Enumerate a box strictly larger than any generator and count the
            # monomials outside the ideal, without using the artinian bound.
            span = [limit] * nvars
            return sum(
                1
                for point in product(*(range(s) for s in span))
                if not any(all(a <= b for a, b in zip(g, point)) for g in generators)
            )

        rng = random.Random(31337)
        checked = 0
        for _ in range(200):
            nvars = rng.randint(1, 3)
            bounds = [rng.randint(1, 3) for _ in range(nvars)]
            generators = [tuple(b if i == j else 0 for i in range(nvars)) for j, b in enumerate(bounds)]
            for _ in range(rng.randint(0, 3)):
                generators.append(tuple(rng.randint(0, b) for b in bounds))
            generators = [g for g in generators if any(g)]
            with self.subTest(generators=tuple(generators)):
                self.assertEqual(standard_monomial_count(generators, nvars), brute(generators, nvars))
            checked += 1
        self.assertEqual(checked, 200)

    def test_an_oversized_box_is_refused_rather_than_enumerated(self):
        from multidegree.monomial import MAX_BOX

        huge = [tuple(10**3 if i == j else 0 for i in range(4)) for j in range(4)]
        self.assertGreater(10**12, MAX_BOX)
        with self.assertRaises(ValueError) as caught:
            standard_monomial_count(huge, 4)
        self.assertIn("box", str(caught.exception))


class StubPolynomial:
    """Minimal stand-in for a Sage polynomial: it only needs a degree."""

    def __init__(self, degree):
        self._degree = degree

    def degree(self):
        return self._degree


class TestMultidegreeContract(unittest.TestCase):
    def test_declared_codimension_must_match_the_polynomial(self):
        # A multidegree of the wrong degree is a bug that must surface at the
        # boundary rather than silently corrupt an artifact.
        with self.assertRaises(RuntimeError) as caught:
            Multidegree(polynomial=StubPolynomial(5), ring=None, family="morin-a", order=6, codim=7)
        self.assertIn("codimension 7", str(caught.exception))

    def test_consistent_result_is_accepted(self):
        result = Multidegree(polynomial=StubPolynomial(13), ring=None, family="morin-a", order=7, codim=13)
        self.assertEqual((result.order, result.codim, result.family), (7, 13, "morin-a"))

    def test_it_is_frozen(self):
        result = Multidegree(polynomial=StubPolynomial(1), ring=None, family="morin-a", order=4, codim=1)
        with self.assertRaises(Exception):
            result.codim = 2


if __name__ == "__main__":
    unittest.main()
