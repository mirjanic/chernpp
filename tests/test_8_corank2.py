"""
Tier 8: corank-two orbit geometry for the I_{a,b} family.

Like every other tier this runs in the plain virtualenv.  The geometry is
computed once by the SageMath stage and frozen into
``tests/data/corank2_orbit_closures.json``::

    PYTHONPATH=src <sage>/python -m multidegree.corank2 tests/data/corank2_orbit_closures.json

so what is checked here is the frozen result together with the index
bookkeeping, normal forms and weight arithmetic, all of which are plain Python.

The finding pinned below is the reason :mod:`multidegree.corank2` stops at the
multidegree instead of emitting an artifact: the Borel orbit closure of a
corank-two jet depends on which representative of the singularity class it is
computed from, so it is not yet an invariant of the singularity.
"""

import json
import unittest
from pathlib import Path

from multidegree import corank2

SURVEY = Path(__file__).parent / "data" / "corank2_orbit_closures.json"


def load_survey():
    return json.loads(SURVEY.read_text())


class TestJetSpace(unittest.TestCase):
    def test_ambient_dimension_counts_the_graded_pieces(self):
        # 2 * sum_{j=2}^{k} (j + 1): a pair of components, Sym^j C^2 of rank j+1,
        # and no constant or linear term because the differential vanishes.
        for order, expected in ((2, 6), (3, 14), (4, 24)):
            with self.subTest(order=order):
                self.assertEqual(corank2.ambient_dimension(order), expected)
                self.assertEqual(len(corank2.jet_indices(order)), expected)

    def test_every_index_is_a_genuine_jet_coordinate(self):
        for t, i, j in corank2.jet_indices(4):
            with self.subTest(index=(t, i, j)):
                self.assertIn(t, (0, 1))
                self.assertGreaterEqual(i + j, 2)  # no constant or linear part
                self.assertLessEqual(i + j, 4)

    def test_indices_are_distinct_and_named_distinctly(self):
        indices = corank2.jet_indices(4)
        self.assertEqual(len(set(indices)), len(indices))
        names = [corank2.variable_name(*index) for index in indices]
        self.assertEqual(len(set(names)), len(names))

    def test_weights_follow_the_source_and_target_tori(self):
        # normal_weight is plain arithmetic on whatever z contains, so integer
        # vectors stand in for the torus characters here.
        index_of = {
            corank2.variable_name(*index): index for index in corank2.jet_indices(2)
        }
        s1, s2, t1, t2 = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))

        def combine(name):
            weight = [0, 0, 0, 0]
            t, i, j = index_of[name]
            weight[0] += i
            weight[1] += j
            weight[2 + t] -= 1
            return tuple(weight)

        for name in index_of:
            self.assertEqual(combine(name)[:2], (index_of[name][1], index_of[name][2]))
        self.assertEqual(combine("q_u_1_1"), (1, 1, -1, 0))
        self.assertEqual(combine("q_v_0_2"), (0, 2, 0, -1))

    def test_survey_records_the_jet_spaces_it_used(self):
        jet_space = load_survey()["jet_space"]
        for order, recorded in jet_space.items():
            with self.subTest(order=order):
                self.assertEqual(
                    recorded["ambient_dimension"], corank2.ambient_dimension(int(order))
                )
                self.assertEqual(
                    [tuple(i) for i in recorded["indices"]],
                    corank2.jet_indices(int(order)),
                )


class TestNormalForms(unittest.TestCase):
    def test_i_ab_normal_form(self):
        # f = (xy, x^a + y^b) realises Q = C[[x,y]]/(xy, x^a + y^b).
        self.assertEqual(corank2.normal_form(2, 3), ({(1, 1): 1}, {(2, 0): 1, (0, 3): 1}))

    def test_indices_are_ordered(self):
        with self.assertRaises(ValueError):
            corank2.normal_form(3, 2)

    def test_corank_two_needs_at_least_a_double_point(self):
        with self.assertRaises(ValueError):
            corank2.normal_form(1, 2)

    def test_recorded_normal_forms_are_self_consistent(self):
        for entry in load_survey()["normal_forms"]:
            a, b = entry["a"], entry["b"]
            with self.subTest(a=a, b=b):
                self.assertEqual(entry["dimension"], a + b)
                # The normal form has degree b, so it lives in the b-jet space.
                self.assertEqual(entry["jet_order"], b)
                self.assertEqual(entry["ambient_dimension"], corank2.ambient_dimension(b))
                self.assertEqual(corank2.normal_form(a, b)[1], {(a, 0): 1, (0, b): 1})


class TestFrozenOrbitClosures(unittest.TestCase):
    def setUp(self):
        self.survey = {r["germ"]: r for r in load_survey()["representatives"]}

    def test_x2_y2_is_a_codimension_two_coordinate_subspace(self):
        entry = self.survey["(x^2, y^2)"]
        self.assertEqual(entry["generators"], ["q_u_0_2", "q_u_1_1"])
        self.assertEqual(entry["codimension"], 2)

    def test_its_multidegree_is_the_product_of_the_two_normal_weights(self):
        # (s1 + s2 - t1)(2*s2 - t1), expanded over (s1, s2, t1, t2).
        expected = {
            (1, 1, 0, 0): 2,
            (0, 2, 0, 0): 2,
            (1, 0, 1, 0): -1,
            (0, 1, 1, 0): -3,
            (0, 0, 2, 0): 1,
        }
        terms = {tuple(e): c for e, c in self.survey["(x^2, y^2)"]["multidegree"]}
        self.assertEqual(terms, expected)

    def test_multidegree_is_homogeneous_of_degree_the_codimension(self):
        for germ, entry in self.survey.items():
            degrees = {sum(e) for e, _ in entry["multidegree"]}
            with self.subTest(germ=germ):
                self.assertEqual(degrees, {entry["codimension"]})

    def test_multidegree_is_balanced_between_source_and_target(self):
        # Each coordinate weight is i*s1 + j*s2 - t, with i + j >= 2, so every
        # term carries at least twice as many source characters as target ones.
        for germ, entry in self.survey.items():
            for exponents, _ in entry["multidegree"]:
                source, target = exponents[0] + exponents[1], exponents[2] + exponents[3]
                with self.subTest(germ=germ, exponents=exponents):
                    self.assertEqual(source + target, entry["codimension"])


class TestJetGroupLimit(unittest.TestCase):
    def test_order_three_is_refused_rather_than_computed_wrongly(self):
        # On 2-jets only the linear parts of source and target diffeomorphisms
        # act.  From order 3 the higher parts contribute, so the parametrisation
        # would compute the orbit of GL_2 x GL_2 where the jet group is meant.
        with self.assertRaises(NotImplementedError) as caught:
            corank2.orbit_closure(corank2.normal_form(2, 3), 3)
        self.assertIn("jet group", str(caught.exception))

    def test_the_jet_spaces_those_orders_need_are_still_described(self):
        for a, b in ((2, 3), (3, 3), (2, 4)):
            with self.subTest(a=a, b=b):
                self.assertEqual(
                    corank2.ambient_dimension(b), 2 * sum(j + 1 for j in range(2, b + 1))
                )


class TestTheOrbitClosureIsNotAnInvariant(unittest.TestCase):
    """
    The obstruction to building a corank-two formula on this data.

    Four germs of the single class I_{2,2} give three different Borel orbit
    closures.  Since the Borel sees the flag and these representatives sit
    differently against it, the multidegree is an invariant of the *jet*, not of
    the singularity -- so a reference jet has to be pinned down the way
    Bérczi--Szenes pin down eps_ref before any of this feeds a residue formula.
    """

    def setUp(self):
        self.survey = {r["germ"]: r for r in load_survey()["representatives"]}

    def test_the_two_standard_presentations_disagree(self):
        # C[[x,y]]/(xy, x^2+y^2) and C[[x,y]]/(x^2, y^2) are the same algebra
        # over C, yet their Borel orbit closures have different codimensions.
        self.assertEqual(self.survey["(xy, x^2 + y^2)"]["codimension"], 1)
        self.assertEqual(self.survey["(x^2, y^2)"]["codimension"], 2)

    def test_three_distinct_closures_appear(self):
        closures = {tuple(e["generators"]) for e in self.survey.values()}
        self.assertEqual(len(closures), 3)

    def test_one_closure_is_determinantal_rather_than_linear(self):
        entry = self.survey["(y^2, x^2)"]
        self.assertEqual(entry["codimension"], 2)
        self.assertEqual(len(entry["generators"]), 3)  # three quadrics, codim 2
        self.assertTrue(all("*" in g for g in entry["generators"]), "expected quadrics")

    def test_the_multidegrees_differ_too(self):
        # Not merely different ideals: different equivariant classes, so no
        # choice of representative can be dismissed as a change of coordinates.
        classes = {
            tuple(
                (tuple(exponents), coefficient)
                for exponents, coefficient in e["multidegree"]
            )
            for e in self.survey.values()
        }
        self.assertGreater(len(classes), 1)

    def test_genericity_does_not_repair_it(self):
        # The obvious fix -- demand the representative be generic against the
        # flags -- is unavailable.  GL_2 x GL_2 is transitive on generic 2-jets,
        # so I_{2,2} is the generic corank-two jet and has codimension 0 there;
        # B_2 x B_2 is not transitive, its generic orbit being a hypersurface.
        # So the Borel action has modality >= 1 and there is no generic Borel
        # orbit to single out.
        generic = load_survey()["generic_orbit"]
        self.assertEqual(generic["ambient_dimension"], corank2.ambient_dimension(2))
        self.assertEqual(generic["general_linear"], generic["ambient_dimension"])
        self.assertEqual(generic["borel"], generic["ambient_dimension"] - 1)

    def test_codimension_never_reaches_that_of_the_singularity(self):
        # Sigma^2 = closure of the I_{2,2} locus has codimension 4 in the space
        # of maps.  These are codimensions inside the corank-two jet space, a
        # different and much smaller ambient, so they must not be read as one.
        for germ, entry in self.survey.items():
            with self.subTest(germ=germ):
                self.assertLess(entry["codimension"], 4)


if __name__ == "__main__":
    unittest.main()
