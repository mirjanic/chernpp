"""
Tier 8: corank-two orbit geometry for the I_{a,b} family.

Unlike the other tiers this one needs SageMath, so it skips in the plain
virtualenv and runs under Sage's interpreter::

    PYTHONPATH=src ~/miniforge3/envs/sage/bin/python -m unittest tests.test_8_corank2

The results pinned here are the reason :mod:`multidegree.corank2` stops at the
multidegree instead of emitting an artifact: the Borel orbit closure of a
corank-two jet depends on which representative of the singularity class it is
computed from, so it is not yet an invariant of the singularity.
"""

import unittest

try:
    from sage.all import QQ, PolynomialRing

    from multidegree import corank2

    HAVE_SAGE = True
except ImportError:  # pragma: no cover - depends on the interpreter
    HAVE_SAGE = False


@unittest.skipUnless(HAVE_SAGE, "needs SageMath")
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

    def test_weights_follow_the_source_and_target_tori(self):
        ring, index_of = corank2.coordinate_ring(QQ, 2)
        weights = PolynomialRing(QQ, ["s1", "s2", "t1", "t2"])
        s1, s2, t1, t2 = weights.gens()
        index_of = {str(v): idx for v, idx in index_of.items()}
        self.assertEqual(
            corank2.normal_weight("q_u_1_1", index_of, weights.gens()), s1 + s2 - t1
        )
        self.assertEqual(
            corank2.normal_weight("q_v_0_2", index_of, weights.gens()), 2 * s2 - t2
        )


@unittest.skipUnless(HAVE_SAGE, "needs SageMath")
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


@unittest.skipUnless(HAVE_SAGE, "needs SageMath")
class TestOrbitClosure(unittest.TestCase):
    def test_x2_y2_is_a_codimension_two_coordinate_subspace(self):
        ideal, ring = corank2.orbit_closure(({(2, 0): 1}, {(0, 2): 1}), 2)
        self.assertEqual(sorted(str(g) for g in ideal.gens()), ["q_u_0_2", "q_u_1_1"])
        self.assertEqual(ring.ngens() - ideal.dimension(), 2)

    def test_its_multidegree_is_the_product_of_the_two_normal_weights(self):
        weights = PolynomialRing(QQ, ["s1", "s2", "t1", "t2"])
        s1, s2, t1, _ = weights.gens()
        polynomial, codimension = corank2.multidegree(({(2, 0): 1}, {(0, 2): 1}), 2)
        self.assertEqual(codimension, 2)
        self.assertEqual(polynomial, (s1 + s2 - t1) * (2 * s2 - t1))

    def test_multidegree_is_homogeneous_of_degree_the_codimension(self):
        for jet in (({(2, 0): 1}, {(0, 2): 1}), ({(0, 2): 1}, {(2, 0): 1})):
            polynomial, codimension = corank2.multidegree(jet, 2)
            with self.subTest(jet=jet):
                self.assertEqual(polynomial.degree(), codimension)
                self.assertTrue(polynomial.is_homogeneous())


@unittest.skipUnless(HAVE_SAGE, "needs SageMath")
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
        self.survey = corank2.representative_survey()

    def test_the_two_standard_presentations_disagree(self):
        # C[[x,y]]/(xy, x^2+y^2) and C[[x,y]]/(x^2, y^2) are the same algebra
        # over C, yet their Borel orbit closures have different codimensions.
        self.assertEqual(self.survey["(xy, x^2 + y^2)"][1], 1)
        self.assertEqual(self.survey["(x^2, y^2)"][1], 2)

    def test_three_distinct_closures_appear(self):
        closures = {tuple(gens) for gens, _ in self.survey.values()}
        self.assertEqual(len(closures), 3)

    def test_one_closure_is_determinantal_rather_than_linear(self):
        generators, codimension = self.survey["(y^2, x^2)"]
        self.assertEqual(codimension, 2)
        self.assertEqual(len(generators), 3)  # three quadrics cutting codim 2
        self.assertTrue(all("*" in g for g in generators), "expected quadrics")

    def test_codimension_never_reaches_that_of_the_singularity(self):
        # Sigma^2 = closure of the I_{2,2} locus has codimension 4 in the space
        # of maps.  These are codimensions inside the corank-two jet space, a
        # different and much smaller ambient, so they must not be read as one.
        for label, (_, codimension) in self.survey.items():
            with self.subTest(label=label):
                self.assertLess(codimension, 4)


if __name__ == "__main__":
    unittest.main()
