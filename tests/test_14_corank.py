"""
Tier 14: the corank filtration C_1 < C_2 < ... of Thom polynomials (:mod:`chernpp.corank`).

Every membership answer carries an exact certificate; these tests pin the
symmetric-function arithmetic, the certificates on small Porteous and corank-2
classes, and the refutation of the naive statement rho = corank.
"""

import unittest
from fractions import Fraction

from chernpp import corank


class TestSchurArithmetic(unittest.TestCase):
    def test_jacobi_trudi_round_trip(self):
        for nu in ((2, 2), (3, 2, 1), (4, 1, 1), (2, 2, 2)):
            with self.subTest(nu=nu):
                self.assertEqual(corank.chern_to_schur(corank.schur_to_chern({nu: 1})), {nu: 1})

    def test_pieri_and_littlewood_richardson(self):
        self.assertEqual(corank.schur_product([(3, 3), (2,)]), {(5, 3): 1, (4, 3, 1): 1, (3, 3, 2): 1})
        self.assertEqual(corank.schur_product([(1,), (1,)]), {(2,): 1, (1, 1): 1})

    def test_porteous_class_of_sigma2(self):
        # c_2^2 - c_1 c_3 = s_22: the Thom polynomial of Sigma^2 at l = 0.
        self.assertEqual(corank.chern_to_schur({(2, 2): 1, (3, 1): -1}), {(2, 2): 1})


class TestMembership(unittest.TestCase):
    def test_two_row_rectangle_is_in_c2_not_c1(self):
        self.assertTrue(corank.membership({(2, 2): 1}, 2).member)
        self.assertFalse(corank.membership({(2, 2): 1}, 1).member)

    def test_three_rows_are_out_of_c2(self):
        for target in ({(1, 1, 1): 1}, {(2, 2, 1): 1}):
            with self.subTest(target=target):
                m = corank.membership(target, 2)
                self.assertFalse(m.member)
                self.assertLess(sum(m.farkas.get(k, 0) * v for k, v in target.items()), 0)

    def test_decompositions_are_exact(self):
        m = corank.membership({(3, 3): 2, (4, 2): 1, (5, 1): 1, (6,): 1}, 2)
        self.assertTrue(m.member)
        for w in m.decomposition.values():
            self.assertIsInstance(w, Fraction)
            self.assertGreaterEqual(w, 0)


class TestRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = {(t.singularity, t.relative_dimension): t for t in corank.load_registry()}

    def schur(self, name, l):
        return corank.chern_to_schur(self.tables[(name, l)].chern)

    def test_chern_sum_is_the_one_row_coefficient(self):
        for (name, l), t in self.tables.items():
            if t.codimension == 0:
                continue  # A_0: Tp = 1, the empty partition
            with self.subTest(name=name, l=l):
                s = corank.chern_to_schur(t.chern)
                self.assertEqual(sum(t.chern.values()), s.get((t.codimension,), 0))
                if name.startswith("A_") and name != "A_0":
                    d = int(name[2:])
                    from math import factorial

                    self.assertEqual(sum(t.chern.values()), factorial(d) ** (l + 1))
                elif corank.algebra_corank(t) and corank.algebra_corank(t) >= 2:
                    self.assertEqual(sum(t.chern.values()), 0)

    def test_every_table_is_schur_positive(self):
        for (name, l), t in self.tables.items():
            with self.subTest(name=name, l=l):
                self.assertTrue(all(v > 0 for v in corank.chern_to_schur(t.chern).values()))

    def test_balanced_i_family_is_in_c2(self):
        for name in ("I_2,2", "I_2,3", "I_3,3", "I_3,4"):
            with self.subTest(name=name):
                self.assertTrue(corank.membership(self.schur(name, 0), 2).member)

    def test_i24_refutes_rho_equals_corank(self):
        target = self.schur("I_2,4", 0)
        m = corank.membership(target, 2)
        self.assertFalse(m.member)
        self.assertEqual(target[(3, 3)] + target[(2, 2, 2)] - target[(3, 2, 1)], -3)
        self.assertTrue(corank.membership(target, 3).member)

    def test_iii23_certificate(self):
        # 8 s_53 + 4 s_431 + 2 s_332: every usable C_2 generator weighs s_332 at
        # least as much as s_431, and Tp has twice as much on s_431.
        target = self.schur("III_2,3", 1)
        self.assertEqual(target, {(5, 3): 8, (4, 3, 1): 4, (3, 3, 2): 2})
        self.assertFalse(corank.membership(target, 2).member)


if __name__ == "__main__":
    unittest.main()
