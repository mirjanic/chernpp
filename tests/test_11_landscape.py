"""
Tier 11: the 0-Hecke positivity landscape (:mod:`chernpp.hecke`).

Pins the operator algebra (so that ``Pi_w`` is well defined) and the d = 5
landscape: ``P_5 = S_5 minus <s_3, s_4>``, whose minimal elements are
``s_1, s_2, s_2 s_3, s_2 s_3 s_4``.
"""

import unittest
from itertools import permutations

import numpy as np

from chernpp import boxes, hecke


class TestOperators(unittest.TestCase):
    def setUp(self):
        self.packet = hecke.Packet((2, 1, 0, -1, -2))
        rng = np.random.default_rng(0)
        self.f = rng.integers(-5, 6, size=len(self.packet.words))

    def test_zero_hecke_relations(self):
        p, f = self.packet, self.f
        for i in range(1, 5):
            self.assertTrue(np.array_equal(p.pi(i, p.pi(i, f)), p.pi(i, f)))
        for i in range(1, 4):
            a = p.pi(i, p.pi(i + 1, p.pi(i, f)))
            b = p.pi(i + 1, p.pi(i, p.pi(i + 1, f)))
            self.assertTrue(np.array_equal(a, b))
        self.assertTrue(np.array_equal(p.pi(1, p.pi(3, f)), p.pi(3, p.pi(1, f))))

    def test_longest_element_collects_the_packet_sum(self):
        p, g = self.packet, self.f
        for i in hecke.reduced_word(tuple(reversed(range(5)))):
            g = p.pi(i, g)
        top = p.index[p.multiset]
        self.assertEqual(g[top], self.f.sum())
        self.assertEqual(np.count_nonzero(np.delete(g, top)), 0)

    def test_reduced_words_have_the_right_length(self):
        for w in permutations(range(4)):
            self.assertEqual(len(hecke.reduced_word(w)), hecke.length(w))


class TestLandscape(unittest.TestCase):
    def test_d5(self):
        r = hecke.landscape(5, boxes.level_box(5, 4))
        self.assertEqual(r.positive_set_size, 120 - 6)
        self.assertEqual(sorted(map(tuple, r.minimal)), sorted([(1,), (2,), (2, 3), (2, 3, 4)]))

    def test_d4_is_already_positive(self):
        r = hecke.landscape(4, boxes.level_box(4, 4))
        self.assertEqual(r.negative_packets, 0)
        self.assertEqual(r.minimal, [[]])


if __name__ == "__main__":
    unittest.main()
