"""
Tier 10: the box engine in :mod:`chernpp.boxes`.

The sweep replaces the fixed point of :mod:`chernpp.chern` by one pass per
denominator factor, so it is checked cell by cell against that fixed point and,
through the grouped sums, against the independent CRT path of
:mod:`chernpp.crt`.  The packet counts pin the box/packet correspondence that
every packet-level statement in the report relies on.
"""

import unittest

import numpy as np

from chernpp import boxes
from chernpp.chern import laurent_grid
from chernpp.crt import chern_coefficients_exact


class TestPackets(unittest.TestCase):
    def test_beta_max_dominates_every_ballot_ordering(self):
        for m in ((2, 1, 0, -1, -2), (1, 1, 1, 0, -1, -1, -1), (3, -1, -1, -1)):
            top = boxes.beta_max(m)
            for _, beta in boxes.ballot_words(m):
                self.assertTrue(all(b <= t for b, t in zip(beta, top)), (m, beta))

    def test_level_and_charge_boxes_hold_the_same_number_of_packets(self):
        # min M >= -L and max M <= L are exchanged by M -> -M, and so are the boxes.
        expected = [15, 105, 436]
        for L, n in enumerate(expected, start=1):
            with self.subTest(L=L):
                self.assertEqual(len(boxes.complete_packets(boxes.level_box(7, L))), n)
                self.assertEqual(len(boxes.complete_packets(boxes.charge_box(7, L))), n)

    def test_level_box_packets_are_exactly_min_at_least_minus_L(self):
        box = boxes.level_box(5, 2)
        found = set(boxes.complete_packets(box))
        for m in found:
            self.assertGreaterEqual(min(m), -2)
            self.assertEqual(sum(m), 0)
            self.assertEqual(tuple(sorted(m, reverse=True)), m)
        self.assertEqual(len(found), len(chern_coefficients_exact(5, 1, verify=False)))


class TestSweep(unittest.TestCase):
    def test_sweep_matches_the_fixed_point_cell_by_cell(self):
        for d, l in ((4, 2), (5, 2), (6, 1)):
            with self.subTest(d=d, l=l):
                grid = boxes.exact_box(boxes.level_box(d, l + 1))
                self.assertTrue(np.array_equal(grid, laurent_grid(d, l)))

    def test_grouped_sums_match_the_crt_path(self):
        d, l = 5, 4
        box = boxes.level_box(d, l + 1)
        table = boxes.chern_table(boxes.exact_box(box), box)
        reference = {tuple(sorted(k, reverse=True)): v for k, v in chern_coefficients_exact(d, l).items()}
        self.assertEqual(table, reference)

    def test_smoke_coefficient_at_d7(self):
        box = boxes.level_box(7, 1)
        stats = boxes.packet_stats(boxes.exact_box(box), (1, 1, 1, 0, -1, -1, -1))
        self.assertEqual(stats.chern, 35)
        self.assertEqual(stats.positive_mass - stats.negative_mass, 35)

    def test_incomplete_packet_is_refused(self):
        grid = boxes.exact_box(boxes.level_box(5, 1))
        with self.assertRaises(ValueError):
            boxes.packet_values(grid, (2, 1, 0, -1, -2))


if __name__ == "__main__":
    unittest.main()
