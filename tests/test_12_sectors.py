"""
Tier 12: closed forms on sectors of the Chern table (:mod:`chernpp.sectors`).

These pin two claims from the external reports that the report now relies on:
the plane-sector values (Kreweras) and stabilisation in ``d``.
"""

import unittest

from chernpp import boxes, sectors


class TestPlaneSector(unittest.TestCase):
    def test_kreweras_matches_every_plane_packet(self):
        for d in (4, 5, 6):
            with self.subTest(d=d):
                table = boxes.chern_table_exact(boxes.level_box(d, d))
                plane = [m for m in table if sectors.is_plane(m)]
                self.assertEqual(sectors.plane_sector_mismatches(table), [])
                # one packet per partition of d (the block sizes)
                self.assertEqual(len(plane), {4: 5, 5: 7, 6: 11}[d])

    def test_worked_d7_example(self):
        self.assertEqual(sectors.kreweras((1, 1, 1, 0, -1, -1, -1)), 35)

    def test_value_one_only_at_the_ends(self):
        self.assertEqual(sectors.kreweras((0,) * 5), 1)
        self.assertEqual(sectors.kreweras((1, 1, 1, 1, -4)), 1)

    def test_non_plane_is_refused(self):
        with self.assertRaises(ValueError):
            sectors.block_sizes((2, -1, -1))


class TestStabilisation(unittest.TestCase):
    def test_top_face_is_the_previous_series(self):
        for d in (5, 6, 7):
            with self.subTest(d=d):
                self.assertTrue(sectors.stabilisation_holds(d, [12, 10, 8, 6, 4][: d - 2]))


if __name__ == "__main__":
    unittest.main()
