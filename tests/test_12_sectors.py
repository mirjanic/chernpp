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


class TestBallot(unittest.TestCase):
    def test_ballot_count_matches_enumeration(self):
        for m in ((1, 1, 1, 0, -1, -1, -1), (2, 1, -1, -2), (3, -1, -1, -1), (2, 2, -4)):
            with self.subTest(m=m):
                self.assertEqual(sectors.ballot_count(m), sum(1 for _ in boxes.ballot_words(m)))

    def test_plane_ballot_count_is_kreweras(self):
        # the cycle lemma
        for d in (4, 5, 6, 7):
            for m in boxes.complete_packets(boxes.charge_box(d, 1)):
                self.assertEqual(sectors.ballot_count(m), sectors.kreweras(m))

    def test_one_monotone_factorisation_per_ballot_vector(self):
        # step 2 of the plane-sector proof, exhaustively for n <= 6
        from itertools import product

        for n in range(2, 7):
            found = sectors.monotone_factorisations(n)
            self.assertEqual(set(found.values()), {1})
            ballot = set()
            for tail in product(range(n), repeat=n - 1):
                if sum(tail) != n - 1:
                    continue
                alpha = tuple(1 - e for e in tail)
                if all(sum(alpha[: j + 1]) >= 0 for j in range(len(alpha) - 1)):
                    ballot.add((0,) + tail)
            self.assertEqual(set(found), ballot)

    def test_ballot_conjecture_on_small_boxes(self):
        for d, L in ((2, 6), (3, 8), (4, 6), (5, 5)):
            with self.subTest(d=d):
                table = boxes.chern_table_exact(boxes.level_box(d, L))
                violations, plane, equal = sectors.ballot_violations(table)
                self.assertEqual((violations, plane, equal), ([], [], []))


class TestZeroInsertion(unittest.TestCase):
    def test_newton_coefficients_invert_binomial_sums(self):
        from math import comb

        coeffs = [2, 5, 4, 1]
        values = [sum(c * comb(z, k) for k, c in enumerate(coeffs)) for z in range(7)]
        self.assertEqual(sectors.newton_coefficients(values), coeffs + [0, 0, 0])

    def test_charge_two_family_is_a_cubic_with_positive_newton_coefficients(self):
        # C(2, -2, 0^z) = 2 + 5z + 4 binom(z,2) + binom(z,3), checked out to z = 5
        by_d = {d: boxes.chern_table_exact(boxes.charge_box(d, 2)) for d in range(2, 8)}
        series = sectors.zero_insertion_series(by_d, (2, -2))
        self.assertEqual(len(series), 6)
        self.assertEqual(sectors.newton_coefficients([c for c, _ in series]), [2, 5, 4, 1, 0, 0])
        self.assertTrue(all(v >= 0 for v in sectors.newton_coefficients([c - b for c, b in series])))


class TestStabilisation(unittest.TestCase):
    def test_top_face_is_the_previous_series(self):
        for d in (5, 6, 7):
            with self.subTest(d=d):
                self.assertTrue(sectors.stabilisation_holds(d, [12, 10, 8, 6, 4][: d - 2]))


if __name__ == "__main__":
    unittest.main()
