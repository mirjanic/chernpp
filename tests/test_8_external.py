"""
Tier 8: agreement with Rimányi's published Thom polynomial tables.

The classical checks in tier 3 only pin down relative dimension zero. The
tables at

    https://tpp.web.unc.edu/thom-polynomials-relative-dimension-1/

give the full Thom polynomials at relative dimension one, computed by Rimányi's
restriction-equation method -- mathematics entirely independent of the
Bérczi--Szenes residue formula this suite implements. Agreement across all 15,
30 and 58 coefficients for A_4, A_5 and A_6 is therefore a genuine external
validation of Q_d, not a self-consistency check.

This does not replace our own computation; it constrains it.
"""

import re
import unittest

from chernpp.chern import chern_coefficients

#: Monomial-basis Thom polynomials at relative dimension l = 1, transcribed
#: verbatim from the table cited above. c[i] is the i-th relative Chern class.
RELATIVE_DIMENSION_ONE = {
    4: (
        "6*c[1]^3*c[5] +9*c[1]^2*c[2]*c[4] +2*c[1]^2*c[3]^2 +6*c[1]*c[2]^2*c[3] "
        "+c[2]^4 +54*c[1]^2*c[6] +53*c[1]*c[2]*c[5] +17*c[1]*c[3]*c[4] "
        "+16*c[2]^2*c[4] +4*c[2]*c[3]^2 +156*c[1]*c[7] +76*c[2]*c[6] +21*c[3]*c[5] "
        "+11*c[4]^2 +144*c[8]"
    ),
    5: (
        "24*c[1]^4*c[6] +38*c[1]^3*c[2]*c[5] +12*c[1]^3*c[3]*c[4] "
        "+25*c[1]^2*c[2]^2*c[4] +10*c[1]^2*c[2]*c[3]^2 +10*c[1]*c[2]^3*c[3] "
        "+c[2]^5 +336*c[1]^3*c[7] +400*c[1]^2*c[2]*c[6] +115*c[1]^2*c[3]*c[5] "
        "+39*c[1]^2*c[4]^2 +170*c[1]*c[2]^2*c[5] +95*c[1]*c[2]*c[3]*c[4] "
        "+5*c[1]*c[3]^3 +30*c[2]^3*c[4] +10*c[2]^2*c[3]^2 +1704*c[1]^2*c[8] "
        "+1366*c[1]*c[2]*c[7] +389*c[1]*c[3]*c[6] +233*c[1]*c[4]*c[5] "
        "+285*c[2]^2*c[6] +136*c[2]*c[3]*c[5] +68*c[2]*c[4]^2 +19*c[3]^2*c[4] "
        "+3696*c[1]*c[9] +1508*c[2]*c[8] +450*c[3]*c[7] +268*c[4]*c[6] +78*c[5]^2 "
        "+2880*c[10]"
    ),
    6: (
        "120*c[1]^5*c[7] +202*c[1]^4*c[2]*c[6] +55*c[1]^4*c[3]*c[5] "
        "+17*c[1]^4*c[4]^2 +141*c[1]^3*c[2]^2*c[5] +79*c[1]^3*c[2]*c[3]*c[4] "
        "+5*c[1]^3*c[3]^3 +55*c[1]^2*c[2]^3*c[4] +30*c[1]^2*c[2]^2*c[3]^2 "
        "+15*c[1]*c[2]^4*c[3] +c[2]^6 +2400*c[1]^4*c[8] +3272*c[1]^3*c[2]*c[7] "
        "+884*c[1]^3*c[3]*c[6] +450*c[1]^3*c[4]*c[5] +1704*c[1]^2*c[2]^2*c[6] "
        "+861*c[1]^2*c[2]*c[3]*c[5] +280*c[1]^2*c[2]*c[4]^2 +109*c[1]^2*c[3]^2*c[4] "
        "+425*c[1]*c[2]^3*c[5] +315*c[1]*c[2]^2*c[3]*c[4] +30*c[1]*c[2]*c[3]^3 "
        "+50*c[2]^4*c[4] +20*c[2]^3*c[3]^2 +18600*c[1]^3*c[9] "
        "+19358*c[1]^2*c[2]*c[8] +5393*c[1]^2*c[3]*c[7] +2594*c[1]^2*c[4]*c[6] "
        "+919*c[1]^2*c[5]^2 +6737*c[1]*c[2]^2*c[7] +3354*c[1]*c[2]*c[3]*c[6] "
        "+1890*c[1]*c[2]*c[4]*c[5] +378*c[1]*c[3]^2*c[5] +269*c[1]*c[3]*c[4]^2 "
        "+818*c[2]^3*c[6] +514*c[2]^2*c[3]*c[5] +247*c[2]^2*c[4]^2 "
        "+126*c[2]*c[3]^2*c[4] +3*c[3]^4 +69600*c[1]^2*c[10] +49432*c[1]*c[2]*c[9] "
        "+14544*c[1]*c[3]*c[8] +7012*c[1]*c[4]*c[7] +3868*c[1]*c[5]*c[6] "
        "+8686*c[2]^2*c[8] +4506*c[2]*c[3]*c[7] +2496*c[2]*c[4]*c[6] "
        "+706*c[2]*c[5]^2 +520*c[3]^2*c[6] +544*c[3]*c[4]*c[5] +86*c[4]^3 "
        "+125280*c[1]*c[11] +45816*c[2]*c[10] +14428*c[3]*c[9] +7064*c[4]*c[8] "
        "+3284*c[5]*c[7] +1408*c[6]^2 +86400*c[12]"
    ),
}

#: Expected term counts, as a guard against a transcription slip silently
#: shrinking the comparison.
TERM_COUNTS = {4: 15, 5: 30, 6: 58}

_TERM = re.compile(r"c\[(\d+)\](?:\^(\d+))?")


def parse_table(text):
    """``6*c[1]^3*c[5] + ...`` -> ``{sorted index tuple: coefficient}``."""
    result = {}
    for term in text.replace("-", "+-").split("+"):
        term = term.strip()
        if not term:
            continue
        head, _, _ = term.partition("c[")
        head = head.rstrip("*")
        coefficient = 1 if head in ("", "+") else (-1 if head == "-" else int(head))
        indices = []
        for index, power in _TERM.findall(term):
            indices += [int(index)] * (int(power) if power else 1)
        key = tuple(sorted(indices))
        if key in result:
            raise ValueError(f"duplicate monomial {key} in table")
        result[key] = coefficient
    return result


def computed_table(dim, relative_dimension):
    """Our Thom polynomial in the same ``{index tuple: coefficient}`` form."""
    _, chern = chern_coefficients(dim=dim, l_max=relative_dimension)
    result = {}
    for multiset, coefficient in chern.items():
        indices = [relative_dimension + 1 + alpha for alpha in multiset]
        if min(indices) < 0:
            continue  # c_j = 0 for j < 0
        key = tuple(sorted(i for i in indices if i > 0))  # c_0 = 1
        result[key] = result.get(key, 0) + coefficient
    return {k: v for k, v in result.items() if v != 0}


class TestTableParser(unittest.TestCase):
    def test_parses_coefficients_and_powers(self):
        parsed = parse_table("6*c[1]^3*c[5] +c[2]^4 +144*c[8]")
        self.assertEqual(parsed, {(1, 1, 1, 5): 6, (2, 2, 2, 2): 1, (8,): 144})

    def test_rejects_a_duplicated_monomial(self):
        with self.assertRaises(ValueError):
            parse_table("2*c[1] +3*c[1]")


class TestAgreementWithPublishedTables(unittest.TestCase):
    """Every coefficient must match, in both directions."""

    def test_transcription_has_the_expected_size(self):
        for dim, expected in TERM_COUNTS.items():
            with self.subTest(dim=dim):
                self.assertEqual(len(parse_table(RELATIVE_DIMENSION_ONE[dim])), expected)

    def test_thom_polynomials_agree_at_relative_dimension_one(self):
        for dim, text in RELATIVE_DIMENSION_ONE.items():
            published, computed = parse_table(text), computed_table(dim, 1)
            with self.subTest(dim=dim):
                self.assertEqual(
                    set(published), set(computed), "the two tables list different monomials"
                )
                for monomial in sorted(published):
                    self.assertEqual(
                        computed[monomial],
                        published[monomial],
                        f"A_{dim}, monomial {monomial}",
                    )

    def test_top_coefficient_matches_the_known_closed_form(self):
        # The coefficient of c_{d(l+1)} is (d-1)! * (l+1)^? -- rather than assume
        # a formula, just pin the published values: 144, 2880, 86400.
        for dim, expected in ((4, 144), (5, 2880), (6, 86400)):
            published = parse_table(RELATIVE_DIMENSION_ONE[dim])
            with self.subTest(dim=dim):
                self.assertEqual(published[(2 * dim,)], expected)
                self.assertEqual(computed_table(dim, 1)[(2 * dim,)], expected)


if __name__ == "__main__":
    unittest.main()
