"""
The ballot conjecture C(M) >= b(M) at d = 4, reduced to exact integer-set inclusions.

The argument (report/ballot.tex, "The conjecture for d = 4"):

1. Lemma 4 writes F_4 = (1 + X + S + Y)(1 + V1)(1 + V2)(1 + V3)(1 + V4), every
   factor a series with nonnegative integer coefficients.  Expanding, F_4 is a
   sum of 64 products, each with coefficient >= 1 exactly on its support, a
   Minkowski sum of the supports below.  Hence F_4(beta) >= N(beta), the number
   of products whose support contains beta.

2. Z_4 = P_4 - 1_{zero set} + 1_T, where P_4 is the pure chain series and the
   zero set of F_4 is moved onto T by an explicit packet-preserving bijection.
   So the packet sums of Z_4 are b(M).

3. It remains to show N >= w + 1_D, where w = 1_{W1} + 1_T is the coefficient
   of Z_4 and D is the set of decreasing cells with beta_1 >= 2 (every packet
   with max M >= 2 has its decreasing ordering there).  Since w + 1_D <= 3,
   this is exactly: for k = 1, 2, 3 the set {w + 1_D >= k} lies inside
   the set of cells covered by at least k supports.  Those are inclusions of
   Presburger sets, decided exactly by isl -- no truncation anywhere.

Then C(M) >= packet sum of N >= packet sum of Z_4 = b(M), with one extra unit
whenever max M >= 2, which proves the conjecture at d = 4 with its equality
case (the plane sector at fixed d is finite and checked directly).
"""

from __future__ import annotations

from itertools import combinations, product
from typing import Dict, List

#: Supports of the factors of Lemma 4 (coefficient >= 1 exactly there).
FACTOR_SUPPORTS: Dict[str, str] = {
    "1": "{ [a,b,c] : a = 0 and b = 0 and c = 0 }",
    "X": "{ [a,b,c] : a >= 1 and b = 0 and c = 0 }",  # x1/(1-2x1)
    "S": "{ [a,b,c] : a = 0 and b >= 1 and c = b }",  # s/(1-2s), s = x2x3
    "Y": "{ [a,b,c] : a >= 2 and b >= 1 and c = b }",  # x1^2 s/((1-2x1)(1-s-t))
    "V1": "{ [a,b,c] : a >= 1 and b = a and c = 0 }",  # x1x2/(1-2x1x2)
    "V2": "{ [a,b,c] : a >= 1 and b >= a and c = 0 }",  # x1x2/(1-x2-x1x2)
    "V3": "{ [a,b,c] : a >= 1 and b = a and c = a }",  # t/(1-2t)
    "V4": "{ [a,b,c] : a >= 1 and b = a and c >= a }",  # t/(1-x3-t)
}

#: Cells of Z_4 with coefficient >= 1 (the support of F_4) and the targets T (coefficient 2).
W1 = (
    "{ [a,b,c] : a = 0 and b = 0 and c = 0; [a,b,c] : a = 0 and b >= 1 and c = b;"
    " [a,b,c] : a >= 1 and b >= 1 and c >= 0; [a,b,c] : a >= 1 and b = 0 and c = 0 }"
)
TARGETS = (
    "{ [a,b,c] : a >= 1 and b >= 1 and c = b; [a,b,c] : a >= 1 and b = a and c = 0;"
    " [a,b,c] : a >= 1 and c >= 1 and b = a + c }"
)
#: Decreasing cells: alpha = (a, b - a, c - b, -c) weakly decreasing, with a >= 2.
DECREASING = "{ [a,b,c] : a >= 2 and a >= b - a and b - a >= c - b and c - b >= -c and b >= 0 and c >= 0 }"
ZERO_SET = "{ [a,b,c] : a = 0 and b >= 0 and c >= 0 and b != c; [a,b,c] : a >= 1 and b = 0 and c >= 1 }"
ORTHANT = "{ [a,b,c] : a >= 0 and b >= 0 and c >= 0 }"


def product_supports():
    """The 64 product supports as isl sets (empty ones dropped)."""
    import islpy as isl

    base = [isl.Set(FACTOR_SUPPORTS[k]) for k in ("1", "X", "S", "Y")]
    vs = [isl.Set(FACTOR_SUPPORTS[k]) for k in ("V1", "V2", "V3", "V4")]
    out = []
    for b in base:
        for use in product((0, 1), repeat=4):
            s = b
            for u, v in zip(use, vs):
                if u:
                    s = s.sum(v)
            s = s.coalesce()
            if not s.is_empty():
                out.append(s)
    return out


def covered_at_least(supports: List, k: int):
    """Cells lying in at least ``k`` of the supports."""
    import islpy as isl

    acc = isl.Set("{ [a,b,c] : 1 = 0 }")
    for combo in combinations(supports, k):
        inter = combo[0]
        for s in combo[1:]:
            inter = inter.intersect(s)
            if inter.is_empty():
                break
        else:
            acc = acc.union(inter).coalesce()
    return acc


def verify_d4() -> Dict[str, bool]:
    """Every set inclusion the d = 4 proof needs, decided exactly."""
    import islpy as isl

    orthant = isl.Set(ORTHANT)
    w1, t, dec, zero = (isl.Set(x) for x in (W1, TARGETS, DECREASING, ZERO_SET))
    sup = product_supports()
    checks = {
        # W1 is exactly the orthant minus the zero set, and T lies inside W1
        "W1 = orthant minus zero set": w1.is_equal(orthant.subtract(zero)),
        "T inside W1": t.is_subset(w1),
        "D inside W1": dec.is_subset(w1),
    }
    # w + 1_D >= k  ->  covered by >= k supports, for k = 1, 2, 3
    level1 = w1.union(dec)
    level2 = t.union(dec)
    level3 = t.intersect(dec)
    checks["level 1 covered"] = level1.is_subset(covered_at_least(sup, 1))
    checks["level 2 covered"] = level2.is_subset(covered_at_least(sup, 2))
    checks["level 3 covered"] = level3.is_subset(covered_at_least(sup, 3))
    return checks


# --------------------------------------------------------------------------
# d = 5: the GBC-gauged chamber series
# --------------------------------------------------------------------------


def verify_d5_block(path=None) -> bool:
    """
    Re-verify, in exact rationals, the certificate for the six-factor block of
    the GBC-gauged d = 5 series (report/ballot.tex, "Toward d = 5"):

        N_block = sum_S P_S * prod_{r in S} (1 - f_r)  +  P_empty,

    with every P_S and the remainder P_empty coefficientwise nonnegative.
    Dividing by prod_r (1 - f_r) exhibits the block as a nonnegative series.
    """
    import json
    from fractions import Fraction
    from pathlib import Path

    from .polynomial import poly_mul

    path = Path(path or Path(__file__).resolve().parents[2] / "results" / "d5_block_certificate.json")
    data = json.loads(path.read_text())

    def poly(rows, frac=False):
        return {tuple(k): (Fraction(v) if frac else v) for k, v in rows}

    def one_minus(p):
        out = {(0, 0, 0, 0): 1}
        for k, v in p.items():
            out[k] = out.get(k, 0) - v
        return out

    num = {(0, 0, 0, 0): 1}
    for f in data["numerator_factors"]:
        num = poly_mul(num, one_minus(poly(f["poly"])))
    dens = [poly(f["poly"]) for f in data["denominator_factors"]]
    remainder = {k: Fraction(v) for k, v in num.items()}
    for part in data["parts"]:
        P = poly(part["poly"], frac=True)
        if any(v < 0 for v in P.values()):
            return False
        g = {(0, 0, 0, 0): Fraction(1)}
        for r in part["subset"]:
            g = poly_mul(g, {k: Fraction(v) for k, v in one_minus(dens[r]).items()})
        for k, v in poly_mul(P, g).items():
            remainder[k] = remainder.get(k, 0) - v
    return all(v >= 0 for v in remainder.values())
