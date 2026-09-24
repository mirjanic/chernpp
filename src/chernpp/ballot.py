"""
The ballot conjecture C(M) >= b(M) for d <= 5, reduced to exact checks.

Everything here returns named checks that are decided exactly -- isl for
integer-set statements, exact integer or rational arithmetic for identities,
CRT-verified tables for finitely many packets.  Nothing is sampled.

d = 4 (report/ballot.tex, "The conjecture for d = 4"):

1. Lemma 4 writes F_4 = (1 + X + S + Y)(1 + V1)(1 + V2)(1 + V3)(1 + V4), every
   factor a series with nonnegative integer coefficients.  Expanding, F_4 is a
   sum of 64 products, each with coefficient >= 1 exactly on its support, a
   Minkowski sum of the supports below.  Hence F_4(beta) >= N(beta), the number
   of products whose support contains beta.

2. Z_4 = P_4 - 1_{zero set} + 1_T, where P_4 is the pure chain series and the
   zero set of F_4 is moved onto T by the explicit word-preserving bijection
   :data:`BIJECTION_D4`.  So the packet sums of Z_4 are b(M).

3. It remains to show N >= w + 1_D, where w = 1_{W1} + 1_T is the coefficient
   of Z_4 and D is the set of decreasing cells with beta_1 >= 2 (every packet
   with max M >= 2 has its decreasing ordering there).  Since w + 1_D <= 3,
   this is exactly: for k = 1, 2, 3 the set {w + 1_D >= k} lies inside
   the set of cells covered by at least k supports.  Those are inclusions of
   Presburger sets, decided exactly by isl -- no truncation anywhere.

Then C(M) >= packet sum of N >= packet sum of Z_4 = b(M), with one extra unit
whenever max M >= 2, which proves the conjecture at d = 4 with its equality
case (the plane sector at fixed d is finite and checked directly).

d = 5 (report/ballot.tex, "The conjecture for d = 5") uses a different and
simpler endgame, the *dominant cell*:

1. The kernel GBC is residue-null (:func:`d5_nullity_checks`), so C(M) is also
   the packet sum of the gauged series F_5^GDJ of the numerator Q_5 - GBC = GDJ.
2. F_5^GDJ = C * prod_{i<=6} (1 + V_i) with C an explicit sum of eleven
   nonnegative series (:func:`d5_positive_form_checks`), so F_5^GDJ >= 0.
3. Hence C(M) >= F_5^GDJ(beta_max(M)), the coefficient of the decreasing
   ordering.  For max M >= 10 that single coefficient is >= 2^7 = 128 > 120 >=
   b(M) (:func:`growth_cover`, an isl inclusion on the decreasing cone).
4. The finitely many packets with max M <= 9 are checked from an exact table.

The dominant-cell argument re-proves d = 4 as well (:func:`verify_d4_dominant`),
independently of steps 2-3 above.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations, product
from typing import Dict, List, Optional, Sequence, Tuple

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

#: The bijection of step 2, as (domain, permutation) pieces.  A piece sends the
#: word ``alpha`` of a cell to the rearrangement ``alpha[perm]``, so it preserves
#: the multiset (the packet) by construction; the image cell is its prefix sums.
BIJECTION_D4: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    # (a, -a, c, -c) -> (a, c, -a, -c): swap the middle letters
    ("{ [a,b,c] : b = 0 and a >= 1 and c >= 1 }", (0, 2, 1, 3)),
    # (0, u, v, w) -> (u, v, 0, w): the leading zero moves to third place
    ("{ [a,b,c] : a = 0 and b >= 1 and c >= 1 and b != c }", (1, 2, 0, 3)),
    # (0, 0, c, -c) -> (c, 0, 0, -c)
    ("{ [a,b,c] : a = 0 and b = 0 and c >= 1 }", (2, 0, 1, 3)),
    # (0, b, -b, 0) -> (b, 0, -b, 0)
    ("{ [a,b,c] : a = 0 and c = 0 and b >= 1 }", (1, 0, 2, 3)),
)


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


# --------------------------------------------------------------------------
# words and cells
# --------------------------------------------------------------------------


def _word_forms(cell: Sequence[str]) -> List[str]:
    """The word ``alpha`` of a cell ``beta``, as affine forms in its coordinates."""
    forms = [cell[0]] + [f"({cell[k]}) - ({cell[k - 1]})" for k in range(1, len(cell))]
    return forms + [f"-({cell[-1]})"]


def _rearrangement_map(domain: str, perm: Sequence[int], cell: Sequence[str]):
    """The isl map ``beta -> prefix sums of alpha[perm]`` on ``domain``."""
    import islpy as isl

    word = _word_forms(cell)
    new = [word[p] for p in perm]
    outs = [f"o{k}" for k in range(len(cell))]
    eqs = [f"o{k} = {' + '.join('(' + w + ')' for w in new[: k + 1])}" for k in range(len(cell))]
    # The affine map alone, then restricted: splicing the domain's constraints into
    # the string would bind the equations to only one disjunct of a union.
    affine = isl.Map(f"{{ [{', '.join(cell)}] -> [{', '.join(outs)}] : {' and '.join(eqs)} }}")
    return affine.intersect_domain(isl.Set(domain))


def verify_d4_bijection() -> Dict[str, bool]:
    """Step 2 at d = 4: :data:`BIJECTION_D4` is a bijection from the zero set onto T."""
    import islpy as isl

    cell = ("a", "b", "c")
    zero, targets = isl.Set(ZERO_SET), isl.Set(TARGETS)
    maps = [_rearrangement_map(dom, perm, cell) for dom, perm in BIJECTION_D4]
    domains = [isl.Set(dom) for dom, _ in BIJECTION_D4]
    images = [m.range() for m in maps]
    union_dom = domains[0]
    union_img = images[0]
    for s in domains[1:]:
        union_dom = union_dom.union(s)
    for s in images[1:]:
        union_img = union_img.union(s)
    return {
        "bijection: pieces cover the zero set": union_dom.coalesce().is_equal(zero),
        "bijection: pieces are disjoint": all(p.intersect(q).is_empty() for p, q in combinations(domains, 2)),
        "bijection: each piece is injective": all(m.is_injective() for m in maps),
        "bijection: images are disjoint": all(p.intersect(q).is_empty() for p, q in combinations(images, 2)),
        "bijection: images fill T": union_img.coalesce().is_equal(targets),
    }


def verify_d4() -> Dict[str, bool]:
    """Every set inclusion and the bijection the d = 4 proof needs, decided exactly."""
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
    checks.update(verify_d4_bijection())
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
    the GBC-gauged d = 5 series in the other pairing of factors
    (report/ballot.tex, section "The conjecture for d = 5", the remark on a second block):

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


# --------------------------------------------------------------------------
# the dominant cell: explicit exponential lower bounds on product terms
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Atom:
    """
    A series with nonnegative coefficients, parametrised over its support.

    The exponent is ``cell`` (affine forms in ``params``, all >= 0, subject to
    ``constraints``), one exponent per parameter value, and the coefficient
    there is at least ``2^E`` with ``E = sum(powers) + sum_j min(n_j, m_j - n_j)``
    over ``binomials``.  That covers the two shapes that occur:

    * ``1/(1 - 2u)`` has coefficient exactly ``2^k`` at ``u^k`` (a *power*);
    * ``1/(1 - u - v)`` has coefficient ``C(m, n) >= 2^min(n, m - n)`` at
      ``u^(m - n) v^n`` (a *binomial*; ``C(m, n) >= C(2j, j) >= 2^j``).

    Several parameter tuples may reach the same cell (``x^2 s / ((1 - 2x)(1 - s - xs))``
    reaches ``x^3 s^2`` from ``(k, n) = (1, 0)`` and ``(0, 1)``).  Each tuple
    contributes its product to the coefficient there and every contribution is
    nonnegative, so the coefficient is at least the bound of any single tuple --
    which is all :func:`term_set` uses.  :func:`atom_series` lists the best
    such bound per cell, so tests can hold both claims against the true series.
    """

    params: Tuple[str, ...]
    cell: Tuple[str, ...]
    constraints: Tuple[str, ...] = ()
    powers: Tuple[str, ...] = ()
    binomials: Tuple[Tuple[str, str], ...] = ()


def term_set(atoms: Sequence[Atom], E: int, cell_names: Sequence[str]):
    """
    ``{beta : the product of atoms has coefficient >= 2^E at beta}``, as an isl set.

    Sound, not exact: a product's coefficient at ``beta`` is a sum over all ways
    of splitting ``beta`` among the atoms, and this set asks only that *one*
    split carry ``2^E`` on its own.
    """
    import islpy as isl

    params: List[str] = []
    cons: List[str] = []
    sums: List[List[str]] = [[] for _ in cell_names]
    exps: List[str] = []
    for idx, atom in enumerate(atoms):
        ren = {p: f"{p}_{idx}" for p in atom.params}

        def rename(s: str) -> str:
            for p in sorted(ren, key=len, reverse=True):
                s = s.replace(p, ren[p])
            return s

        params += list(ren.values())
        cons += [f"{v} >= 0" for v in ren.values()] + [rename(c) for c in atom.constraints]
        for j, form in enumerate(atom.cell):
            sums[j].append(f"({rename(form)})")
        exps += [f"({rename(p)})" for p in atom.powers]
        for n, m in atom.binomials:
            e = f"E{len(params)}_{idx}"
            params.append(e)
            n, m = rename(n), rename(m)
            cons += [f"{e} >= 0", f"{e} <= {n}", f"{e} <= ({m}) - ({n})"]
            exps.append(e)
    cons += [f"{v} = {' + '.join(s) if s else '0'}" for v, s in zip(cell_names, sums)]
    cons.append(f"{' + '.join(exps) if exps else '0'} >= {E}")
    exists = f"exists {', '.join(params)} : " if params else ""
    return isl.Set(f"{{ [{', '.join(cell_names)}] : {exists}{' and '.join(cons)} }}")


def product_terms(base: Dict[str, Atom], factors: Dict[str, Atom]):
    """The expansion of ``(sum of base) * prod (1 + factor)``: ``(name, atoms)`` per term."""
    for bname, batom in base.items():
        for use in product((0, 1), repeat=len(factors)):
            chosen = [k for u, k in zip(use, factors) if u]
            yield (bname,) + tuple(chosen), [batom] + [factors[k] for k in chosen]


def decreasing_cone(d: int, cell_names: Sequence[str], min_charge: int):
    """Cells whose word is weakly decreasing with first letter >= ``min_charge``."""
    import islpy as isl

    word = _word_forms(cell_names)
    cons = [f"{c} >= 0" for c in cell_names] + [f"{cell_names[0]} >= {min_charge}"]
    cons += [f"{word[k]} >= {word[k + 1]}" for k in range(d - 1)]
    return isl.Set(f"{{ [{', '.join(cell_names)}] : {' and '.join(cons)} }}")


def growth_cover(
    base: Dict[str, Atom],
    factors: Dict[str, Atom],
    region,
    E: int,
    cell_names: Sequence[str],
):
    """
    Remove from ``region`` every cell where some product term is ``>= 2^E``.

    Returns ``(remaining, used)``.  ``remaining`` empty is the statement that the
    whole series is at least ``2^E`` on ``region`` -- decided by isl, exactly.
    """
    rest, used = region, []
    for name, atoms in product_terms(base, factors):
        if rest.is_empty():
            break
        s = term_set(atoms, E, cell_names)
        if rest.intersect(s).is_empty():
            continue
        rest = rest.subtract(s).coalesce()
        used.append(name)
    return rest, used


def atom_series(atom: Atom, nvars: int, limit: int) -> Dict[Tuple[int, ...], int]:
    """
    ``{cell: best 2^E lower bound}`` over parameter values up to ``limit``, for
    tests: each atom's claimed support and bound are compared with the true series.
    """
    out: Dict[Tuple[int, ...], int] = {}
    names = atom.params
    for values in product(range(limit + 1), repeat=len(names)):
        env = dict(zip(names, values))
        if not all(eval(c, {}, env) for c in atom.constraints):  # constraints are "n <= m"
            continue
        cell = tuple(eval(f, {}, env) for f in atom.cell)
        if len(cell) != nvars:
            raise ValueError("cell has the wrong number of coordinates")
        e = sum(eval(p, {}, env) for p in atom.powers)
        e += sum(min(eval(n, {}, env), eval(m, {}, env) - eval(n, {}, env)) for n, m in atom.binomials)
        out[cell] = max(out.get(cell, 0), 2**e)
    return out


# --------------------------------------------------------------------------
# small charge: finitely many packets, from an exact table
# --------------------------------------------------------------------------


def small_charge_checks(d: int, top_charge: int) -> Dict[str, bool]:
    """
    Every packet with ``max M <= top_charge``: ``C = b`` on the plane sector and
    ``C >= b + 1`` off it.  Finitely many packets, from the exact (CRT-verified)
    charge box, so this is a proof for them, not a sample.
    """
    from . import boxes, sectors

    table = boxes.chern_table_exact(boxes.charge_box(d, top_charge))
    plane_ok = all(c == sectors.ballot_count(m) for m, c in table.items() if max(m) <= 1)
    off_ok = all(c >= sectors.ballot_count(m) + 1 for m, c in table.items() if max(m) >= 2)
    return {
        f"max M <= 1: C = b ({sum(1 for m in table if max(m) <= 1)} packets)": plane_ok,
        f"2 <= max M <= {top_charge}: C >= b + 1 ({sum(1 for m in table if max(m) >= 2)} packets)": off_ok,
    }


def max_ballot_count(d: int) -> int:
    """``b(M) <= d!``: a ballot ordering is in particular an ordering."""
    from math import factorial

    return factorial(d)


# --------------------------------------------------------------------------
# d = 4 by the dominant cell (a second, independent proof)
# --------------------------------------------------------------------------

#: Lemma 4 in atoms: F_4 = (1 + X + S + Y) * prod (1 + V_i), cells (a, b, c).
D4_BASE: Dict[str, Atom] = {
    "1": Atom((), ("0", "0", "0")),
    "X": Atom(("k",), ("1+k", "0", "0"), powers=("k",)),
    "S": Atom(("k",), ("0", "1+k", "1+k"), powers=("k",)),
    # x1^2 s / ((1 - 2x1)(1 - s - t)), t = x1 s: x1^(2+k+n) s^(1+m), coefficient 2^k C(m, n)
    "Y": Atom(("k", "n", "m"), ("2+k+n", "1+m", "1+m"), ("n <= m",), ("k",), (("n", "m"),)),
}
D4_FACTORS: Dict[str, Atom] = {
    "V1": Atom(("k",), ("1+k", "1+k", "0"), powers=("k",)),
    "V2": Atom(("n", "m"), ("1+n", "1+m", "0"), ("n <= m",), binomials=(("n", "m"),)),
    "V3": Atom(("k",), ("1+k", "1+k", "1+k"), powers=("k",)),
    "V4": Atom(("n", "m"), ("1+n", "1+n", "1+m"), ("n <= m",), binomials=(("n", "m"),)),
}


def verify_d4_dominant(E: int = 5, min_charge: int = 9) -> Dict[str, bool]:
    """
    d = 4 again: F_4 >= 0 (Lemma 4), so C(M) >= F_4(beta_max(M)) >= 2^E > 4! >= b(M)
    once max M >= ``min_charge``; the packets below are checked from the table.
    """
    if 2**E <= max_ballot_count(4):
        raise ValueError("2^E must exceed 4! for the dominant cell to beat every b(M)")
    cells = ("a", "b", "c")
    rest, used = growth_cover(D4_BASE, D4_FACTORS, decreasing_cone(4, cells, min_charge), E, cells)
    checks = {f"decreasing cells with max M >= {min_charge} have F_4 >= 2^{E}": rest.is_empty()}
    checks.update(small_charge_checks(4, min_charge - 1))
    return checks


# --------------------------------------------------------------------------
# d = 5, step 1: GBC is residue-null (the formal two-swap argument)
# --------------------------------------------------------------------------

Form = Tuple[int, ...]  # a linear form sum_j c_j z_j, as (c_1, ..., c_d)

#: The weights of the external findings summary, section 4.3 (credited in the report).
D5_WEIGHTS: Dict[str, Form] = {
    "B": (2, -1, 0, 0, 0),
    "F": (2, 0, -1, 0, 0),
    "E": (1, 1, -1, 0, 0),
    "D": (0, 2, 0, -1, 0),
    "M": (0, 1, 1, 0, -1),
    "C": (1, 0, 0, 1, -1),
    "G": (2, 1, 0, 0, -1),
    "J": (2, 0, 1, 0, -1),
}
#: The source multiset S of P(t) = prod_{alpha in S} (alpha - t).
D5_SOURCES: Tuple[Form, ...] = ((2, 0, 0, 0, 0), (1, 1, 0, 0, 0), (0, 2, 0, 0, 0), (1, 0, 1, 0, 0))


def _root(alpha: Form, k: int) -> Form:
    """``alpha - z_k`` (``k`` from 1)."""
    return tuple(c - (1 if j == k - 1 else 0) for j, c in enumerate(alpha))


def _swap(form: Form, i: int) -> Form:
    """The transposition ``s_i = (i, i+1)`` (``i`` from 1) acting on a linear form."""
    f = list(form)
    f[i - 1], f[i] = f[i], f[i - 1]
    return tuple(f)


def _normal(form: Form) -> Tuple[Form, int]:
    """``(form', sign)`` with ``form = sign * form'`` and the top coefficient of ``form'`` negative."""
    top = max(j for j, c in enumerate(form) if c)
    return (form, 1) if form[top] < 0 else (tuple(-c for c in form), -1)


def _as_multiset(forms: Sequence[Form]) -> Tuple[Counter, int]:
    sign, out = 1, Counter()
    for f in forms:
        g, s = _normal(f)
        out[g] += 1
        sign *= s
    return out, sign


def invariant_product(forms: Sequence[Form], i: int) -> bool:
    """Whether ``prod forms`` is invariant (not merely up to sign) under ``s_i``."""
    a, sa = _as_multiset(forms)
    b, sb = _as_multiset([_swap(f, i) for f in forms])
    return a == b and sa == sb


def swap_safe(forms: Sequence[Form], i: int) -> bool:
    """
    The chamber-safety condition for ``s_i``: no factor whose top variable is
    ``z_{i+1}`` involves ``z_i``.  Then every factor expands in the chamber
    domain as a series in monomials ``z_a / z_b`` (``a < b``) other than
    ``z_i / z_{i+1}``, a set that ``s_i`` permutes; so the expansion commutes with
    ``s_i``.  (Every form here has a nonzero top coefficient, hence a chamber expansion.)
    """
    for f in forms:
        top = max(j for j, c in enumerate(f) if c) + 1
        if top == i + 1 and f[i - 1] != 0:
            return False
    return True


def _poly_of_forms(forms: Sequence[Form]):
    """``prod forms`` as a polynomial ``{exponent: coefficient}`` in ``z``."""
    from .polynomial import poly_mul

    d = len(forms[0])
    out = {(0,) * d: 1}
    for f in forms:
        lin = {tuple(int(j == k) for j in range(d)): c for k, c in enumerate(f) if c}
        out = poly_mul(out, lin)
    return {e: c for e, c in out.items() if c}


def _linear_factors(factor: Dict[Tuple[int, ...], int], d: int) -> Form:
    """
    A stored chamber denominator ``1 - f`` as the linear form ``lambda - z_k``.

    With ``z_d = 1`` and ``x_j = z_j / z_{j+1}``, the chamber monomial
    ``x_a x_{a+1} ... x_{k-1}`` is ``z_a / z_k``; every monomial of ``f`` must be
    of that shape with one common ``k``, and then ``1 - f = (z_k - lambda) / z_k``.
    """
    lam, ks = [0] * d, set()
    for e, c in factor.items():
        nz = [j for j, v in enumerate(e) if v]
        a, k = nz[0], nz[-1] + 1
        if any(e[j] != (1 if a <= j < k else 0) for j in range(len(e))):
            raise ValueError(f"monomial {e} is not a ratio z_a / z_k")
        lam[a] += int(c)
        ks.add(k)
    if len(ks) != 1:
        raise ValueError(f"factor {factor} has no common top variable")
    return _root(tuple(lam), ks.pop() + 1)


def d5_nullity_checks() -> Dict[str, bool]:
    """
    The hypotheses of the two-swap argument that GBC is residue-null at d = 5.

    With ``D_5`` the product of the thirteen denominator forms and ``V`` the
    Vandermonde, every chamber series is the chamber-domain expansion of
    ``+-P V / D_5`` (``d5_normalisation``), and ``GBC = BMC + FBC`` because
    ``G = F + M``.  Then (report/ballot.tex, Lemma "swap"):

    * ``BMC / D_5 = 1 / (F E P(z4) P(z5))`` is ``s_4``-invariant with an
      ``s_4``-safe denominator, so its packet sums vanish;
    * ``H = FBC / D_5 = 1 / (E P(z4) P(z5) M(z5))``: the part ``H + s_4 H`` is
      ``s_4``-invariant and ``s_4``-safe; the part ``H - s_4 H`` equals
      ``(z5 - z4) / (E T(z4) T(z5))``, ``T(t) = P(t) M(t)``, which is
      ``s_1``-invariant and ``s_1``-safe.
    """
    from .artifacts import load_algebra
    from .optimisation.gauge import _z_degree, to_z

    w = D5_WEIGHTS
    alg = load_algebra(5)
    Q = to_z(alg.multidegree, 5, _z_degree(alg))
    stored = [_linear_factors(f, 5) for f in alg.denominator_factors]
    P4 = [_root(a, 4) for a in D5_SOURCES]
    P5 = [_root(a, 5) for a in D5_SOURCES]
    M4, M5 = _root((0, 1, 1, 0, 0), 4), w["M"]
    T_sources = list(D5_SOURCES) + [(0, 1, 1, 0, 0)]
    T4 = [_root(a, 4) for a in T_sources]
    T5 = [_root(a, 5) for a in T_sources]

    from .polynomial import poly_add

    gdj_gbc = poly_add(_poly_of_forms([w["G"], w["D"], w["J"]]), _poly_of_forms([w["G"], w["B"], w["C"]]))
    checks = {
        "Q_5 = G D J + G B C": {e: c for e, c in gdj_gbc.items() if c} == {e: c for e, c in Q.items() if c},
        "G = F + M": tuple(a + b for a, b in zip(w["F"], w["M"])) == w["G"],
        # up to one overall sign, which multiplies every numerator alike; this is
        # also what makes B M C / D_5 = 1 / (F E P(z4) P(z5)) and F B C / D_5 = H
        "D_5 = B F E P(z4) P(z5) M(z5) C": _as_multiset(stored)[0]
        == _as_multiset([w["B"], w["F"], w["E"], *P4, *P5, w["M"], w["C"]])[0],
        "1 / (F E P(z4) P(z5)) is s4-invariant": invariant_product([w["F"], w["E"], *P4, *P5], 4),
        "1 / (F E P(z4) P(z5)) is s4-safe": swap_safe([w["F"], w["E"], *P4, *P5], 4),
        "H + s4 H has an s4-safe denominator": swap_safe([w["E"], *P4, *P5, M4, M5], 4),
        "M(z4) - M(z5) = z5 - z4": tuple(a - b for a, b in zip(M4, M5)) == (0, 0, 0, -1, 1),
        "E T(z4) T(z5) is s1-invariant": invariant_product([w["E"], *T4, *T5], 1),
        "z5 - z4 is s1-invariant": _swap((0, 0, 0, -1, 1), 1) == (0, 0, 0, -1, 1),
        "(z5 - z4) / (E T(z4) T(z5)) is s1-safe": swap_safe([w["E"], *T4, *T5], 1),
        # the argument needs the swaps: D_5 itself is safe for neither
        "control: D_5 is not s4-safe": not swap_safe(stored, 4),
        "control: F B C / D_5 is not s4-invariant": not invariant_product([w["E"], *P4, *P5, M5], 4),
    }
    checks.update(d5_normalisation())
    return checks


def _chamber_z(d: int):
    """``z_j`` in chamber coordinates (``z_d = 1``, ``z_j = x_j ... x_{d-1}``), as polynomials."""
    return [{tuple(int(k >= j) for k in range(d - 1)): 1} for j in range(d - 1)] + [{(0,) * (d - 1): 1}]


def _chamber_form(form: Form, d: int):
    """A linear form in ``z`` as a polynomial in the chamber variables."""
    out = {}
    for mono, c in zip(_chamber_z(d), form):
        if c:
            ((e, _),) = mono.items()
            out[e] = out.get(e, 0) + c
    return out


def _chamber_poly_z(p: Dict[Tuple[int, ...], int], d: int):
    """A polynomial in ``z`` (exponent tuples of length ``d``) in chamber coordinates."""
    from .polynomial import poly_add

    out: Dict[Tuple[int, ...], int] = {}
    for e, c in p.items():
        b = tuple(sum(e[: k + 1]) for k in range(d - 1))  # z^e = x^b with b_k = e_1 + ... + e_k
        out = poly_add(out, {b: c})
    return {k: v for k, v in out.items() if v}


def _product(polys, nvars: int):
    from .polynomial import poly_mul

    out = {(0,) * nvars: 1}
    for p in polys:
        out = poly_mul(out, p)
    return {k: v for k, v in out.items() if v}


def d5_normalisation() -> Dict[str, bool]:
    """
    The chamber series of a numerator ``P`` is the chamber expansion of
    ``+-P V / D_5``, with no monomial factor: ``V = prod_{i<j} (z_j - z_i)``.

    Checked as a polynomial identity in the chamber variables for the stored
    ``Q_5``: ``N_5(x) * D_5(z(x)) = +- Q_5(z(x)) V(z(x)) * prod (1 - f_r(x))``.
    """
    from .artifacts import load_algebra
    from .optimisation.gauge import _z_degree, to_z
    from .polynomial import one_minus

    d, alg = 5, load_algebra(5)
    Q = _chamber_poly_z(to_z(alg.multidegree, d, _z_degree(alg)), d)
    V = _product(
        [
            _chamber_form(tuple(int(k == j) - int(k == i) for k in range(d)), d)
            for i in range(d)
            for j in range(i + 1, d)
        ],
        d - 1,
    )
    D = _product([_chamber_form(_linear_factors(f, d), d) for f in alg.denominator_factors], d - 1)
    lhs = _product([dict(alg.numerator), D], d - 1)
    rhs = _product([Q, V] + [one_minus(f, d - 1) for f in alg.denominator_factors], d - 1)
    neg = {k: -v for k, v in rhs.items()}
    return {"chamber series = +- expansion of Q_5 V / D_5": lhs in (rhs, neg)}


# --------------------------------------------------------------------------
# d = 5, step 2: the positive form of the gauged series
# --------------------------------------------------------------------------

#: F_5^GDJ = C * prod (1 + V_i) in chamber variables x1..x4, cells (a, b, c, e).
#: C (the block, in x = x1, b = x2, r = x3 x4) is the sum of the eleven series
#: of :data:`D5_BLOCK_TERMS`; the V_i are the six Lemma-1 ratios minus one.
D5_BLOCK_TERMS: Dict[str, Atom] = {
    "1": Atom((), ("0", "0", "0", "0")),
    # x / A, A = 1 - 2x
    "x/A": Atom(("k",), ("1+k", "0", "0", "0"), powers=("k",)),
    # xb / S, S = 1 - 2xb
    "xb/S": Atom(("k",), ("1+k", "1+k", "0", "0"), powers=("k",)),
    # x^2 b / (A S)
    "x2b/AS": Atom(("k1", "k2"), ("2+k1+k2", "1+k2", "0", "0"), powers=("k1+k2",)),
    # (br + br^2) / Q, Q = 1 - 2br, split into its two monomials
    "br/Q": Atom(("k",), ("0", "1+k", "1+k", "1+k"), powers=("k",)),
    "br2/Q": Atom(("k",), ("0", "1+k", "2+k", "2+k"), powers=("k",)),
    # x b^2 r / (S Q)
    "xb2r/SQ": Atom(("k1", "k2"), ("1+k1", "2+k1+k2", "1+k2", "1+k2"), powers=("k1+k2",)),
    # x^2 b r / (A U), U = 1 - r - xbr
    "x2br/AU": Atom(("k", "n", "m"), ("2+k+n", "1+n", "1+m", "1+m"), ("n <= m",), ("k",), (("n", "m"),)),
    # x^3 b^2 r / (A S U)
    "x3b2r/ASU": Atom(
        ("k", "s", "n", "m"), ("3+k+s+n", "2+s+n", "1+m", "1+m"), ("n <= m",), ("k+s",), (("n", "m"),)
    ),
    # x^2 b^3 r^2 / (S Q U)
    "x2b3r2/SQU": Atom(
        ("s", "t", "n", "m"), ("2+s+n", "3+s+t+n", "2+t+m", "2+t+m"), ("n <= m",), ("s+t",), (("n", "m"),)
    ),
    # x b^2 r^3 / (P V), P = 1 - 2xbr, V = 1 - r - br
    "xb2r3/PV": Atom(("k", "n", "m"), ("1+k", "2+k+n", "3+k+m", "3+k+m"), ("n <= m",), ("k",), (("n", "m"),)),
    # b r^3 / (Q V)
    "br3/QV": Atom(("k", "n", "m"), ("0", "1+k+n", "3+k+m", "3+k+m"), ("n <= m",), ("k",), (("n", "m"),)),
}
D5_FACTORS: Dict[str, Atom] = {
    # (1 - x2)/(1 - x2 - x1x2) - 1 = x1x2 / (1 - x2 - x1x2)
    "V1": Atom(("n", "m"), ("1+n", "1+m", "0", "0"), ("n <= m",), binomials=(("n", "m"),)),
    # (1 - x2x3)/(1 - x2x3 - x1x2x3) - 1
    "V2": Atom(("n", "m"), ("1+n", "1+m", "1+m", "0"), ("n <= m",), binomials=(("n", "m"),)),
    # (1 - x3)/(1 - x3 - x1x2x3) - 1
    "V3": Atom(("n", "m"), ("1+n", "1+n", "1+m", "0"), ("n <= m",), binomials=(("n", "m"),)),
    # (1 - x4)/(1 - x4 - p) - 1, p = x1x2x3x4
    "V4": Atom(("n", "m"), ("1+n", "1+n", "1+n", "1+m"), ("n <= m",), binomials=(("n", "m"),)),
    # (1 - p)/(1 - p - q) - 1, q = x2x3x4
    "V5": Atom(("n", "m"), ("n", "1+m", "1+m", "1+m"), ("n <= m",), binomials=(("n", "m"),)),
    # (1 - x1x2x3)/(1 - 2x1x2x3) - 1
    "V6": Atom(("k",), ("1+k", "1+k", "1+k", "0"), powers=("k",)),
}


def _d5_gauged_numerator(alg):
    """``(sign * GDJ / x^corr * V, divisible)`` in chamber variables: the numerator of F_5^GDJ."""
    from .optimisation.gauge import correction_and_sign

    n = alg.nvars
    corr, sign = correction_and_sign(alg)
    gdj = _product([_chamber_form(D5_WEIGHTS[k], alg.order) for k in "GDJ"], n)
    shifted = {tuple(e[i] - corr[i] for i in range(n)): sign * c for e, c in gdj.items()}
    divisible = all(min(e) >= 0 for e in shifted)
    return _product([shifted, dict(alg.vandermonde)], n), divisible


def gauged_algebra_d5():
    """
    The chamber algebra of F_5^GDJ: the stored denominators with the gauged
    numerator, in the shape :func:`chernpp.boxes.sweep` and ``exact_box`` accept.
    """
    from types import SimpleNamespace

    from .artifacts import load_algebra

    alg = load_algebra(5)
    numerator, divisible = _d5_gauged_numerator(alg)
    if not divisible:
        raise ValueError("GDJ is not divisible by the chamber correction")
    return SimpleNamespace(
        order=5, nvars=4, numerator=numerator, denominator_factors=list(alg.denominator_factors)
    )


def d5_positive_form_checks() -> Dict[str, bool]:
    """
    The gauged series is ``C * prod (1 + V_i)`` with ``C`` the eleven-term sum:
    both identities are checked exactly as polynomial identities after clearing
    denominators, in chamber variables.
    """
    from .artifacts import load_algebra
    from .polynomial import one_minus, poly_add

    d, n = 5, 4
    alg = load_algebra(d)
    w = D5_WEIGHTS

    def mono(*exps):
        return tuple(sum(e[i] for e in exps) for i in range(n))

    x1, x2, x3, x4 = (tuple(int(k == j) for k in range(n)) for j in range(n))
    b, r = x2, mono(x3, x4)
    p, q = mono(x1, x2, x3, x4), mono(x2, x3, x4)

    def om(*terms):  # 1 - sum c * monomial
        return one_minus({e: c for e, c in terms}, n)

    Ng, divisible = _d5_gauged_numerator(alg)
    A, S, P, Qd = om((x1, 2)), om((mono(x1, b), 2)), om((p, 2)), om((q, 2))
    U, Vd = om((r, 1), (p, 1)), om((r, 1), (q, 1))
    N_block = _product(
        [om((x1, 1)), om((mono(x1, b), 1)), om((r, 1)), om((q, 1)), om((r, 1), (p, 2)), om((q, 1), (p, 2))], n
    )
    ratios = [
        (om((x2, 1)), om((x2, 1), (mono(x1, x2), 1))),
        (om((mono(x2, x3), 1)), om((mono(x2, x3), 1), (mono(x1, x2, x3), 1))),
        (om((x3, 1)), om((x3, 1), (mono(x1, x2, x3), 1))),
        (om((x4, 1)), om((x4, 1), (p, 1))),
        (om((p, 1)), om((p, 1), (q, 1))),
        (om((mono(x1, x2, x3), 1)), om((mono(x1, x2, x3), 2))),
    ]
    dens = [one_minus(f, n) for f in alg.denominator_factors]
    lhs = _product([Ng] + [den for _, den in ratios] + [A, S, P, Qd, U, Vd], n)
    rhs = _product([num for num, _ in ratios] + [N_block] + dens, n)
    # the eleven terms, each times A S P Q U V: (monomial numerator, factors it keeps)
    keep = {"A": A, "S": S, "P": P, "Q": Qd, "U": U, "V": Vd}
    terms = [
        ({(0,) * n: 1}, "ASPQUV"),
        ({x1: 1}, "SPQUV"),
        ({mono(x1, b): 1}, "APQUV"),
        ({mono(x1, x1, b): 1}, "PQUV"),
        ({mono(b, r): 1, mono(b, r, r): 1}, "ASPUV"),
        ({mono(x1, b, b, r): 1}, "APUV"),
        ({mono(x1, x1, b, r): 1}, "SPQV"),
        ({mono(x1, x1, x1, b, b, r): 1}, "PQV"),
        ({mono(x1, x1, b, b, b, r, r): 1}, "APV"),
        ({mono(x1, b, b, r, r, r): 1}, "ASQU"),
        ({mono(b, r, r, r): 1}, "ASPU"),
    ]
    block_sum: Dict[Tuple[int, ...], int] = {}
    for num, kept in terms:
        block_sum = poly_add(block_sum, _product([num] + [keep[k] for k in kept], n))
    block_sum = {k: v for k, v in block_sum.items() if v}
    return {
        "GDJ is divisible by the chamber correction": divisible,
        "F_5^GDJ = C * prod (1 + V_i)": lhs == rhs,
        "C = eleven-term Stanley decomposition": block_sum == N_block,
    }


def verify_d5(E: int = 7, min_charge: int = 10) -> Dict[str, bool]:
    """
    Every check the d = 5 proof needs: nullity of GBC, the positive form, the
    dominant-cell growth on max M >= ``min_charge``, and the finite table below.
    """
    if 2**E <= max_ballot_count(5):
        raise ValueError("2^E must exceed 5! for the dominant cell to beat every b(M)")
    checks = {}
    checks.update(d5_nullity_checks())
    checks.update(d5_positive_form_checks())
    cells = ("a", "b", "c", "e")
    rest, used = growth_cover(D5_BLOCK_TERMS, D5_FACTORS, decreasing_cone(5, cells, min_charge), E, cells)
    checks[f"decreasing cells with max M >= {min_charge} have F_5^GDJ >= 2^{E}"] = rest.is_empty()
    checks.update(small_charge_checks(5, min_charge - 1))
    return checks


def d5_alternative_block(order: int = 5, degree: int = 6) -> Dict[str, bool]:
    """
    The other pairing of factors: ``(1 - q - p)`` in the block and ``(1 - p)/(1 - 2p)``
    among the ratios, in variables ``(u, y, w) = (x1, x2, x3 x4)``.  A denominator
    certificate of order 5 exists (re-verified exactly by :mod:`chernpp.certificates`,
    so it is a proof that this block is nonnegative); the degree-3 projection is
    infeasible at every order <= 4, which excludes lower orders at every degree.
    """
    from . import certificates
    from .polynomial import one_minus, poly_mul_many

    u, y, w = (1, 0, 0), (0, 1, 0), (0, 0, 1)
    yw, uy, uyw = (0, 1, 1), (1, 1, 0), (1, 1, 1)
    numerator = poly_mul_many(
        [one_minus(g, 3) for g in ({u: 1}, {yw: 1}, {yw: 1, uyw: 2}, {uy: 1}, {w: 1}, {w: 1, uyw: 2})], 3
    )
    factors = [{u: 2}, {yw: 1, uyw: 1}, {yw: 2}, {uy: 2}, {w: 1, uyw: 1}, {w: 1, yw: 1}]
    cert = certificates.search_certificate(numerator, factors, 3, order=order, max_degree=degree)
    lower = [certificates.projection_is_feasible(numerator, factors, 3, k, 3) for k in range(1, order)]
    return {
        f"order-{order} certificate of degree {degree} found and verified": cert is not None
        and cert.is_valid(numerator, factors),
        f"orders 1..{order - 1} are infeasible at projection degree 3": not any(lower),
    }
