"""
Analysis of the Bérczi--Szenes chamber series.

Everything here works on the exact truncated Taylor expansion

    F_d(x_1, ..., x_{d-1})  =  sum_beta A_beta x^beta
                            =  expanded_num / prod_r (1 - f_r),

computed by :func:`chernpp.polynomial.expand_rational`.  Truncating at total degree
``D`` is exact for every coefficient of degree ``<= D``, because each ``f_r``
has zero constant term.

Two conjectures live here (see ``papers/``):

* **Strong Laurent positivity** (Bérczi--Szenes): ``A_beta >= 0`` for all beta.
  True for ``d = 4``; false for ``d = 5``, the first counterexample being
  ``A_(1,1,2,1) = -1``.
* **Rimányi weak Chern positivity**: the *Chern* coefficients, which are
  signed sums of the ``A_beta`` over ballot orderings, are nonnegative.  This
  is the real target; it is tested in :mod:`chernpp.chern`.

The reductions recorded in ``a5_weak_positivity_handoff.pdf`` sit in between,
and the helpers below evaluate them for arbitrary ``d``:

* the **unpaired tail** ``i > j  ==>  A_{i,j,...} >= 0`` (Proposition 3, proved
  for ``d = 5``);
* the **paired inequality** ``A_{i,j,...} + A_{j-i,j,...} >= 0`` for
  ``0 <= i <= j``, which would imply the weak conjecture for ``d = 5``;
* **prefix positivity** ``F_d / (1 - a) >= 0`` (Section 8), proposed there as a
  stepping stone.
"""

from fractions import Fraction
from itertools import permutations
from typing import Dict, List, Optional, Sequence, Tuple

from .artifacts import ChamberAlgebra, load_algebra
from .logger import get_logger
from .polynomial import (
    Exponent,
    Poly,
    evaluate_variable,
    expand_rational,
    negative_terms,
    poly_scale,
)

logger = get_logger(__name__)


def chamber_series(
    dim: int,
    max_deg: int,
    extra_factors: Sequence[Poly] = (),
    algebra: Optional[ChamberAlgebra] = None,
) -> Poly:
    """
    The chamber series ``F_d`` truncated at total degree ``max_deg``.

    ``extra_factors`` adjoins further denominator factors: passing the single
    monomial ``a`` yields the prefix series ``F_d / (1 - a)``.
    """
    alg = algebra or load_algebra(dim)
    return expand_rational(
        alg.numerator, list(alg.denominator_factors) + list(extra_factors), max_deg
    )


def monomial(nvars: int, index: int) -> Poly:
    """The single chamber variable at position ``index``, as a polynomial."""
    return {tuple(1 if i == index else 0 for i in range(nvars)): 1}


def sorted_negatives(series: Poly) -> List[Tuple[Exponent, int]]:
    """Negative coefficients, lowest total degree first."""
    return sorted(negative_terms(series).items(), key=lambda kv: (sum(kv[0]), kv[0]))


def tau(beta: Exponent) -> Optional[Exponent]:
    """
    The first adjacent transposition ``tau(i, j, ...) = (j - i, j, ...)``.

    This is the involution induced by swapping the first two increments of the
    underlying ballot sequence; it fixes every partial sum after the second.
    Returns ``None`` when ``j - i < 0``, i.e. when the transposed ordering is
    no longer a ballot sequence.
    """
    i, j = beta[0], beta[1]
    if j - i < 0:
        return None
    return (j - i,) + beta[1:]


def unpaired_tail_defects(series: Poly, max_deg: int) -> List[Tuple[Exponent, int]]:
    """
    Coefficients violating ``i > j ==> A_beta >= 0``.

    Proposition 3 of the handoff note proves there are none for ``d = 5``.
    """
    out = []
    for beta, c in series.items():
        if sum(beta) <= max_deg and beta[0] > beta[1] and c < 0:
            out.append((beta, c))
    return sorted(out, key=lambda kv: (sum(kv[0]), kv[0]))


def paired_defects(series: Poly, max_deg: int) -> List[Tuple[Exponent, int]]:
    """
    Coefficients violating ``A_beta + A_{tau(beta)} >= 0`` for ``0 <= i <= j``.

    Only pairs whose *both* members lie inside the truncated region are
    tested, so the verdict is exact for everything reported.  Returns the
    offending ``(beta, A_beta + A_tau(beta))``.
    """
    out = []
    for beta in series:
        i, j = beta[0], beta[1]
        if not (0 <= i <= j):
            continue
        partner = tau(beta)
        if partner is None or sum(beta) > max_deg or sum(partner) > max_deg:
            continue
        total = series.get(beta, 0) + series.get(partner, 0)
        if total < 0:
            out.append((beta, total))
    return sorted(out, key=lambda kv: (sum(kv[0]), kv[0]))


def tail_target(
    dim: int, algebra: Optional[ChamberAlgebra] = None
) -> Tuple[Poly, List[Poly], Tuple[str, ...]]:
    """
    The series ``J_d(1/2, ...)`` that controls the unpaired tail ``i > j``.

    Set ``H(a) = (1-a)/(1-2a)`` and ``J = F_d / H``, so that ``F_d = H * J``.
    After the factor ``H`` is removed, every occurrence of ``a`` in a numerator
    or denominator monomial is accompanied by ``b`` -- the chamber monomials
    containing ``a`` are ``z_1/z_l = a b ... ``, and only ``z_1/z_2 = a`` is
    bare, which is exactly the factor ``1 - 2a`` that ``H`` removes.  Hence

        [b^j c^k ...] J  has a-degree at most j                    (Lemma 2)

    and for ``i > j`` convolution with ``H(a) = 1 + sum_n 2^{n-1} a^n`` gives

        A_{i,j,k,...}  =  2^{i-1} [b^j c^k ...] J(1/2, b, c, ...).

    So the whole unpaired tail is nonnegative as soon as this one series in
    ``d - 2`` variables is.  For ``d = 5`` that series factors into six
    manifestly nonnegative ratios, which is Proposition 3 of the handoff note;
    for ``d = 6`` it is the natural target for a machine proof.

    Returns ``(numerator, denominator_factors, varnames)`` in the remaining
    variables, with exact ``Fraction`` coefficients.
    """
    alg = algebra or load_algebra(dim)
    nvars = alg.nvars

    bare_a = {tuple(1 if i == 0 else 0 for i in range(nvars)): 2}
    remaining = [f for f in alg.denominator_factors if f != bare_a]
    if len(remaining) != len(alg.denominator_factors) - 1:
        raise RuntimeError(
            f"A_{dim}: expected exactly one denominator factor equal to 2a, "
            f"found {len(alg.denominator_factors) - len(remaining)}"
        )

    half = Fraction(1, 2)
    # J = N / ((1 - a) * prod_{r != 2a} (1 - f_r)); at a = 1/2 the (1 - a)
    # divides out as a factor of 2.
    numerator = poly_scale(evaluate_variable(alg.numerator, 0, half), 2)
    factors = [evaluate_variable(f, 0, half) for f in remaining]
    return numerator, factors, alg.chamber_vars[1:]


def vandermonde_factors(dim: int) -> List[Poly]:
    """
    The ``u`` for which the numerator's Vandermonde part is ``prod (1 - u)``.

    These are the ``z_m/z_l = x_m x_{m+1} ... x_{l-1}`` for ``1 <= m < l <= d``,
    so there are ``binomial(d, 2)`` of them.  Having the numerator in factored
    form is what makes the multiplicative certificates of :mod:`chernpp.lemma1`
    possible: the factors are exactly what gets paired against denominators.
    """
    nvars = dim - 1
    out = []
    for l in range(2, dim + 1):
        for m in range(1, l):
            exponents = [0] * nvars
            for k in range(m, l):
                exponents[k - 1] = 1
            out.append({tuple(exponents): 1})
    return out


def tail_target_factored(
    dim: int, algebra: Optional[ChamberAlgebra] = None
) -> Tuple[List[Poly], Poly, List[Poly], Tuple[str, ...]]:
    """
    :func:`tail_target` with the numerator kept in factored form.

    ``J_d = (V / (1 - a)) * P_full / prod_{r != 2a} (1 - f_r)``: the ``(1 - a)``
    introduced by ``H`` cancels the Vandermonde factor ``1 - z_1/z_2 = 1 - a``
    exactly, so the numerator is the remaining Vandermonde factors times the
    normalised numerator, all evaluated at ``a = 1/2``.

    Returns ``(numerator_factors, residual, denominator_factors, varnames)``.
    """
    alg = algebra or load_algebra(dim)
    nvars = alg.nvars
    half = Fraction(1, 2)

    bare_a = {tuple(1 if i == 0 else 0 for i in range(nvars)): 1}
    factors = [
        evaluate_variable(u, 0, half) for u in vandermonde_factors(dim) if u != bare_a
    ]
    residual = evaluate_variable(alg.normalized_numerator, 0, half)

    doubled_a = {tuple(1 if i == 0 else 0 for i in range(nvars)): 2}
    denominators = [
        evaluate_variable(f, 0, half) for f in alg.denominator_factors if f != doubled_a
    ]
    return factors, residual, denominators, alg.chamber_vars[1:]


def ballot_orderings(multiset: Sequence[int]) -> List[Tuple[Exponent, Exponent]]:
    """
    The orderings of a zero-sum multiset that index one Chern coefficient.

    Returns distinct pairs ``(alpha, beta)`` where ``alpha`` is an ordering of
    ``multiset`` all of whose proper partial sums are nonnegative, and
    ``beta_j = alpha_1 + ... + alpha_j`` for ``j = 1, ..., d-1``.  By Lemma 2
    of ``report.pdf`` this is a bijection onto the exponents contributing to
    the Chern monomial indexed by ``multiset``.
    """
    if sum(multiset) != 0:
        raise ValueError("multiset must sum to zero")
    out = []
    for alpha in sorted(set(permutations(multiset))):
        betas, s = [], 0
        for a in alpha[:-1]:
            s += a
            if s < 0:
                break
            betas.append(s)
        else:
            out.append((alpha, tuple(betas)))
    return out


def chern_coefficient(series: Poly, multiset: Sequence[int], max_deg: int) -> int:
    """
    The l-free Chern coefficient ``C(M) = sum over ballot orderings of A_beta``.

    Theorem 5 of ``report.pdf``: for every relative dimension ``l`` and every
    Chern monomial ``prod_i c_{p_i}``, the coefficient of that monomial in
    ``Tp^l_{A_d}`` equals ``C(M)`` with ``M = {p_i - (l+1)}``.  In particular
    it does not depend on ``l`` except through ``M``, so Rimányi's conjecture
    "for all l" is the single statement ``C(M) >= 0`` for all zero-sum ``M``.

    ``max_deg`` is the truncation degree of ``series``; a missing coefficient
    inside that range is genuinely zero, one outside it is unknown, so this
    raises rather than silently undercounting.
    """
    total = 0
    for _, beta in ballot_orderings(multiset):
        if sum(beta) > max_deg:
            raise ValueError(
                f"exponent {beta} has total degree {sum(beta)} > truncation {max_deg}; "
                "recompute the series to a higher degree"
            )
        total += series.get(beta, 0)
    return total


def prefix_report(dim: int, max_deg: int) -> Dict[str, object]:
    """
    Compare ``F_d`` with its prefix series ``F_d / (1 - a)``.

    Section 8 of the handoff note observes that every tested coefficient of
    ``F_5 / (1 - a)`` is nonnegative and proposes proving that as a route to
    the weak conjecture.  The obstruction to the route at a given ``d`` is
    visible immediately: the prefix sum runs over ``r <= i``, so any negative
    coefficient with ``i = 0`` survives it untouched.
    """
    alg = load_algebra(dim)
    nvars = len(alg.chamber_vars)
    base = chamber_series(dim, max_deg, algebra=alg)
    pref = chamber_series(dim, max_deg, extra_factors=[monomial(nvars, 0)], algebra=alg)

    base_neg = sorted_negatives(base)
    return {
        "dim": dim,
        "max_deg": max_deg,
        "chamber_vars": alg.chamber_vars,
        "base_negatives": base_neg,
        "prefix_negatives": sorted_negatives(pref),
        "base_negatives_at_i0": [(b, c) for b, c in base_neg if b[0] == 0],
    }
