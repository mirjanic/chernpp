"""
Multiplicative (Lemma 1) certificates: the technique the papers actually use.

Both successful positivity proofs in the literature are *multiplicative*, not
additive.  Lemma 1 of ``rimanyi_positivity.pdf`` says

    (1 - u) / (1 - v)  is coefficientwise nonnegative whenever v - u >= 0,

because ``(1-f)/(1-f-g) = 1 + g/(1-f-g)``.  Theorem 1 (strong positivity at
``d = 4``) and Proposition 3 (the unpaired tail at ``d = 5``) are both proved by
pairing each numerator factor with a denominator factor so that every ratio is
nonnegative by this lemma, and multiplying.

That shape is *not* reachable by the additive certificates of
:mod:`chernpp.certificates`: a product of ratios is not a finite sum
``sum_S P_S prod_{r in S}(1 - f_r)`` with nonnegative ``P_S``.  Trying to certify
the A_5 tail additively fails even though the statement is a published theorem,
which is the diagnostic that motivates this module.

A certificate here is a proof.  Given a numerator presented in factored form,
we look for a matching of numerator factors to denominator factors satisfying
the Lemma 1 condition, cancel whatever divides exactly, and require the
remaining numerator to be coefficientwise nonnegative.  Everything is exact.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .logger import get_logger
from .polynomial import (
    poly_one,
    poly_add,
    Poly,
    is_nonneg,
    negative_terms,
    one_minus,
    poly_mul,
    poly_mul_many,
    poly_sub,
    poly_to_string,
    total_degree,
)

logger = get_logger(__name__)


def dominates(v: Poly, u: Poly) -> bool:
    """Whether ``v - u`` is coefficientwise nonnegative, i.e. Lemma 1 applies."""
    return is_nonneg(poly_sub(v, u))


def positive_part(p: Poly) -> Poly:
    return {e: c for e, c in p.items() if c > 0}


def negative_part(p: Poly) -> Poly:
    """The negated negative coefficients, so ``p = positive_part - negative_part``."""
    return {e: -c for e, c in p.items() if c < 0}


def absorbs(numerator: Poly, denominators: Sequence[Poly], nvars: int) -> bool:
    """
    Absorption criterion -- a strict generalisation of Lemma 1.

    Write ``N = N_+ - N_-`` with both parts nonnegative, and let ``S`` be a set
    of denominators with ``1 - W = prod_{v in S} (1 - v)``.  If

        N_-  <=  N_+ * W        (coefficientwise)

    then ``N - N_+ prod_S (1 - v) = N_+ W - N_- >= 0``, so ``N = N_+ prod_S(1-v) + P``
    with ``P >= 0`` and

        N / prod_all (1 - v)  =  N_+ / prod_{not S} (1 - v)  +  P / prod_all (1 - v),

    a sum of nonnegative series.  Lemma 1 is the case ``N = 1 - u``, ``S = {v}``,
    where the condition reads ``u <= v``.

    Unlike Lemma 1 this does not need the numerator in factored form, and it
    lets several denominators cooperate on one numerator.
    """
    W = poly_sub(
        {(0,) * nvars: 1},
        poly_mul_many([one_minus(v, nvars) for v in denominators], nvars),
    )
    return is_nonneg(poly_sub(poly_mul(positive_part(numerator), W), negative_part(numerator)))


def pairs_generalised(u: Poly, v: Poly, nvars: int) -> bool:
    """
    ``(1 - u)/(1 - v) >= 0``, by a criterion strictly weaker than Lemma 1.

    Since ``1 - u = (1 - v) + (v - u)``,

        (1 - u)/(1 - v)  =  1 + (v - u)/(1 - v),

    so the ratio is nonnegative as soon as ``(v - u)/(1 - v)`` is -- and that is
    a question :func:`absorbs` can answer, rather than requiring ``v - u >= 0``
    outright.  Lemma 1 is the case where ``v - u`` has no negative part at all.

    Note that scalar ratios ``(1 - a v)/(1 - b v)`` give nothing beyond Lemma 1:
    expanding,

        (1 - a v)/(1 - b v)  =  1 + sum_{k >= 1} b^{k-1} (b - a) v^k,

    which is coefficientwise nonnegative exactly when ``a <= b``, i.e. exactly
    when ``bv - av >= 0``.  The freedom is in the *shape* of ``u`` and ``v``,
    not in rescaling them.
    """
    return absorbs(poly_sub(v, u), [v], nvars)


def unroll(numerator: Poly, v: Poly, times: int, nvars: int):
    """
    Rewrite ``N/(1 - v)`` as ``N (1 + v + ... + v^{m-1}) / (1 - v^m)``.

    Both sides are equal because ``1 - v^m = (1 - v)(1 + ... + v^{m-1})``, and
    ``v^m`` is again nonnegative with zero constant term, so it is a legitimate
    denominator.  Unrolling spreads the numerator's positive mass over higher
    degrees, which is what lets absorption reach negative coefficients that sit
    too deep for the un-unrolled form.

    Returns ``(new_numerator, new_denominator)``.
    """
    if times < 1:
        raise ValueError("times must be at least 1")
    spread = poly_one(nvars)
    power = poly_one(nvars)
    for _ in range(times - 1):
        power = poly_mul(power, v)
        spread = poly_add(spread, power)
    return poly_mul(numerator, spread), poly_mul(power, v)


# A note on scaling the denominators, since the idea is tempting and wrong.
#
# One would like to enlarge the search by absorbing against `lambda * v` rather
# than `v`, turning a finite set of subsets into a continuum.  It does not work.
# To replace the factor `1 - v` by `1 - lambda v` one needs
# `(1 - lambda v)/(1 - v)` to be coefficientwise nonnegative, which by Lemma 1
# requires `v - lambda v = (1 - lambda) v >= 0`, that is `lambda <= 1`.  And for
# `lambda <= 1` the absorption condition `N_- <= N_+ * lambda v` is strictly
# harder than at `lambda = 1`, so scaling can only ever weaken the criterion.
#
# Taking `lambda > 1` is unsound, not merely useless: with `N = 1 - 3x` and
# `v = x` the condition `N_- <= N_+ * 3x` holds, yet
# `N/(1 - x) = 1 - 2x - 2x^2 - ...` is visibly negative.


def absorbing_subset(
    numerator: Poly,
    denominators: Sequence[Poly],
    nvars: int,
    max_size: Optional[int] = None,
) -> Optional[Tuple[int, ...]]:
    """
    Smallest subset of ``denominators`` that absorbs ``numerator``, if any.

    Returns the indices, or ``None``.  A hit is a proof that
    ``numerator / prod (1 - v)`` is coefficientwise nonnegative.
    """
    from itertools import combinations

    if is_nonneg(numerator):
        return ()
    top = len(denominators) if max_size is None else min(max_size, len(denominators))
    for size in range(1, top + 1):
        for subset in combinations(range(len(denominators)), size):
            if absorbs(numerator, [denominators[i] for i in subset], nvars):
                return subset
    return None


def divide_exactly(p: Poly, v: Poly, nvars: int) -> Optional[Poly]:
    """
    ``p / (1 - v)`` when the division is exact, else ``None``.

    Since ``v`` has zero constant term, ``p * sum_k v^k`` truncated at
    ``deg p`` equals the quotient whenever one exists; the result is then
    verified by multiplying back, so a false positive is impossible.
    """
    if not p:
        return {}
    cap = total_degree(p)
    quotient, term = dict(p), dict(p)
    for _ in range(cap + 1):
        term = poly_mul(term, v, cap)
        if not term:
            break
        quotient = poly_add(quotient, term)
    else:
        return None  # series did not terminate within the degree bound
    return quotient if poly_mul(quotient, one_minus(v, nvars)) == p else None


def _match(numerator_us: Sequence[Poly], denominator_vs: Sequence[Poly]) -> Dict[int, int]:
    """
    Maximum bipartite matching of numerator factors to denominator factors.

    Edge ``i -- j`` exists exactly when ``(1 - u_i)/(1 - v_j)`` is nonnegative
    by Lemma 1.  Plain augmenting paths: the graphs here have tens of nodes.
    """
    allowed = [[j for j, v in enumerate(denominator_vs) if dominates(v, u)] for u in numerator_us]
    taken_by: Dict[int, int] = {}

    def augment(i: int, seen: set) -> bool:
        for j in allowed[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in taken_by or augment(taken_by[j], seen):
                taken_by[j] = i
                return True
        return False

    for i in range(len(numerator_us)):
        augment(i, set())
    return {i: j for j, i in taken_by.items()}


@dataclass
class Lemma1Certificate:
    """A verified multiplicative proof that a series is coefficientwise nonnegative."""

    pairs: List[Tuple[Poly, Poly]]
    leftover_numerator: Poly
    leftover_denominators: List[Poly]
    cancelled: List[Poly]
    nvars: int
    varnames: Tuple[str, ...] = ()
    notes: List[str] = field(default_factory=list)

    @property
    def proved(self) -> bool:
        return is_nonneg(self.leftover_numerator)

    def summary(self) -> str:
        names = self.varnames or tuple(f"x{i}" for i in range(self.nvars))
        lines = [
            f"Lemma-1 certificate: {len(self.pairs)} paired ratios, "
            f"{len(self.cancelled)} denominators cancelled exactly, "
            f"{len(self.leftover_denominators)} denominators left"
        ]
        for u, v in self.pairs[:8]:
            lines.append(f"   (1 - [{poly_to_string(u, names)}]) / (1 - [{poly_to_string(v, names)}])")
        if len(self.pairs) > 8:
            lines.append(f"   ... and {len(self.pairs) - 8} more")
        lines.append("   leftover numerator: " + poly_to_string(self.leftover_numerator, names, limit=4))
        lines.append(f"   => {'PROVED' if self.proved else 'not conclusive'}")
        return "\n".join(lines)


@dataclass
class Backoff:
    """
    A partial multiplicative reduction: some Lemma-1 ratios, times a remainder.

    ``kept`` ratios are individually nonnegative by Lemma 1.  ``remainder`` is
    what is left over, as ``numerator / prod (1 - v)`` for ``denominators``.
    The reduction is only a *proof* once the remainder is itself proved
    nonnegative; ``nonneg_to`` records the degree to which it has been checked.
    """

    kept: List[Tuple[Poly, Poly]]
    returned: List[Tuple[Poly, Poly]]
    numerator: Poly
    denominators: List[Poly]
    nonneg_to: Optional[int]
    nvars: int
    varnames: Tuple[str, ...] = ()

    def summary(self) -> str:
        names = self.varnames or tuple(f"x{i}" for i in range(self.nvars))
        lines = [
            f"{len(self.kept)} Lemma-1 ratios kept, {len(self.returned)} returned " f"to the remainder",
            f"remainder: {len(self.numerator)} numerator terms over "
            f"{len(self.denominators)} denominators",
        ]
        if self.returned:
            lines.append("returned:")
            for u, v in self.returned:
                lines.append(f"   (1 - [{poly_to_string(u, names)}]) / (1 - [{poly_to_string(v, names)}])")
        lines.append(
            "remainder nonnegative to degree "
            + (str(self.nonneg_to) if self.nonneg_to is not None else "-- (has negatives)")
        )
        return "\n".join(lines)


def search_with_backoff(
    numerator_factors: Sequence[Poly],
    residual: Poly,
    denominator_factors: Sequence[Poly],
    nvars: int,
    varnames: Tuple[str, ...] = (),
    max_returned: int = 2,
    probe_degree: int = 11,
) -> Optional[Backoff]:
    """
    Peel off Lemma-1 ratios, giving some back if that leaves a better remainder.

    A maximum matching is greedy in the wrong way: it can consume a denominator
    that the remainder needs.  So after matching we try *returning* small
    subsets of ratios to the remainder -- putting ``(1 - u)`` back in its
    numerator and ``(1 - v)`` back in its denominators -- and keep the first
    choice whose remainder is coefficientwise nonnegative as far as
    ``probe_degree``.

    Returns ``None`` if no subset of size ``<= max_returned`` works.  Note that
    a nonnegative remainder here is *checked*, not proved: it still has to be
    certified by other means before the whole thing is a theorem.
    """
    from itertools import combinations

    from .polynomial import expand_rational

    pairing = _match(numerator_factors, denominator_factors)
    pairs = [(numerator_factors[i], denominator_factors[j]) for i, j in sorted(pairing.items())]
    used = set(pairing.values())
    unmatched = [v for j, v in enumerate(denominator_factors) if j not in used]

    for size in range(max_returned + 1):
        for combo in combinations(range(len(pairs)), size):
            numerator, denominators = residual, list(unmatched)
            for index in combo:
                u, v = pairs[index]
                numerator = poly_mul(numerator, one_minus(u, nvars))
                denominators = denominators + [v]
            series = expand_rational(numerator, denominators, probe_degree)
            if not negative_terms(series):
                return Backoff(
                    kept=[p for i, p in enumerate(pairs) if i not in combo],
                    returned=[pairs[i] for i in combo],
                    numerator=numerator,
                    denominators=denominators,
                    nonneg_to=probe_degree,
                    nvars=nvars,
                    varnames=tuple(varnames),
                )
    return None


def search(
    numerator_factors: Sequence[Poly],
    residual: Poly,
    denominator_factors: Sequence[Poly],
    nvars: int,
    varnames: Tuple[str, ...] = (),
) -> Lemma1Certificate:
    """
    Look for a multiplicative proof that

        residual * prod_i (1 - numerator_factors[i]) / prod_j (1 - denominator_factors[j])

    is coefficientwise nonnegative.

    Each numerator factor is matched to a denominator factor it dominates, so
    that ratio is nonnegative by Lemma 1.  Unmatched denominators are then
    cancelled against the remaining numerator wherever the division is exact --
    every ``1/(1 - f)`` that survives is itself a nonnegative series.  The
    certificate is conclusive exactly when what is left of the numerator is
    coefficientwise nonnegative.
    """
    pairing = _match(numerator_factors, denominator_factors)
    pairs = [(numerator_factors[i], denominator_factors[j]) for i, j in pairing.items()]

    unmatched_num = [u for i, u in enumerate(numerator_factors) if i not in pairing]
    used = set(pairing.values())
    unmatched_den = [v for j, v in enumerate(denominator_factors) if j not in used]

    leftover = poly_mul(residual, poly_mul_many([one_minus(u, nvars) for u in unmatched_num], nvars))

    cancelled, remaining_den = [], []
    for v in unmatched_den:
        quotient = divide_exactly(leftover, v, nvars)
        if quotient is None:
            remaining_den.append(v)
        else:
            leftover = quotient
            cancelled.append(v)

    return Lemma1Certificate(
        pairs=pairs,
        leftover_numerator=leftover,
        leftover_denominators=remaining_den,
        cancelled=cancelled,
        nvars=nvars,
        varnames=tuple(varnames),
    )
