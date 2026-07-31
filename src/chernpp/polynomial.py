"""
Exact sparse polynomial arithmetic on exponent-tuple dictionaries.

A polynomial in the ``n`` chamber variables is a ``dict`` mapping an exponent
tuple of length ``n`` to a coefficient (``int`` or ``Fraction``).  This is the
representation the Sage miner exports, so nothing has to be parsed or
re-expanded downstream.  Every routine here is exact; no floating point is
used anywhere in this module.

Truncation is by *total* degree.  Because every denominator factor ``f_r`` has
zero constant term, truncating at degree ``D`` throughout is safe: no term of
degree ``<= D`` in the final answer can depend on a discarded term.
"""

from collections import defaultdict
from itertools import combinations_with_replacement
from typing import Dict, Iterable, List, Tuple

Exponent = Tuple[int, ...]
Poly = Dict[Exponent, object]


def poly_one(nvars: int) -> Poly:
    return {(0,) * nvars: 1}


def strip_zeros(p: Poly) -> Poly:
    return {e: c for e, c in p.items() if c != 0}


def poly_add(*ps: Poly) -> Poly:
    res: Dict[Exponent, object] = defaultdict(int)
    for p in ps:
        for e, c in p.items():
            res[e] += c
    return strip_zeros(res)


def poly_sub(p: Poly, q: Poly) -> Poly:
    res: Dict[Exponent, object] = defaultdict(int, p)
    for e, c in q.items():
        res[e] -= c
    return strip_zeros(res)


def poly_scale(p: Poly, c) -> Poly:
    if c == 0:
        return {}
    return {e: v * c for e, v in p.items()}


def poly_mul(p: Poly, q: Poly, max_deg: int = None) -> Poly:
    """Product, optionally truncated above total degree ``max_deg``."""
    res: Dict[Exponent, object] = defaultdict(int)
    for e1, c1 in p.items():
        d1 = sum(e1)
        for e2, c2 in q.items():
            if max_deg is not None and d1 + sum(e2) > max_deg:
                continue
            res[tuple(a + b for a, b in zip(e1, e2))] += c1 * c2
    return strip_zeros(res)


def poly_mul_many(ps: Iterable[Poly], nvars: int, max_deg: int = None) -> Poly:
    res = poly_one(nvars)
    for p in ps:
        res = poly_mul(res, p, max_deg)
    return res


def total_degree(p: Poly) -> int:
    return max((sum(e) for e in p), default=0)


def negative_terms(p: Poly) -> Poly:
    return {e: c for e, c in p.items() if c < 0}


def is_nonneg(p: Poly) -> bool:
    return all(c >= 0 for c in p.values())


def monomials_up_to(nvars: int, max_deg: int) -> List[Exponent]:
    """All exponent tuples of total degree <= ``max_deg``, in degree order."""
    out = []
    for d in range(max_deg + 1):
        # compositions of d into nvars nonnegative parts
        for cut in combinations_with_replacement(range(nvars), d):
            e = [0] * nvars
            for i in cut:
                e[i] += 1
            out.append(tuple(e))
    return sorted(set(out), key=lambda e: (sum(e), e))


def evaluate_variable(p: Poly, index: int, value) -> Poly:
    """
    Substitute a constant for one variable, dropping that coordinate.

    ``value`` may be a ``Fraction``, so this stays exact: evaluating at
    ``a = 1/2`` is the substitution the unpaired-tail argument turns on.
    """
    result: Dict[Exponent, object] = defaultdict(int)
    for exponents, coefficient in p.items():
        power = exponents[index]
        rest = exponents[:index] + exponents[index + 1 :]
        result[rest] += coefficient * (value**power)
    return strip_zeros(result)


def one_minus(f: Poly, nvars: int) -> Poly:
    """The denominator factor ``1 - f``."""
    return poly_sub(poly_one(nvars), f)


def divide_by_one_minus(p: Poly, f: Poly, max_deg: int) -> Poly:
    """
    ``p / (1 - f)`` truncated at total degree ``max_deg``.

    Requires ``f`` to have zero constant term, so that ``p * sum_k f^k``
    terminates: ``f^k`` has total degree at least ``k``.
    """
    if any(sum(e) == 0 for e in f):
        raise ValueError("f must have zero constant term")
    res = {e: c for e, c in p.items() if sum(e) <= max_deg}
    term = res
    for _ in range(max_deg):
        term = poly_mul(term, f, max_deg)
        if not term:
            break
        res = poly_add(res, term)
    return res


def expand_rational(num: Poly, factors: List[Poly], max_deg: int) -> Poly:
    """
    Taylor expansion of ``num / prod_r (1 - f_r)``, truncated at ``max_deg``.

    This is the chamber series ``F_d`` when ``num`` is the miner's
    ``numerator`` and ``factors`` its ``denominator_factors``.
    """
    series = {e: c for e, c in num.items() if sum(e) <= max_deg}
    for f in factors:
        series = divide_by_one_minus(series, f, max_deg)
    return series


def poly_to_string(p: Poly, varnames: Tuple[str, ...], limit: int = None) -> str:
    """Human-readable rendering, lowest total degree first."""
    if not p:
        return "0"
    items = sorted(p.items(), key=lambda kv: (sum(kv[0]), kv[0]))
    if limit is not None and len(items) > limit:
        items, extra = items[:limit], len(items) - limit
    else:
        extra = 0
    parts = []
    for e, c in items:
        mono = "*".join(v if k == 1 else f"{v}^{k}" for v, k in zip(varnames, e) if k)
        if not mono:
            parts.append(str(c))
        elif c == 1:
            parts.append(mono)
        elif c == -1:
            parts.append(f"-{mono}")
        else:
            parts.append(f"{c}*{mono}")
    out = " + ".join(parts).replace("+ -", "- ")
    return out + (f" + ... ({extra} more terms)" if extra else "")
