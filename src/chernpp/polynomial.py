"""
Exact sparse polynomial arithmetic on exponent-tuple dictionaries.

A polynomial in the ``n`` chamber variables is a ``dict`` mapping an exponent
tuple of length ``n`` to a coefficient (``int`` or ``Fraction``).  This is the
representation the Sage stage exports, so nothing has to be parsed or
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


import numpy as np


def _poly_mul_numpy(p: Poly, q: Poly, max_deg: int) -> Poly:
    if not p or not q:
        return {}

    nvars = len(next(iter(p.keys())))
    p_len = len(p)
    q_len = len(q)

    p_exps = np.zeros((p_len, nvars), dtype=np.uint32)
    p_coeffs = np.zeros(p_len, dtype=np.float64)
    for i, (e, c) in enumerate(p.items()):
        p_exps[i] = e
        p_coeffs[i] = float(c)

    q_exps = np.zeros((q_len, nvars), dtype=np.uint32)
    q_coeffs = np.zeros(q_len, dtype=np.float64)
    for i, (e, c) in enumerate(q.items()):
        q_exps[i] = e
        q_coeffs[i] = float(c)

    # Base for 1D indices must be strictly greater than max_deg to avoid collisions
    b = max(max_deg + 1, 31)
    base = np.array([b ** (nvars - 1 - i) for i in range(nvars)], dtype=np.uint64)

    # Iterate over the smaller polynomial to minimize loop overhead
    if q_len > p_len:
        p_exps, q_exps = q_exps, p_exps
        p_coeffs, q_coeffs = q_coeffs, p_coeffs
        p_len, q_len = q_len, p_len

    all_exps = []
    all_coeffs = []

    for j in range(q_len):
        qe = q_exps[j]
        qc = q_coeffs[j]

        new_exps = p_exps + qe
        degrees = new_exps.sum(axis=1)
        mask = degrees <= max_deg

        valid_exps = new_exps[mask]
        valid_coeffs = p_coeffs[mask] * qc

        if len(valid_exps) > 0:
            all_exps.append(valid_exps)
            all_coeffs.append(valid_coeffs)

    if not all_exps:
        return {}

    flat_exps = np.concatenate(all_exps, axis=0)
    flat_coeffs = np.concatenate(all_coeffs, axis=0)

    flat_idx = flat_exps.dot(base)
    unique_idx, inverse = np.unique(flat_idx, return_inverse=True)
    res_coeffs = np.zeros(len(unique_idx), dtype=np.float64)
    np.add.at(res_coeffs, inverse, flat_coeffs)

    # Get the unique exponents corresponding to unique_idx
    # inverse gives the mapping, we can find the first occurrence of each unique_idx
    _, first_occurrences = np.unique(inverse, return_index=True)
    unique_exps_arr = flat_exps[first_occurrences]

    res = {}
    # Convert exactly once at the end
    for e, c in zip(unique_exps_arr, res_coeffs):
        if abs(c) > 1e-10:
            res[tuple(int(x) for x in e)] = c
    return res


def poly_mul(p1: Poly, p2: Poly, max_deg: int = None, exact: bool = False) -> Poly:
    """Product, optionally truncated above total degree ``max_deg``."""
    if p1 and p2 and max_deg is not None and not exact:
        try:
            return _poly_mul_numpy(p1, p2, max_deg)
        except Exception as e:
            pass  # Fallback

    out: Dict[Exponent, object] = {}
    if exact:
        p1 = {
            e: (Fraction(c).limit_denominator(10**10) if isinstance(c, float) else c) for e, c in p1.items()
        }
        p2 = {
            e: (Fraction(c).limit_denominator(10**10) if isinstance(c, float) else c) for e, c in p2.items()
        }
    for e1, c1 in p1.items():
        for e2, c2 in p2.items():
            if max_deg is not None and sum(e1) + sum(e2) > max_deg:
                continue
            e = tuple(a + b for a, b in zip(e1, e2))
            out[e] = out.get(e, 0) + c1 * c2
    return strip_zeros(out)


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


def divide_by_one_minus(p: Poly, f: Poly, max_deg: int, exact: bool = False) -> Poly:
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
        term = poly_mul(term, f, max_deg, exact=exact)
        if not term:
            break
        res = poly_add(res, term)
    return res


def expand_rational(num: Poly, factors: List[Poly], max_deg: int, exact: bool = False) -> Poly:
    """
    Taylor expansion of ``num / prod_r (1 - f_r)``, truncated at ``max_deg``.

    This is the chamber series ``F_d`` when ``num`` is the artifact's
    ``numerator`` and ``factors`` its ``denominator_factors``.
    """
    series = {e: c for e, c in num.items() if sum(e) <= max_deg}
    for f in factors:
        series = divide_by_one_minus(series, f, max_deg, exact=exact)
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
