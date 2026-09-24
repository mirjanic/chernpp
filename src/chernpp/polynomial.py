"""
Exact sparse polynomial arithmetic on exponent-tuple dictionaries.

A polynomial in the ``n`` chamber variables is a ``dict`` mapping an exponent
tuple of length ``n`` to a coefficient (``int`` or ``Fraction``).  This is the
representation the Sage stage exports, so nothing has to be parsed or
re-expanded downstream.  Every routine here is exact.  Floating point is
used only when a caller opts in with ``exact=False`` and passes float
coefficients (see :func:`poly_mul`); on the default path a float raises.

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
from fractions import Fraction
from numbers import Integral

#: The int64 path refuses products whose worst-case accumulated magnitude could
#: reach this; they go to exact Python integers instead.
_INT64_SAFE = 2**62


def _is_integral(p: Poly) -> bool:
    return all(isinstance(c, Integral) and not isinstance(c, bool) for c in p.values())


def _has_float(p: Poly) -> bool:
    return any(isinstance(c, (float, np.floating)) for c in p.values())


def _poly_mul_vectorised(p: Poly, q: Poly, max_deg: int, dtype) -> Poly:
    """
    Truncated product by numpy, in ``int64`` (exact) or ``float64`` (opt-in only).

    The integer path is exact: callers guarantee, through :func:`poly_mul`, that
    no partial sum can exceed ``2**62`` in magnitude, so ``int64`` never wraps.
    """
    if not p or not q:
        return {}
    nvars = len(next(iter(p.keys())))
    if len(q) > len(p):
        p, q = q, p
    p_exps = np.array(list(p.keys()), dtype=np.int64).reshape(len(p), nvars)
    p_coeffs = np.array([int(c) if dtype is np.int64 else float(c) for c in p.values()], dtype=dtype)
    q_items = list(q.items())

    base = max(max_deg + 1, 2)
    weights = np.array([base ** (nvars - 1 - i) for i in range(nvars)], dtype=np.int64)
    if base**nvars >= 2**62:
        raise OverflowError("exponent box too large to index in int64")

    all_idx, all_coeffs = [], []
    p_deg = p_exps.sum(axis=1)
    for qe, qc in q_items:
        mask = p_deg + sum(qe) <= max_deg
        if not mask.any():
            continue
        exps = p_exps[mask] + np.array(qe, dtype=np.int64)
        all_idx.append(exps @ weights)
        all_coeffs.append(p_coeffs[mask] * (int(qc) if dtype is np.int64 else float(qc)))
    if not all_idx:
        return {}
    idx = np.concatenate(all_idx)
    coeffs = np.concatenate(all_coeffs)
    unique_idx, inverse = np.unique(idx, return_inverse=True)
    sums = np.zeros(len(unique_idx), dtype=dtype)
    np.add.at(sums, inverse.ravel(), coeffs)

    out: Dict[Exponent, object] = {}
    for key, c in zip(unique_idx.tolist(), sums.tolist()):
        if c == 0:
            continue
        e, k = [], key
        for w in weights.tolist():
            e.append(k // w)
            k %= w
        out[tuple(e)] = int(c) if dtype is np.int64 else c
    return out


def poly_mul(p1: Poly, p2: Poly, max_deg: int = None, exact: bool = None) -> Poly:
    """
    Product, optionally truncated above total degree ``max_deg``.

    Exact by default.  Integer inputs take a vectorised ``int64`` path when an
    a-priori bound shows no partial sum can wrap, and exact Python integers
    otherwise; ``Fraction`` inputs are multiplied exactly in Python.

    Floating point is used only when the caller asks for it with
    ``exact=False`` *and* supplies float coefficients -- the gauge search does,
    and re-verifies everything it keeps in exact arithmetic.  A float reaching
    the default path raises rather than being silently rounded.  With
    ``exact=True`` float coefficients are rationalised explicitly.
    """
    if not p1 or not p2:
        return {}
    has_float = _has_float(p1) or _has_float(p2)
    if has_float and exact is None:
        raise TypeError(
            "poly_mul got floating-point coefficients on the exact path; pass exact=False to "
            "opt in to float arithmetic, or exact=True to rationalise them"
        )
    if exact:
        p1 = {
            e: (Fraction(c).limit_denominator(10**10) if isinstance(c, (float, np.floating)) else c)
            for e, c in p1.items()
        }
        p2 = {
            e: (Fraction(c).limit_denominator(10**10) if isinstance(c, (float, np.floating)) else c)
            for e, c in p2.items()
        }
        has_float = False
    if max_deg is not None:
        if has_float:
            return {
                e: c for e, c in _poly_mul_vectorised(p1, p2, max_deg, np.float64).items() if abs(c) > 1e-10
            }
        if _is_integral(p1) and _is_integral(p2):
            bound = (
                max(abs(c) for c in p1.values()) * max(abs(c) for c in p2.values()) * min(len(p1), len(p2))
            )
            if bound < _INT64_SAFE and max_deg + 1 < 2**20:
                try:
                    return _poly_mul_vectorised(p1, p2, max_deg, np.int64)
                except OverflowError:
                    pass  # exponent box too large to index; the Python path is exact
    out: Dict[Exponent, object] = {}
    for e1, c1 in p1.items():
        s1 = sum(e1)
        for e2, c2 in p2.items():
            if max_deg is not None and s1 + sum(e2) > max_deg:
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


def divide_by_one_minus(p: Poly, f: Poly, max_deg: int, exact: bool = None) -> Poly:
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


def expand_rational(num: Poly, factors: List[Poly], max_deg: int, exact: bool = None) -> Poly:
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
