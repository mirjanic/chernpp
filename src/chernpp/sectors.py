"""
Closed forms on sectors of the Chern table, checked against the box engine.

*Ballot counts.*  ``b(M)`` is the number of distinct orderings of ``M`` whose
proper partial sums are nonnegative -- the packet sum of the pure chain series
``prod_j 1/(1 - x_j)``, all of whose coefficients are 1.

*Plane sector* (``max M <= 1``).  Here ``C(M) = b(M)``.  The proof has three
steps, and only the first is not ours:

1. The external findings summary (Theorem 5.4, unrefereed) identifies ``C(M)``
   with the number of minimal *monotone* factorisations of an ``n``-cycle,
   ``n = d + 1``, into transpositions ``(a b)``, ``a < b``, ``b`` nondecreasing,
   whose exponent vector ``e_b = #{transpositions with larger point b}`` is a
   rearrangement of ``1 - M`` (``e_1 = 0``).
2. For the cycle ``x -> x + 1`` every exponent vector satisfying the ballot
   condition ``sum_{k <= j} e_k <= j - 1`` has exactly one such factorisation,
   and no other vector has any.  Necessity: the transpositions with ``b <= j``
   form a forest on ``[j]``.  Existence and uniqueness: peel off the
   transpositions through ``n``; writing them ``(a_1 n) ... (a_m n)``, the rest
   is ``F = sigma C^{-1}`` with ``F(x) = x + 1`` away from the ``a_i``, and each
   cycle of ``F`` must contain exactly one ``a_i``.  That forces
   ``a_1 = n - 1 > a_2 > ... > a_m``, so ``F`` is a product of increasing cycles
   on consecutive intervals.  The cut points are forced to be the last visits of
   the deficit walk ``D(j) = j - 1 - sum_{k <= j} e_k`` to the levels
   ``0, 1, ..., m - 1``.  Induct on the intervals.
3. Reading ``alpha = 1 - e`` turns ballot exponent vectors into ballot
   orderings of ``M``.

By the cycle lemma, ``b(M)`` on this sector is also Kreweras' count of
noncrossing partitions of ``[d]`` with block sizes ``s_i = 1 - a`` (``a <= 0``):
``d! / ((d - b + 1)! prod_i m_i!)``.  Step 2 is checked by brute force in
:func:`monotone_factorisations`.

*The ballot conjecture.*  ``C(M) >= b(M)`` for every zero-sum ``M``, with
equality exactly on the plane sector.  Since the decreasing ordering is always
ballot, ``b(M) >= 1``, so this implies Rimányi's conjecture and explains why the
least Chern coefficient is 1 and never 0.  It is verified on every exact table
in ``results/`` (:func:`ballot_violations`).

*Zero insertion.*  For a zero-free ``M0``, ``z -> C(M0 + 0^z)`` (append ``z`` zeros,
i.e. ``z`` singleton blocks) appears to be a polynomial whose Newton coefficients
``Delta^k C(M0 + 0^z)|_{z=0}`` are all nonnegative -- and so are those of
``C - b``.  The second statement refines the ballot conjecture: ``C(M) - b(M)`` is
then a nonnegative combination of binomials in the number of zeros of ``M``.
Verified on every base available (:func:`newton_coefficients`).

*Stabilisation.*  ``F_d |_{x_{d-1} = 0} = F_{d-1}``: the top chamber variable
switched off recovers the previous series.  Checked cell by cell on boxes.
"""

from collections import Counter
from functools import lru_cache
from math import factorial
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import boxes


def is_plane(multiset: Sequence[int]) -> bool:
    return max(multiset) <= 1


def ballot_count(multiset: Sequence[int]) -> int:
    """``b(M)``: distinct orderings of ``multiset`` with nonnegative proper partial sums."""
    values = sorted(set(multiset))
    counts = Counter(multiset)
    n = len(multiset)

    @lru_cache(maxsize=None)
    def walk(remaining: Tuple[int, ...], total: int, placed: int) -> int:
        if placed == n:
            return 1
        out = 0
        for i, c in enumerate(remaining):
            if not c:
                continue
            step = total + values[i]
            if placed < n - 1 and step < 0:
                continue
            rest = remaining[:i] + (c - 1,) + remaining[i + 1 :]
            out += walk(rest, step, placed + 1)
        return out

    return walk(tuple(counts[v] for v in values), 0, 0)


def block_sizes(multiset: Sequence[int]) -> Tuple[int, ...]:
    """The block sizes ``s_i = 1 - a`` of a plane-sector multiset, decreasing."""
    if not is_plane(multiset) or sum(multiset) != 0:
        raise ValueError(f"{tuple(multiset)} is not a zero-sum plane-sector multiset")
    return tuple(sorted((1 - a for a in multiset if a <= 0), reverse=True))


def kreweras(multiset: Sequence[int]) -> int:
    """Noncrossing partitions of ``[d]`` with the block sizes of ``multiset``."""
    d, sizes = len(multiset), block_sizes(multiset)
    b = len(sizes)
    den = factorial(d - b + 1)
    for m in Counter(sizes).values():
        den *= factorial(m)
    value, rem = divmod(factorial(d), den)
    assert rem == 0
    return value


def plane_sector_mismatches(table: Dict[Tuple[int, ...], int]):
    """Every plane-sector packet of ``table`` whose value differs from Kreweras."""
    return [(m, c, kreweras(m)) for m, c in table.items() if is_plane(m) and kreweras(m) != c]


def monotone_factorisations(n: int) -> Counter:
    """
    Minimal monotone factorisations of the long cycle ``x -> x + 1 (mod n)``,
    counted by exponent vector ``(e_1, ..., e_n)``.  Exhaustive; for small ``n``.
    """
    sigma = tuple((x + 1) % n for x in range(n))
    found: Counter = Counter()

    def rec(k: int, bmin: int, perm: Tuple[int, ...], e: List[int]):
        if k == n - 1:
            if perm == sigma:
                found[tuple(e)] += 1
            return
        for b in range(bmin, n):
            for a in range(b):
                nxt = list(perm)
                # perm * (a b): apply the transposition first
                nxt[a], nxt[b] = perm[b], perm[a]
                e[b] += 1
                rec(k + 1, b, tuple(nxt), e)
                e[b] -= 1

    rec(0, 1, tuple(range(n)), [0] * n)
    return found


def ballot_violations(table: Dict[Tuple[int, ...], int]):
    """
    Test the ballot conjecture on ``table``: returns ``(violations, plane_mismatches,
    nonplane_equalities)``, each a list of ``(M, C(M), b(M))``.  All three are empty
    wherever the conjecture holds with its equality case.
    """
    violations, plane, equal = [], [], []
    for m, c in table.items():
        b = ballot_count(m)
        if c < b:
            violations.append((m, c, b))
        if is_plane(m) and c != b:
            plane.append((m, c, b))
        if not is_plane(m) and c == b:
            equal.append((m, c, b))
    return violations, plane, equal


def newton_coefficients(values: Sequence[int]) -> List[int]:
    """Forward differences at 0: ``values[z] = sum_k out[k] * binom(z, k)``."""
    out, cur = [], list(values)
    while cur:
        out.append(cur[0])
        cur = [cur[i + 1] - cur[i] for i in range(len(cur) - 1)]
    return out


def zero_insertion_series(table_by_d: Dict[int, Dict[Tuple[int, ...], int]], base: Sequence[int]):
    """``[(C, b)]`` for ``base + 0^z``, ``z = 0, 1, ...`` while the tables contain it."""
    base = tuple(a for a in base if a != 0)
    out = []
    z = 0
    while True:
        m = tuple(sorted(base + (0,) * z, reverse=True))
        c = table_by_d.get(len(m), {}).get(m)
        if c is None:
            return out
        out.append((c, ballot_count(m)))
        z += 1


def stabilisation_holds(d: int, bounds: Sequence[int]) -> bool:
    """``F_d`` with ``x_{d-1} = 0`` against ``F_{d-1}`` on the box ``bounds`` (length ``d - 2``)."""
    top = boxes.exact_box(boxes.Box(d, tuple(bounds) + (0,), f"A_{d} face"))
    low = boxes.exact_box(boxes.Box(d - 1, tuple(bounds), f"A_{d - 1}"))
    return bool(np.array_equal(top[..., 0], low))
