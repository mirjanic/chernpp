"""
The chamber series on boxes, and the packets that live inside them.

Every ballot ordering ``beta`` of a zero-sum multiset ``M`` satisfies
``beta <= beta_max(M)`` coordinatewise, where ``beta_max`` is the sequence of
partial sums of the decreasing ordering: the first ``j`` entries sum to at most
the ``j`` largest.  Boxes, not total-degree truncations, are therefore the
natural domains for packet computations:

* the **level box** ``beta_j <= (d - j) L`` holds every packet with
  ``min M >= -L`` -- the Chern monomials of ``Tp^{L-1}``;
* the **charge box** ``beta_j <= j p`` holds every packet with ``max M <= p``;
* the **rectangle** ``beta_j <= min(j p, (d - j) L)`` holds exactly the packets
  with both bounds.

A packet is complete in a box ``b`` iff ``beta_max(M) <= b``.

The expansion is a *sweep*, not a fixed point.  Every monomial of a level-``l``
denominator factor contains ``x_{l-1}`` exactly once, so ``H = G + f H`` is a
recurrence along that one axis: the slice at index ``t`` depends only on the
slice at ``t - 1``.  One pass over the grid per factor replaces the
``sum(shape)`` passes of the fixed point in :mod:`chernpp.chern`, which stays as
the independent reference implementation.

Exactness follows the repository's rule: values are computed modulo word-sized
primes, reconstructed by CRT, and checked against a prime held back from the
reconstruction; a float64 shadow of the same sweep bounds every intermediate
magnitude, which fixes how many primes are needed.  Nothing is returned that
has not passed the check.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .artifacts import ChamberAlgebra, load_algebra
from .crt import DEFAULT_PRIMES
from .logger import get_logger

logger = get_logger(__name__)

Shape = Tuple[int, ...]


# --------------------------------------------------------------------------
# boxes and packets
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    """Inclusive per-coordinate bounds ``beta_j <= bounds[j]`` for the chamber series of ``A_d``."""

    d: int
    bounds: Tuple[int, ...]
    label: str = ""

    @property
    def shape(self) -> Shape:
        return tuple(b + 1 for b in self.bounds)

    @property
    def cells(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))


def level_box(d: int, L: int) -> Box:
    """Every packet with ``min M >= -L``: the Chern monomials of ``Tp^{L-1}``."""
    return Box(d, tuple((d - j) * L for j in range(1, d)), f"level L={L}")


def charge_box(d: int, p: int) -> Box:
    """Every packet with ``max M <= p``."""
    return Box(d, tuple(j * p for j in range(1, d)), f"charge p={p}")


def rect_box(d: int, p: int, L: int) -> Box:
    """Exactly the packets with ``max M <= p`` and ``min M >= -L``."""
    return Box(d, tuple(min(j * p, (d - j) * L) for j in range(1, d)), f"rect p={p},L={L}")


def beta_max(multiset: Sequence[int]) -> Tuple[int, ...]:
    """Partial sums of the decreasing ordering: the coordinatewise maximum over all orderings."""
    ordered = sorted(multiset, reverse=True)
    out, s = [], 0
    for a in ordered[:-1]:
        s += a
        out.append(s)
    return tuple(out)


def packet_box(multiset: Sequence[int]) -> Box:
    """The smallest box containing every ballot ordering of ``multiset``."""
    return Box(len(multiset), beta_max(multiset), f"packet {tuple(sorted(multiset, reverse=True))}")


def packet_depth(multiset: Sequence[int]) -> int:
    """The total degree a truncated series needs for the packet of ``multiset`` to be complete."""
    return sum(beta_max(multiset))


def is_complete(multiset: Sequence[int], box: Box) -> bool:
    return all(b <= c for b, c in zip(beta_max(multiset), box.bounds))


def complete_packets(box: Box) -> List[Tuple[int, ...]]:
    """
    Every zero-sum multiset (sorted decreasingly) whose packet is complete in ``box``.

    Depth-first over decreasing tuples, pruning on the running partial sums,
    which are exactly ``beta_max``.
    """
    d, b = box.d, box.bounds
    out: List[Tuple[int, ...]] = []

    def rec(prefix: List[int], s: int, hi: int):
        k = len(prefix)
        if k == d - 1:
            if -s <= hi:
                out.append(tuple(prefix + [-s]))
            return
        # The remaining d - k entries are each <= a and sum to -s, so
        # (d - k) a >= -s; and the next partial sum s + a must fit b[k].
        a, lower = min(hi, b[k] - s), -(s // (d - k))
        while a >= lower:
            rec(prefix + [a], s + a, a)
            a -= 1

    rec([], 0, b[0] if d > 1 else 0)
    return out


def ballot_words(multiset: Sequence[int]) -> Iterator[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Distinct orderings ``alpha`` with nonnegative proper partial sums, with their ``beta``."""
    from itertools import permutations

    for alpha in sorted(set(permutations(multiset))):
        s, beta, ok = 0, [], True
        for a in alpha[:-1]:
            s += a
            if s < 0:
                ok = False
                break
            beta.append(s)
        if ok:
            yield alpha, tuple(beta)


# --------------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------------


def _factor_terms(algebra: ChamberAlgebra) -> List[Tuple[int, List[Tuple[int, Tuple[int, ...]]]]]:
    """Each denominator factor as ``(axis, [(coefficient, shift), ...])`` with ``shift[axis] == 1``."""
    out = []
    for factor in algebra.denominator_factors:
        terms = [(int(c), tuple(int(x) for x in e)) for e, c in factor.items()]
        axes = {max(i for i, x in enumerate(e) if x) for _, e in terms}
        if len(axes) != 1:
            raise ValueError(f"factor {factor} has no common last variable; the sweep does not apply")
        axis = axes.pop()
        if any(e[axis] != 1 for _, e in terms):
            raise ValueError(f"factor {factor} is not linear in its last variable")
        out.append((axis, terms))
    return out


def _shifted(src: np.ndarray, shift: Tuple[int, ...]) -> np.ndarray:
    """``src`` translated by ``+shift`` in every coordinate (zeros shifted in), same shape."""
    out = np.zeros_like(src)
    dst_sl, src_sl = [], []
    for s, n in zip(shift, src.shape):
        if s >= n:
            return out
        dst_sl.append(slice(s, n))
        src_sl.append(slice(0, n - s))
    out[tuple(dst_sl)] = src[tuple(src_sl)]
    return out


def _seed(algebra: ChamberAlgebra, shape: Shape, dtype, modulus: Optional[int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=dtype)
    exps = [e for e in algebra.numerator if all(x < n for x, n in zip(e, shape))]
    if exps:
        idx = tuple(np.array(exps, dtype=np.int64).T)
        vals = [algebra.numerator[e] for e in exps]
        if modulus is not None:
            grid[idx] = np.array([v % modulus for v in vals], dtype=dtype)
        else:
            grid[idx] = np.array(vals, dtype=dtype)
    return grid


def sweep(algebra: ChamberAlgebra, shape: Shape, modulus: Optional[int] = None, track: bool = False):
    """
    ``N_d / prod_r (1 - f_r)`` on the box of ``shape``, modulo ``modulus`` or in float64.

    With ``modulus=None`` the sweep runs in float64 and, if ``track``, also
    returns the largest magnitude reached at any intermediate stage -- the
    bound that decides how many primes the exact path needs.
    """
    dtype = np.float64 if modulus is None else np.int64
    grid = _seed(algebra, shape, dtype, modulus)
    peak = float(np.abs(grid).max()) if track and grid.size else 0.0
    for axis, terms in _factor_terms(algebra):
        if shape[axis] <= 1:
            continue
        # recurrence along `axis`: H[t] = G[t] + sum_c c * shift_rest(H[t - 1])
        for t in range(1, shape[axis]):
            prev = grid.take(t - 1, axis=axis)
            acc = np.zeros_like(prev)
            for c, e in terms:
                rest = e[:axis] + e[axis + 1 :]
                term = _shifted(prev, rest)
                acc += c * term if modulus is None else (c * term) % modulus
            cur = np.take(grid, t, axis=axis) + acc
            if modulus is not None:
                cur %= modulus
            index = [slice(None)] * grid.ndim
            index[axis] = t
            grid[tuple(index)] = cur
        if track:
            peak = max(peak, float(np.abs(grid).max()))
    return (grid, peak) if track else grid


def exact_box(box: Box, algebra: Optional[ChamberAlgebra] = None, primes: Sequence[int] = DEFAULT_PRIMES):
    """
    The exact coefficients ``A_beta`` on ``box``, as an int64 array.

    The float64 sweep gives each final value to within a rounding error bounded
    by the largest intermediate magnitude; one prime suffices when that bound
    is below ``p / 4``, two when it is below ``p q / 4 < 2^61``.  The two-prime
    reconstruction (Garner) stays in int64 throughout, so a 50-million-cell box
    never becomes an array of Python ints.  Either way the result is checked
    against a prime held back from the reconstruction, and anything beyond two
    primes is refused rather than approximated.
    """
    alg = algebra or load_algebra(box.d)
    shape = box.shape
    approx, peak = sweep(alg, shape, None, track=True)
    # |true - float| <= peak * (#terms) * eps; 2^-30 is a very generous stand-in.
    bound = (float(np.abs(approx).max()) if approx.size else 0.0) + peak * 2.0**-30 + 1.0
    del approx
    p, q, spare = primes[0], primes[1], primes[2]
    if 4 * bound < p:
        value = sweep(alg, shape, p)
        value = np.where(value > p // 2, value - p, value)
    elif 4 * bound < p * q and p * q < 2**63:
        r = sweep(alg, shape, p)
        s = sweep(alg, shape, q)
        k = ((s - r) % q) * pow(p, -1, q) % q  # both factors < 2^31: no wrap
        value = r + p * k  # < p q < 2^63
        value = np.where(value > (p * q) // 2, value - p * q, value)
    else:
        raise OverflowError(
            f"{box.label}: values up to ~{bound:.3g} need more than two primes; "
            "use a grouped reconstruction (chern_table_exact) instead of cell values"
        )
    if not np.array_equal(value % spare, sweep(alg, shape, spare)):
        raise ArithmeticError(f"{box.label}: reconstruction failed its check modulo {spare}")
    logger.info("%s: %d cells, |A| <= %.3g, peak %.3g", box.label, value.size, bound, peak)
    return value


# --------------------------------------------------------------------------
# packet statistics
# --------------------------------------------------------------------------


@dataclass
class PacketStats:
    multiset: Tuple[int, ...]
    chern: int
    positive_mass: int
    negative_mass: int
    orderings: int
    nonzero: int
    negative: int
    dominant: int
    largest: int
    smallest: int


def packet_values(grid: np.ndarray, multiset: Sequence[int]) -> Dict[Tuple[int, ...], int]:
    """``{alpha: A_beta(alpha)}`` over the ballot orderings of ``multiset`` (which must be complete)."""
    out = {}
    for alpha, beta in ballot_words(multiset):
        if any(b >= n for b, n in zip(beta, grid.shape)):
            raise ValueError(f"packet {tuple(multiset)} is not complete in the grid")
        out[alpha] = int(grid[beta]) if beta else int(grid)
    return out


def packet_stats(grid: np.ndarray, multiset: Sequence[int]) -> PacketStats:
    vals = packet_values(grid, multiset)
    dominant = tuple(sorted(multiset, reverse=True))
    values = list(vals.values())
    return PacketStats(
        multiset=dominant,
        chern=sum(values),
        positive_mass=sum(v for v in values if v > 0),
        negative_mass=-sum(v for v in values if v < 0),
        orderings=len(values),
        nonzero=sum(1 for v in values if v),
        negative=sum(1 for v in values if v < 0),
        dominant=vals[dominant],
        largest=max(values),
        smallest=min(values),
    )


def _packet_index(box: Box):
    """
    Group the cells of ``box`` by the multiset of ``alpha``: returns
    ``(multisets, inverse)`` with ``inverse`` a flat array of group indices.

    Each sorted ``alpha`` is packed into one int64 key (entries lie in
    ``[-B, sum(bounds)]``), and rows are processed a slab at a time, so the
    grouping of a 50-million-cell box never materialises an ``(n, d)`` array.
    """
    nvars, shape = box.d - 1, box.shape
    lo = -max(box.bounds)
    base = max(box.bounds) - lo + 1
    if base**box.d >= 2**62:
        raise OverflowError(f"{box.label}: packet keys do not fit in int64")
    tail = np.stack(np.meshgrid(*[np.arange(n) for n in shape[1:]], indexing="ij"), axis=-1).reshape(
        -1, nvars - 1
    )
    keys = np.empty(box.cells, dtype=np.int64)
    step = tail.shape[0]
    for b0 in range(shape[0]):
        coords = np.concatenate([np.full((step, 1), b0, dtype=np.int64), tail], axis=1)
        alphas = np.empty((step, box.d), dtype=np.int64)
        alphas[:, 0] = coords[:, 0]
        alphas[:, 1:nvars] = coords[:, 1:] - coords[:, :-1]
        alphas[:, nvars] = -coords[:, -1]
        alphas.sort(axis=1)
        k = np.zeros(step, dtype=np.int64)
        for j in range(box.d):
            k = k * base + (alphas[:, j] - lo)
        keys[b0 * step : (b0 + 1) * step] = k
    uniq, inverse = np.unique(keys, return_inverse=True)
    del keys
    multisets = []
    for key in uniq.tolist():
        digits = []
        for _ in range(box.d):
            key, r = divmod(key, base)
            digits.append(r + lo)
        multisets.append(tuple(digits))  # least significant digit = largest entry
    return multisets, np.asarray(inverse).ravel()


def _grouped(
    values: np.ndarray, inverse: np.ndarray, ngroups: int, modulus: Optional[int] = None
) -> np.ndarray:
    order = np.argsort(inverse, kind="stable")
    flat = values.reshape(-1)[order]
    sorted_inv = inverse[order]
    starts = np.searchsorted(sorted_inv, np.arange(ngroups))
    present = np.zeros(ngroups, dtype=bool)
    present[sorted_inv] = True
    out = np.zeros(ngroups, dtype=np.int64)
    if flat.size:
        sums = np.add.reduceat(flat, np.minimum(starts, flat.size - 1))
        out = np.where(present, sums, 0)
    return out % modulus if modulus else out


def chern_table(grid: np.ndarray, box: Box) -> Dict[Tuple[int, ...], int]:
    """
    ``{M: C(M)}`` for every packet complete in ``box`` (``M`` sorted decreasingly).

    Grouping is by grid geometry, not by value.  Packets only partly inside
    the box are dropped: their sums would be truncations, not Chern
    coefficients.  Sums are exact provided they fit int64, which is checked.
    """
    if grid.shape != box.shape:
        raise ValueError(f"grid of shape {grid.shape} does not match {box.label} of shape {box.shape}")
    multisets, inverse = _packet_index(box)
    peak = float(np.abs(grid).max()) * grid.size if grid.size else 0.0
    if peak >= 2**62:
        raise OverflowError(f"{box.label}: grouped sums may exceed int64; use chern_table_exact")
    sums = _grouped(grid, inverse, len(multisets))
    return _complete_only(box, multisets, [int(v) for v in sums])


def _complete_only(box: Box, multisets, values) -> Dict[Tuple[int, ...], int]:
    complete = set(complete_packets(box))
    out = {m: v for m, v in zip(multisets, values) if m in complete}
    if len(out) != len(complete):
        raise AssertionError(f"{len(complete) - len(out)} complete packets have no cell in the grid")
    return out


def chern_table_exact(
    box: Box, algebra: Optional[ChamberAlgebra] = None, primes: Sequence[int] = DEFAULT_PRIMES
):
    """
    ``{M: C(M)}`` on ``box`` with no size ceiling: packet sums modulo each prime,
    CRT on the (few thousand) sums, verified against a held-back prime.

    The number of primes is fixed by an a-priori bound -- (cells) x (largest
    |A_beta| bound from the float sweep) -- not by the reconstructed values.
    """
    alg = algebra or load_algebra(box.d)
    multisets, inverse = _packet_index(box)
    n = len(multisets)
    approx, peak = sweep(alg, box.shape, None, track=True)
    cell_bound = (float(np.abs(approx).max()) if approx.size else 0.0) + peak * 2.0**-30 + 1.0
    del approx
    bound = cell_bound * box.cells
    used, modulus = [], 1
    for p in primes[:-1]:
        if 4 * bound < modulus:
            break
        used.append(p)
        modulus *= p
    if 4 * bound >= modulus:
        raise OverflowError(f"{box.label}: sums up to ~{bound:.3g} exceed the prime pool")
    spare = primes[len(used)]
    combined, m = [0] * n, 1
    for p in used:
        res = _grouped(sweep(alg, box.shape, p), inverse, n, p).tolist()
        inv = pow(m, -1, p) if m > 1 else 0
        combined = [c + m * (((r - c) * inv) % p) if m > 1 else r for c, r in zip(combined, res)]
        m *= p
    values = [c - m if c > m // 2 else c for c in combined]
    check = _grouped(sweep(alg, box.shape, spare), inverse, n, spare).tolist()
    if any(v % spare != c for v, c in zip(values, check)):
        raise ArithmeticError(f"{box.label}: grouped reconstruction failed its check modulo {spare}")
    logger.info("%s: %d packets, %d prime(s), verified mod %d", box.label, n, len(used), spare)
    return _complete_only(box, multisets, values)
