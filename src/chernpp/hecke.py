"""
The positivity landscape of the chamber series under 0-Hecke folds.

On a packet (the orderings ``alpha`` of a zero-sum multiset ``M``, with
``A(alpha) = A_beta`` for ballot orderings and 0 otherwise) the sorting operator

    (pi_i f)(alpha) = f(alpha) + f(s_i alpha)   if alpha_i > alpha_{i+1}
                    = f(alpha)                  if alpha_i = alpha_{i+1}
                    = 0                         if alpha_i < alpha_{i+1}

pushes weight towards decreasing order.  The ``pi_i`` satisfy the 0-Hecke
relations, so ``Pi_w`` is well defined through any reduced word, and
``Pi_{w_0}`` places the whole Chern coefficient ``C(M)`` on the decreasing
ordering.  Hence

    P_d = { w in S_d : Pi_w F_d >= 0 coefficientwise }

contains ``w_0`` iff Rimányi's conjecture holds.  Since ``pi_s`` of a
nonnegative function is nonnegative and ``Pi_{s w} = pi_s Pi_w`` when
``l(s w) = l(w) + 1``, ``P_d`` is an upper set in the left weak order; its
minimal elements are the *minimal cancellation units*: the smallest amounts of
folding after which no negative coefficient survives.  (The operators and the
``w_0`` criterion are from the colleague summary, section 3.3.)

Everything is per packet, so ``P_d`` is the intersection over packets; only
packets containing a negative ``A_beta`` constrain it.  A statement "``w`` in
``P_d``" is verified on the packets complete in a box, never beyond; a
statement "``w`` not in ``P_d``" is a proof, witnessed by a packet and a
negative coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import boxes

Perm = Tuple[int, ...]


def length(w: Perm) -> int:
    return sum(1 for i in range(len(w)) for j in range(i + 1, len(w)) if w[i] > w[j])


def left_mul(i: int, w: Perm) -> Perm:
    """``s_i w`` for permutations as tuples of images (``w[k]`` = image of ``k``)."""
    return tuple(i + 1 if x == i else i if x == i + 1 else x for x in w)


def reduced_word(w: Perm) -> List[int]:
    """A reduced word ``w = s_{i_1} ... s_{i_k}`` (indices from 1)."""
    word, w = [], tuple(w)
    while length(w):
        # find a left descent: s_i w shorter iff w^{-1}(i) > w^{-1}(i+1)
        inv = {x: k for k, x in enumerate(w)}
        i = next(i for i in range(len(w) - 1) if inv[i] > inv[i + 1])
        word.append(i + 1)
        w = left_mul(i, w)
    return word


class Packet:
    """The orderings of ``M`` as an index set, with the action of each ``pi_i`` compiled."""

    def __init__(self, multiset: Sequence[int]):
        self.multiset = tuple(sorted(multiset, reverse=True))
        self.words = sorted(set(permutations(self.multiset)))
        self.index = {a: k for k, a in enumerate(self.words)}
        d = len(self.multiset)
        self.swap, self.gt, self.eq = [], [], []
        for i in range(d - 1):
            partner = np.array([self.index[a[:i] + (a[i + 1], a[i]) + a[i + 2 :]] for a in self.words])
            self.swap.append(partner)
            self.gt.append(np.array([a[i] > a[i + 1] for a in self.words]))
            self.eq.append(np.array([a[i] == a[i + 1] for a in self.words]))

    def values(self, grid: np.ndarray) -> np.ndarray:
        vals = np.zeros(len(self.words), dtype=object if grid.dtype == object else np.int64)
        for alpha, v in boxes.packet_values(grid, self.multiset).items():
            vals[self.index[alpha]] = v
        return vals

    def pi(self, i: int, f: np.ndarray) -> np.ndarray:
        """``pi_i`` with ``i`` from 1."""
        k = i - 1
        return np.where(self.gt[k], f + f[self.swap[k]], np.where(self.eq[k], f, 0))


def nonnegative_set(packet: Packet, f: np.ndarray) -> set:
    """``{w : Pi_w f >= 0}`` by breadth-first search up the left weak order."""
    d = len(packet.multiset)
    identity = tuple(range(d))
    level = {identity: f}
    good = set()
    while level:
        nxt = {}
        for w, g in level.items():
            if (g >= 0).all():
                good.add(w)
            for i in range(d - 1):
                sw = left_mul(i, w)
                if sw not in nxt and length(sw) == length(w) + 1:
                    nxt[sw] = packet.pi(i + 1, g)
        level = nxt
    return good


def minimal_elements(ws: set) -> List[Perm]:
    """Minimal elements of an upper set in the left weak order."""
    out = []
    for w in ws:
        d = len(w)
        below = [left_mul(i, w) for i in range(d - 1)]
        if not any(length(v) == length(w) - 1 and v in ws for v in below):
            out.append(w)
    return sorted(out, key=lambda w: (length(w), w))


@dataclass
class Landscape:
    d: int
    box: str
    packets: int
    negative_packets: int
    positive_set_size: int
    minimal: List[List[int]] = field(default_factory=list)
    witnesses: Dict[str, Tuple[int, ...]] = field(default_factory=dict)


def landscape(d: int, box: boxes.Box, grid: np.ndarray = None) -> Landscape:
    """``P_d`` restricted to the packets complete in ``box``."""
    grid = boxes.exact_box(box) if grid is None else grid
    ms = boxes.complete_packets(box)
    all_w = set(permutations(range(d)))
    P = set(all_w)
    negative = 0
    witness: Dict[Perm, Tuple[int, ...]] = {}
    for m in ms:
        packet = Packet(m)
        f = packet.values(grid)
        if (f >= 0).all():
            continue
        negative += 1
        good = nonnegative_set(packet, f)
        for w in P - good:
            witness.setdefault(w, m)
        P &= good
    minimal = minimal_elements(P)
    return Landscape(
        d=d,
        box=box.label,
        packets=len(ms),
        negative_packets=negative,
        positive_set_size=len(P),
        minimal=[reduced_word(w) for w in minimal],
        witnesses={
            " ".join(map(str, reduced_word(w))) or "e": m for w, m in witness.items() if length(w) <= 2
        },
    )
