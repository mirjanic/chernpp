"""
Closed forms on sectors of the Chern table, checked against the box engine.

*Plane sector* (``max M <= 1``).  Write ``M = (1^{d-b}, 1 - s_1, ..., 1 - s_b)``
with every ``s_i >= 1``; the ``s_i`` are the block sizes of a set partition of
``[d]`` into ``b`` blocks.  The colleague summary proves positivity here via
Jucys--Murphy elements; the value is Kreweras' count of *noncrossing*
partitions of ``[d]`` with those block sizes,

    C(M) = d! / ((d - b + 1)! prod_i m_i!),

``m_i`` the number of blocks of size ``i``.  This is a verified identity (every plane-sector packet at
d <= 7), not a proof.

*Stabilisation.*  ``F_d |_{x_{d-1} = 0} = F_{d-1}``: the top chamber variable
switched off recovers the previous series.  Checked cell by cell on boxes.
"""

from collections import Counter
from math import factorial
from typing import Dict, Sequence, Tuple

import numpy as np

from . import boxes


def is_plane(multiset: Sequence[int]) -> bool:
    return max(multiset) <= 1


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


def stabilisation_holds(d: int, bounds: Sequence[int]) -> bool:
    """``F_d`` with ``x_{d-1} = 0`` against ``F_{d-1}`` on the box ``bounds`` (length ``d - 2``)."""
    top = boxes.exact_box(boxes.Box(d, tuple(bounds) + (0,), f"A_{d} face"))
    low = boxes.exact_box(boxes.Box(d - 1, tuple(bounds), f"A_{d - 1}"))
    return bool(np.array_equal(top[..., 0], low))
