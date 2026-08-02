import math
import numpy as np


def extract_log_concavity_sequence(grid_np, *fixed_indices):
    """
    Extracts 1D slices from the Prefix Sum Laurent generating function (B = F_d / (1-var1)).
    Supports unpacking dynamic variables via *fixed_indices arguments for arbitrary tensors.
    """
    B_grid = np.cumsum(grid_np, axis=0)
    sequence = {}
    for i in range(B_grid.shape[0]):
        idx = (i,) + fixed_indices
        val = B_grid[idx]
        if val > 0:
            sequence[i] = int(val)
    return sequence


def check_strong_log_concavity(sequence):
    """
    The normalised Huh--Brändén inequality for M-convexity, in exact integers.

    Normalising by binomial coefficients and comparing ``a_k^2`` against
    ``a_{k-1} a_{k+1}`` invites three divisions and two multiplications in
    float64, on inputs that reach 10^22 here; a near-tie would then be settled by
    rounding.  Cross-multiplying keeps the whole test in Python integers, which
    is the convention everywhere else in this package.

    An *interior* gap is a constraint that cannot be tested, not one that is
    satisfied, and it is refused.  :func:`extract_log_concavity_sequence` records
    only positive cells, so a zero in the middle leaves a hole; skipping the
    triples that touch it would return ``True`` from a run that checked almost
    nothing, and a bare ``True`` here reads as a mathematical statement.  Where
    the recorded range *starts* is not a gap -- the prefix sums begin at zero, so
    index 0 is routinely absent -- and the triples are taken over the recorded
    range instead.
    """
    if len(sequence) < 3:
        raise ValueError(f"log-concavity needs at least three consecutive terms, got {len(sequence)}")

    n = max(sequence.keys())
    low = min(sequence.keys())
    missing = [k for k in range(low, n + 1) if k not in sequence]
    if missing:
        raise ValueError(
            f"the sequence has interior gaps at {missing}; those are constraints that "
            "cannot be tested, and skipping them would make a True verdict vacuous"
        )

    for k in range(low + 1, n):
        # a_k^2 / C(n,k)^2  >=  a_{k-1} a_{k+1} / (C(n,k-1) C(n,k+1)), cleared.
        lhs = sequence[k] ** 2 * math.comb(n, k - 1) * math.comb(n, k + 1)
        rhs = sequence[k - 1] * sequence[k + 1] * math.comb(n, k) ** 2
        if lhs < rhs:
            return False
    return True
