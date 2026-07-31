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
    Validates against the normalized Huh-Brändén inequality for M-Convexity.
    """
    n = max(sequence.keys()) if sequence else 0
    is_lorentzian = True

    for k in range(1, n):
        if k not in sequence or (k - 1) not in sequence or (k + 1) not in sequence:
            continue

        norm_k = sequence[k] / math.comb(n, k)
        norm_km1 = sequence[k - 1] / math.comb(n, k - 1)
        norm_kp1 = sequence[k + 1] / math.comb(n, k + 1)

        lhs = norm_k**2
        rhs = norm_km1 * norm_kp1

        if lhs < rhs:
            is_lorentzian = False

    return is_lorentzian
