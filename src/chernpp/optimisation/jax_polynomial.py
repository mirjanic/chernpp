import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_exponent_mapping(nvars, max_deg):
    """Generate all monomials of total degree <= max_deg in nvars variables."""
    # Use meshgrid for fast simplex generation
    grids = np.meshgrid(*[np.arange(max_deg + 1) for _ in range(nvars)], indexing="ij")
    flat_grids = [g.ravel() for g in grids]
    exps_arr = np.column_stack(flat_grids)
    # Filter simplex
    mask = exps_arr.sum(axis=1) <= max_deg
    exps_arr = exps_arr[mask]

    # Sort for deterministic ordering (lowest degree first)
    degrees = exps_arr.sum(axis=1)
    sort_idx = np.lexsort((exps_arr[:, -1], degrees))
    exps_arr = exps_arr[sort_idx]

    # Base encoding for fast index lookup
    b = max_deg + 1
    base = np.array([b ** (nvars - 1 - i) for i in range(nvars)], dtype=np.uint64)
    flat_keys = exps_arr.dot(base)

    # Build a lookup table from flat_key to index
    # Since flat_keys can be sparse, we use an array if max_key is small, or searchsorted
    # Because exps_arr is sorted lexically by degree, we need to sort flat_keys to use searchsorted
    sorted_keys_idx = np.argsort(flat_keys)
    sorted_keys = flat_keys[sorted_keys_idx]

    return exps_arr, base, sorted_keys, sorted_keys_idx


def precompute_shifts(exps_arr, base, sorted_keys, sorted_keys_idx, shift_e):
    """Precompute the shift array using vectorized numpy."""
    N = len(exps_arr)
    shift_map = np.full(N, N, dtype=np.int32)

    shift_arr = np.array(shift_e, dtype=np.int32)
    new_exps = exps_arr + shift_arr

    # We only care about shifts that don't exceed max_deg
    # But actually, out-of-bounds will just not be found in sorted_keys.
    new_flat_keys = new_exps.dot(base)

    # Find matching indices
    pos = np.searchsorted(sorted_keys, new_flat_keys)

    # Handle out of bounds pos
    valid = pos < len(sorted_keys)
    # Also check if the key actually matches
    valid_mask = valid.copy()
    valid_mask[valid] = sorted_keys[pos[valid]] == new_flat_keys[valid]

    original_indices = sorted_keys_idx[pos[valid_mask]]
    shift_map[valid_mask] = original_indices
    return shift_map


def expand_rational_jax(num, factors, max_deg, nvars):
    """
    Expand num / prod(1 - f_r) using JAX.
    """
    logger.info(f"Preparing JAX expansion for max_deg={max_deg}, nvars={nvars}")
    exps_arr, base, sorted_keys, sorted_keys_idx = get_exponent_mapping(nvars, max_deg)
    N = len(exps_arr)
    logger.info(f"Simplex size: {N} terms")

    unique_shifts = set()
    for f in factors:
        for e in f.keys():
            unique_shifts.add(e)

    logger.info(f"Precomputing {len(unique_shifts)} shift maps...")
    shift_maps = {}
    for shift_e in unique_shifts:
        # pad shift_e to nvars
        pad_e = tuple(shift_e) + (0,) * (nvars - len(shift_e))
        shift_maps[shift_e] = precompute_shifts(exps_arr, base, sorted_keys, sorted_keys_idx, pad_e)

    logger.info("Initializing JAX arrays...")
    series_arr = np.zeros(N + 1, dtype=np.float64)
    # Map numerator
    for e, c in num.items():
        pad_e = tuple(e) + (0,) * (nvars - len(e))
        k = np.array(pad_e).dot(base)
        p = np.searchsorted(sorted_keys, k)
        if p < len(sorted_keys) and sorted_keys[p] == k:
            series_arr[sorted_keys_idx[p]] = float(c)

    j_series = jnp.array(series_arr)

    @jax.jit
    def apply_factor(series, coeffs_jnp, shifts_jnp):
        def step(val, _):
            term, res = val
            next_term_N = jnp.sum(coeffs_jnp * term[shifts_jnp], axis=0)
            next_term = jnp.zeros_like(term).at[:N].set(next_term_N)
            res += next_term
            return (next_term, res), None

        init_val = (series, series)
        final_val, _ = jax.lax.scan(step, init_val, None, length=max_deg)
        return final_val[1]

    logger.info("Running JAX loops...")
    for i, f in enumerate(factors):
        logger.debug(f"Applying factor {i+1}/{len(factors)}...")
        coeffs = []
        shifts = []
        for e, c in f.items():
            coeffs.append(float(c))
            shifts.append(shift_maps[e])
        coeffs_jnp = jnp.array(coeffs)[:, None]
        shifts_jnp = jnp.array(shifts)
        j_series = apply_factor(j_series, coeffs_jnp, shifts_jnp)

    logger.info("Converting back to dictionary...")
    result_arr = np.array(j_series)[:N]

    mask = np.abs(result_arr) > 1e-12
    nonzeros = np.nonzero(mask)[0]

    out = {}
    for i in nonzeros:
        out[tuple(int(x) for x in exps_arr[i])] = result_arr[i]

    logger.info("JAX expansion complete.")
    return out
