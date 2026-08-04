import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_exponent_mapping(nvars, max_deg):
    import itertools

    c = np.array(list(itertools.combinations(range(max_deg + nvars), nvars)), dtype=np.int32)
    exps_arr = np.zeros((c.shape[0], nvars), dtype=np.int32)
    if nvars > 0:
        exps_arr[:, 0] = c[:, 0]
        for i in range(1, nvars):
            exps_arr[:, i] = c[:, i] - c[:, i - 1] - 1

    degrees = exps_arr.sum(axis=1)
    if nvars > 0:
        sort_idx = np.lexsort((exps_arr[:, -1], degrees))
    else:
        sort_idx = np.argsort(degrees)
    exps_arr = exps_arr[sort_idx]

    b = max_deg + 1
    base = np.array([b ** (nvars - 1 - i) for i in range(nvars)], dtype=np.uint64)
    flat_keys = exps_arr.astype(np.uint64).dot(base)

    sorted_keys_idx = np.argsort(flat_keys)
    sorted_keys = flat_keys[sorted_keys_idx]

    return exps_arr, base, sorted_keys, sorted_keys_idx


def precompute_shifts(
    exps_arr, base, sorted_keys, sorted_keys_idx, shift_e, max_deg, direction="forward", return_sparse=False
):
    N = len(exps_arr)
    shift_arr = np.array(shift_e, dtype=np.int64)

    if direction == "forward":
        degrees = exps_arr.sum(axis=1)
        deg_mask = degrees + shift_arr.sum() <= max_deg
        shift_key = shift_arr.dot(base)
        # Add shift_key to existing keys
        valid_keys = exps_arr[deg_mask].dot(base) + shift_key
    else:
        deg_mask = (exps_arr >= shift_arr).all(axis=1)
        shift_key = shift_arr.dot(base)
        valid_keys = exps_arr[deg_mask].dot(base) - shift_key

    # Only search for keys that passed the degree/bounds check
    pos = np.searchsorted(sorted_keys, valid_keys)

    # Filter those that actually exist in the mapping
    valid_idx = pos < len(sorted_keys)
    valid_idx[valid_idx] = sorted_keys[pos[valid_idx]] == valid_keys[valid_idx]

    original_indices = sorted_keys_idx[pos[valid_idx]]
    src_indices = np.where(deg_mask)[0][valid_idx]

    if return_sparse:
        return original_indices.astype(np.int32, copy=False), src_indices.astype(np.int32, copy=False)

    shift_map = np.full(N, N, dtype=np.int32)
    shift_map[src_indices] = original_indices
    return shift_map


@jax.jit(static_argnums=(3, 4))
def _apply_factor_single(series, coeffs_jnp, shifts_jnp, N, max_deg):
    def step(val, _):
        term, res = val
        next_term = jnp.zeros_like(term)

        def add_shift(j, nt):
            return nt.at[shifts_jnp[j]].add(coeffs_jnp[j, 0] * term[:N])

        next_term = jax.lax.fori_loop(0, shifts_jnp.shape[0], add_shift, next_term)
        next_term = next_term.at[N].set(0.0)
        res += next_term
        return (next_term, res), None

    init_val = (series, series)
    final_val, _ = jax.lax.scan(step, init_val, None, length=max_deg)
    return final_val[1]


def expand_rational_jax(num, factors, max_deg, nvars):
    """Fallback for single numerator."""
    exps_arr, base, sorted_keys, sorted_keys_idx = get_exponent_mapping(nvars, max_deg)
    N = len(exps_arr)

    unique_shifts = set()
    for f in factors:
        for e in f.keys():
            unique_shifts.add(e)

    shift_maps = {}
    for shift_e in unique_shifts:
        pad_e = tuple(shift_e) + (0,) * (nvars - len(shift_e))
        shift_maps[shift_e] = precompute_shifts(exps_arr, base, sorted_keys, sorted_keys_idx, pad_e, max_deg)

    series_arr = np.zeros(N + 1, dtype=np.float64)
    exps = []
    coeffs = []
    for e, c in num.items():
        exps.append(tuple(e) + (0,) * (nvars - len(e)))
        coeffs.append(float(c))
    if exps:
        k = np.array(exps).dot(base)
        p = np.searchsorted(sorted_keys, k)
        p = np.minimum(p, len(sorted_keys) - 1)
        valid = sorted_keys[p] == k
        series_arr[sorted_keys_idx[p[valid]]] = np.array(coeffs, dtype=np.float64)[valid]

    j_series = jnp.array(series_arr)

    for i, f in enumerate(factors):
        coeffs = []
        shifts = []
        for e, c in f.items():
            coeffs.append(float(c))
            shifts.append(shift_maps[e])
        coeffs_jnp = jnp.array(coeffs)[:, None]
        shifts_jnp = jnp.array(shifts)
        j_series = _apply_factor_single(j_series, coeffs_jnp, shifts_jnp, N, max_deg)

    result_arr = np.array(j_series)[:N]
    mask = np.abs(result_arr) > 1e-12
    nonzeros = np.nonzero(mask)[0]
    out = {}
    for i in nonzeros:
        out[tuple(int(x) for x in exps_arr[i])] = result_arr[i]
    return out
