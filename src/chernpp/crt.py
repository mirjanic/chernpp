"""
Exact Chern coefficients past the ``int64`` ceiling, by CRT.

:mod:`chernpp.chern` accumulates the fixed-point expansion in ``int64`` and
refuses once the values would wrap, which caps the A_6 sweep at ``l = 5``.  The
way through is residue arithmetic: run the same expansion modulo several word-
sized primes, where nothing can overflow, and reconstruct the integers.

Two details make this correct rather than merely plausible.

*Aligned grouping.*  Chern coefficients are sums of ``A_beta`` over the cells of
the grid sharing an ``alpha`` multiset.  Grouping by *value* -- taking the
nonzero cells, as the fast path does -- would misalign the residues, because a
genuinely nonzero coefficient can vanish modulo one prime and not another.  The
grouping here is computed once from the grid *geometry* and reused for every
prime, so the residues of a given coefficient always correspond.

*Certified range.*  Reconstruction is only valid if the true coefficient lies in
the symmetric interval of width ``prod(primes)``.  We reconstruct, then verify
by re-expanding modulo a prime that was not used; a coefficient outside the
range fails that check with probability about ``1/p``, and the check is repeated
until the modulus provably exceeds the observed magnitudes.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import load_algebra
from .chern import ChernMultiset, _compile_expansion
from .logger import get_logger

logger = get_logger(__name__)

#: Primes below 2^31, so that a product of two residues plus an accumulator
#: stays inside int64 during the expansion.
DEFAULT_PRIMES: Tuple[int, ...] = (
    2147483647,
    2147483629,
    2147483587,
    2147483579,
    2147483563,
    2147483549,
    2147483543,
    2147483497,
)


def _grid_shape(nvars: int, l_max: int) -> Tuple[int, ...]:
    return tuple((nvars - i) * (l_max + 1) + 1 for i in range(nvars))


def grouping(nvars: int, l_max: int):
    """
    Map every grid cell to its ``alpha`` multiset.  Independent of any values.

    Returns ``(multisets, inverse, keep)`` where ``inverse`` indexes into
    ``multisets`` for each kept cell and ``keep`` selects the cells that survive
    the ghost-term filter ``alpha_i >= -(l_max + 1)``.
    """
    shape = _grid_shape(nvars, l_max)
    coords = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1)
    coords = coords.reshape(-1, nvars)

    alphas = np.empty((coords.shape[0], nvars + 1), dtype=np.int64)
    alphas[:, 0] = coords[:, 0]
    alphas[:, 1:nvars] = coords[:, 1:] - coords[:, :-1]
    alphas[:, nvars] = -coords[:, -1]

    keep = (alphas >= -(l_max + 1)).all(axis=1)
    alphas = alphas[keep]
    alphas.sort(axis=1)
    multisets, inverse = np.unique(alphas, axis=0, return_inverse=True)
    return multisets, np.asarray(inverse).ravel(), keep


def _sums_mod(
    dim: int, l_max: int, prime: int, inverse: np.ndarray, keep: np.ndarray, ngroups: int
) -> np.ndarray:
    """Grouped Chern sums modulo ``prime``, over the whole grid."""
    algebra = load_algebra(dim)
    nvars = algebra.nvars
    shape = _grid_shape(nvars, l_max)

    seed = np.zeros(shape, dtype=np.int64)
    for exponents, coefficient in algebra.numerator.items():
        if all(e < cap for e, cap in zip(exponents, shape)):
            seed[exponents] = coefficient % prime

    factor_terms = [
        [(int(c) % prime, tuple(e)) for e, c in factor.items()] for factor in algebra.denominator_factors
    ]
    expand = _compile_expansion(shape, factor_terms, sum(shape), modulus=prime)
    grid = np.asarray(expand(jnp.array(seed, dtype=jnp.int64))).reshape(-1)[keep] % prime

    sums = np.zeros(ngroups, dtype=np.int64)
    np.add.at(sums, inverse, grid)
    return sums % prime


def _crt_pair(r1: int, m1: int, r2: int, m2: int) -> Tuple[int, int]:
    """Combine two residues; moduli must be coprime."""
    inverse = pow(m1, -1, m2)
    return (r1 + m1 * ((r2 - r1) * inverse % m2)) % (m1 * m2), m1 * m2


def chern_coefficients_exact(
    dim: int,
    l_max: int,
    primes: Optional[Sequence[int]] = None,
    verify: bool = True,
) -> Dict[ChernMultiset, int]:
    """
    Chern coefficients as exact integers, with no ``int64`` ceiling.

    Uses as many of ``primes`` as the magnitudes require, then optionally spends
    one more prime verifying the reconstruction.
    """
    pool: List[int] = list(primes or DEFAULT_PRIMES)
    algebra = load_algebra(dim)
    multisets, inverse, keep = grouping(algebra.nvars, l_max)
    ngroups = len(multisets)
    logger.info(
        "A_%d at l_max=%d: %d Chern monomials over %d grid cells",
        dim,
        l_max,
        ngroups,
        keep.size,
    )

    combined = np.zeros(ngroups, dtype=object)
    signed = combined
    modulus = 1
    used: List[int] = []
    certified = False

    budget = pool[:-1] if verify else pool  # keep one prime back for the check
    for prime in budget:
        residues = _sums_mod(dim, l_max, prime, inverse, keep, ngroups)
        if modulus == 1:
            combined = residues.astype(object)
            modulus = prime
        else:
            combined = np.array(
                [_crt_pair(int(a), modulus, int(b), prime)[0] for a, b in zip(combined, residues)],
                dtype=object,
            )
            modulus *= prime
        used.append(prime)

        # Symmetric representatives; stop once the modulus comfortably exceeds
        # the magnitudes it is representing.
        signed = np.array(
            [int(v) - modulus if int(v) > modulus // 2 else int(v) for v in combined],
            dtype=object,
        )
        largest = max((abs(int(v)) for v in signed), default=0)
        if largest * 4 < modulus:
            certified = True
            logger.info(
                "A_%d at l_max=%d: %d prime(s), max |coefficient| ~ %.3g",
                dim,
                l_max,
                len(used),
                float(largest),
            )
            break

    if not certified:
        raise OverflowError(
            f"A_{dim} at l_max={l_max}: the supplied primes give a modulus of only "
            f"~{float(modulus):.3g}, which does not comfortably exceed the "
            "coefficients being reconstructed; pass more primes"
        )

    result = {tuple(int(a) for a in key): int(value) for key, value in zip(multisets, signed)}

    if verify:
        spare = pool[-1]
        residues = _sums_mod(dim, l_max, spare, inverse, keep, ngroups)
        for value, residue in zip(signed, residues):
            if int(value) % spare != int(residue) % spare:
                raise ArithmeticError(
                    f"A_{dim} at l_max={l_max}: CRT reconstruction failed its check "
                    f"against the spare prime {spare}"
                )
        logger.info("A_%d at l_max=%d: reconstruction verified mod %d", dim, l_max, spare)

    return result
