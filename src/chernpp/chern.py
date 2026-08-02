"""
Chern coefficients of the Thom polynomial, by dense expansion of ``F_d``.

The chamber series is expanded on a bounded integer grid: for relative
dimension ``l`` only exponents with ``beta_j <= (d - j)(l + 1)`` can contribute,
so a box of that size captures every relevant coefficient exactly.  The
expansion is a fixed-point iteration -- multiplying by ``1/(1 - f_r)`` is
repeated shifted accumulation -- compiled by XLA and run in int64.

A Laurent monomial ``x^beta`` equals ``z^alpha`` with

    alpha = (beta_1, beta_2 - beta_1, ..., beta_{d-1} - beta_{d-2}, -beta_{d-1}),

and contributes ``A_beta * prod_i c_{l+1+alpha_i}``.  Grouping by the multiset
of ``alpha`` therefore yields the Chern coefficients ``C(M)`` directly.  Since
``C(M)`` depends on ``l`` only through ``M`` (the l-free reduction), the same
value appears at every relative dimension admitting that multiset.
"""

import os
from typing import Dict, Tuple

# By default JAX grabs three quarters of the GPU on the first import, and if
# something else already holds that memory the allocator walks a ladder of
# smaller requests, printing each failure to stderr as an ERROR.  Those messages
# are alarming and mean nothing -- the run then succeeds on a smaller arena.
# Allocating on demand removes the ladder without suppressing any logging, so a
# genuine out-of-memory still surfaces.  ``setdefault`` leaves an explicit
# setting in the environment alone.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import load_algebra
from .logger import get_logger

jax.config.update("jax_enable_x64", True)
logger = get_logger(__name__)

ChernMultiset = Tuple[int, ...]

#: Coefficients accumulate in int64.  Past this bound the accumulator wraps,
#: and a wrapped value reads as a large *negative* number -- indistinguishable
#: from a counterexample to the very conjecture under test.  We refuse instead.
_INT64_SAFE_BOUND = 2.0**62


def _compile_expansion(shape, factor_terms, steps, modulus=None):
    """
    Compile the fixed-point expansion of ``1 / prod_r (1 - f_r)`` on a grid.

    With ``modulus`` set, every accumulation is reduced, so nothing can
    overflow however large the true coefficients are.  That is what
    :mod:`chernpp.crt` uses to get past the int64 ceiling.
    """

    @jax.jit
    def expand(initial):
        grid = initial
        for terms in factor_terms:

            def step(current, _, base=grid, terms=terms):
                nxt = base
                for coefficient, shift in terms:
                    padded = jnp.pad(current, tuple((s, 0) for s in shift))
                    nxt = nxt + coefficient * padded[tuple(slice(0, n) for n in shape)]
                return (nxt if modulus is None else nxt % modulus), None

            grid, _ = jax.lax.scan(step, grid, None, length=steps)
        return grid

    return expand


def laurent_grid(dim: int, l_max: int = 2, check_overflow: bool = True) -> np.ndarray:
    """
    The coefficients ``A_beta`` of ``F_d`` on the box relevant to ``l_max``.

    With ``check_overflow`` the same fixed point is replayed in float64 as a
    magnitude probe, and :class:`OverflowError` is raised rather than returning
    silently wrapped values.
    """
    algebra = load_algebra(dim)
    nvars = algebra.nvars

    shape = tuple((nvars - i) * (l_max + 1) + 1 for i in range(nvars))
    seed = np.zeros(shape, dtype=np.int64)
    for exponents, coefficient in algebra.numerator.items():
        if all(e < cap for e, cap in zip(exponents, shape)):
            seed[exponents] = coefficient

    factor_terms = [[(int(c), tuple(e)) for e, c in factor.items()] for factor in algebra.denominator_factors]
    expand = _compile_expansion(shape, factor_terms, sum(shape))
    grid = np.asarray(expand(jnp.array(seed, dtype=jnp.int64)))

    if check_overflow:
        # The iteration accumulates a partial sum whose terms have mixed signs,
        # so a cell can wrap mid-scan and come back inside the range by the end.
        # Comparing the final grid against the float64 replay catches a wrap
        # wherever it happened; testing only the final magnitude does not.
        probe = np.asarray(expand(jnp.array(seed, dtype=jnp.float64)))
        peak = float(np.abs(probe).max()) if probe.size else 0.0
        wrapped = probe.size and not np.allclose(grid.astype(np.float64), probe, rtol=1e-9, atol=0.0)
        if peak > _INT64_SAFE_BOUND or wrapped:
            raise OverflowError(
                f"A_{dim} at l_max={l_max}: coefficients reach ~{peak:.3g}, beyond the "
                f"int64 range ({_INT64_SAFE_BOUND:.3g}). The grid has wrapped and would "
                "report spurious negative coefficients. Lower l_max, or move this "
                "evaluation to exact arithmetic."
            )
    return grid


def chern_coefficients(
    dim: int, l_max: int = 2, check_overflow: bool = True
) -> Tuple[np.ndarray, Dict[ChernMultiset, int]]:
    """
    Return the Laurent grid and ``{alpha multiset: C(M)}`` for relative dimension ``l_max``.

    Rimányi's conjecture at this ``l`` is exactly the assertion that every value
    in the returned mapping is nonnegative.
    """
    grid = laurent_grid(dim, l_max, check_overflow)
    nvars = grid.ndim
    if nvars == 0:
        # d = 1 has no chamber variables, so F_1 = 1 and the only zero-sum
        # multiset is (0): Porteous's Tp(A_1) = c_{l+1}, with coefficient 1.
        return grid, {(0,): int(grid)}

    coords = np.stack(np.nonzero(grid), axis=1)
    if coords.size == 0:
        return grid, {}
    values = grid[tuple(coords.T)]

    alphas = np.empty((coords.shape[0], nvars + 1), dtype=np.int64)
    alphas[:, 0] = coords[:, 0]
    alphas[:, 1:nvars] = coords[:, 1:] - coords[:, :-1]
    alphas[:, nvars] = -coords[:, -1]

    # c_j = 0 for j < 0, so any alpha_i < -(l_max + 1) kills the whole monomial.
    keep = (alphas >= -(l_max + 1)).all(axis=1)
    alphas, values = alphas[keep], values[keep]
    if alphas.size == 0:
        return grid, {}

    alphas.sort(axis=1)
    unique, inverse = np.unique(alphas, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()

    # Exact int64 grouped sum; bincount would route through float64 and lose bits.
    order = np.argsort(inverse, kind="stable")
    starts = np.searchsorted(inverse[order], np.arange(len(unique)))
    totals = np.add.reduceat(values[order], starts)

    return grid, {tuple(int(a) for a in key): int(total) for key, total in zip(unique, totals)}


def format_monomial(multiset: ChernMultiset, l_max: int) -> str:
    """
    Render an alpha multiset as a Chern monomial, e.g. ``c_2 * c_1^2``.

    Returns ``"0"`` if some ``c_{l+1+alpha}`` has a negative index, and ``"1"``
    for the multiset consisting entirely of ``c_0 = 1``.
    """
    classes = []
    for alpha in multiset:
        index = l_max + 1 + alpha
        if index < 0:
            return "0"
        if index > 0:
            classes.append(index)
    if not classes:
        return "1"
    return " * ".join(
        f"c_{index}" if classes.count(index) == 1 else f"c_{index}^{classes.count(index)}"
        for index in sorted(set(classes), reverse=True)
    )


def thom_polynomial(dim: int, l_max: int = 0) -> str:
    """The Thom polynomial of A_dim at relative dimension ``l_max``, as a string."""
    _, coefficients = chern_coefficients(dim, l_max)
    terms = []
    for multiset, coefficient in coefficients.items():
        if coefficient == 0:
            continue
        monomial = format_monomial(multiset, l_max)
        if monomial == "0":
            continue
        terms.append(f"{coefficient}*{monomial}" if monomial != "1" else str(coefficient))
    return " + ".join(terms).replace("+ -", "- ")
