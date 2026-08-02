"""
Exact domination proofs for infinite families of Laurent coefficients.

The negative coefficients of ``F_d`` are not a finite list: they come in
one-parameter families such as ``A_(1,1,k,k-1) = -1`` for all ``k >= 2``.  A
family like that is a diagonal of a rational function, so its generating
function in the parameter is again rational and can be computed in closed form
rather than sampled.

This module extracts such a diagonal exactly, and adds it to the diagonal of
its partner under the first adjacent transposition
``tau(i, j, ...) = (j - i, j, ...)``.  If the sum is nonnegative, the paired
inequality holds along that whole family -- for every value of the parameter at
once, which no finite expansion can establish.
"""

from typing import Tuple

import sympy as sp

from .artifacts import load_algebra
from .logger import get_logger
from .polynomial import Poly

logger = get_logger(__name__)


def to_sympy(polynomial: Poly, symbols: Tuple[sp.Symbol, ...]) -> sp.Expr:
    """Rebuild a SymPy expression from an exponent dictionary."""
    total = sp.S.Zero
    for exponents, coefficient in polynomial.items():
        term = sp.Integer(coefficient)
        for symbol, power in zip(symbols, exponents):
            if power:
                term *= symbol**power
        total += term
    return total


def rational_function(dim: int):
    """
    ``F_d`` as an exact SymPy rational function, with its chamber symbols.

    Used only where genuine symbolic calculus is needed -- differentiation and
    Laurent expansion.  Coefficient extraction elsewhere goes through the far
    faster dictionary routines in :mod:`chernpp.polynomial`.
    """
    algebra = load_algebra(dim)
    symbols = sp.symbols(" ".join(algebra.chamber_vars))
    numerator = to_sympy(algebra.numerator, symbols)
    denominator = sp.prod([1 - to_sympy(factor, symbols) for factor in algebra.denominator_factors])
    return numerator / denominator, symbols


class FamilyProver:
    """Closed-form diagonals of ``F_d`` and their tau-paired sums."""

    def __init__(self, dim: int = 5):
        self.dim = dim
        self.F, self.vars = rational_function(dim)
        self.t = sp.symbols("t")

    def diagonal(self, i: int, j: int, delta: int) -> sp.Expr:
        """
        The generating function of ``A_(i, j, k, ..., k - delta)`` in ``k``.

        Extract the coefficient of ``x_1^i x_2^j`` by differentiation, then
        move to the Laurent frame ``x_{d-1} = t / x_{d-2}`` so that the
        diagonal ``k, k - delta`` becomes a single coefficient in ``x_{d-2}``.
        """
        expression = self.F
        first, second = self.vars[0], self.vars[1]
        penultimate, last = self.vars[-2], self.vars[-1]

        for _ in range(i):
            expression = sp.diff(expression, first)
        expression = expression.subs(first, 0) / sp.factorial(i)

        for _ in range(j):
            expression = sp.diff(expression, second)
        expression = expression.subs(second, 0) / sp.factorial(j)

        shifted = sp.cancel(sp.cancel(expression).subs(last, self.t / penultimate))
        series = sp.series(shifted, penultimate, 0, delta + 1).removeO()
        return sp.simplify(series.coeff(penultimate, delta))

    def paired_sum(self, i: int, j: int, delta: int) -> sp.Expr:
        """
        ``G_(i,j) + G_(j-i,j)``, the tau-paired sum along a diagonal family.

        A nonnegative result proves the paired inequality for every member of
        the family simultaneously.
        """
        logger.info("A_%d: pairing family (i=%d, j=%d, l=k-%d)", self.dim, i, j, delta)
        target = self.diagonal(i, j, delta)
        partner = self.diagonal(j - i, j, delta)
        logger.info("  target  G(t) = %s", target)
        logger.info("  partner G(t) = %s", partner)
        total = sp.cancel(target + partner)
        logger.info("  sum     G(t) = %s", sp.factor(total))
        return total
