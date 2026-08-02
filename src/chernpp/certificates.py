"""
Denominator certificates for coefficientwise positivity of chamber series.

Setting
-------
Let

    F  =  N / prod_{r=1}^{R} (1 - f_r),

where ``N`` is a polynomial and every ``f_r`` has **nonnegative** coefficients
and zero constant term.  ``F`` is then a well-defined formal power series, and
this is exactly the shape of the Bérczi--Szenes chamber series ``F_d``
(``N = numerator``, ``f_r = denominator_factors``).

Order-k certificate
-------------------
Suppose we can write

    N  =  sum_{|S| <= k}  P_S * prod_{r in S} (1 - f_r),        (*)

over subsets ``S`` of ``{1,...,R}``, with every ``P_S`` coefficientwise
nonnegative.  Then

    F  =  sum_S  P_S / prod_{r not in S} (1 - f_r),

and since ``1/(1 - f_r) = sum_j f_r^j`` is coefficientwise nonnegative, so is
every summand.  Hence ``F >= 0`` coefficientwise.

Order 0 is just the statement "``N`` itself is nonnegative".  This is a
*sufficient* condition only: failure at order ``k`` proves nothing about
positivity, it only rules out certificates of that shape.  Section 10.3 of
``a5_weak_positivity_handoff.pdf`` reports a first-order search of this kind
coming back infeasible for the A_5 paired series.

How the search works
--------------------
``P_empty`` is *not* a free unknown.  Since ``prod_{r in empty}(1-f_r) = 1``,
equation (*) determines it:

    P_empty  =  N - sum_{S nonempty} P_S * prod_{r in S} (1 - f_r).

So the search is a pure feasibility LP in the nonempty-``S`` coefficients with
the single requirement that this remainder come out nonnegative.  That keeps
the program small and, more importantly, makes the result **exactly
verifiable**: whatever rationals the floating-point LP suggests for the
nonempty parts, ``P_empty`` is then recomputed in exact arithmetic, and the
certificate is accepted only if that exact remainder is coefficientwise
nonnegative.  A certificate returned by this module is therefore a proof,
independently of anything the LP did.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .logger import get_logger
from .polynomial import (
    Poly,
    is_nonneg,
    monomials_up_to,
    negative_terms,
    one_minus,
    poly_add,
    poly_mul,
    poly_mul_many,
    poly_sub,
    poly_to_string,
    total_degree,
)

logger = get_logger(__name__)

# Denominators tried, in order, when turning the LP's floating-point answer
# into exact rationals.  Small ones first: certificates that exist tend to have
# simple coefficients, and simple coefficients survive verification.
_DENOMINATORS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 48, 120, 720, 5040)


@dataclass
class Certificate:
    """
    A verified identity ``N = sum_S P_S * prod_{r in S} (1 - f_r)``.

    ``parts`` maps each subset ``S`` (a sorted tuple of factor indices, the
    empty tuple included) to its coefficientwise-nonnegative ``P_S``.
    """

    parts: Dict[Tuple[int, ...], Poly]
    nvars: int
    varnames: Tuple[str, ...]
    order: int
    max_degree: int
    factor_names: Tuple[str, ...] = ()
    notes: List[str] = field(default_factory=list)

    def residual(self, N: Poly, factors: Sequence[Poly]) -> Poly:
        """``N - sum_S P_S * prod_{r in S}(1 - f_r)``, computed exactly."""
        acc: Poly = {}
        for S, P in self.parts.items():
            g = poly_mul_many([one_minus(factors[r], self.nvars) for r in S], self.nvars)
            acc = poly_add(acc, poly_mul(P, g))
        return poly_sub(N, acc)

    def is_valid(self, N: Poly, factors: Sequence[Poly]) -> bool:
        """True iff the identity holds exactly and every ``P_S`` is nonnegative."""
        if not all(is_nonneg(P) for P in self.parts.values()):
            return False
        return not self.residual(N, factors)

    @property
    def support_size(self) -> int:
        return sum(len(P) for P in self.parts.values())

    def summary(self) -> str:
        lines = [
            f"order-{self.order} certificate, "
            f"deg <= {self.max_degree}, {self.support_size} nonzero coefficients"
        ]
        for S in sorted(self.parts, key=lambda s: (len(s), s)):
            P = self.parts[S]
            if not P:
                continue
            if S:
                label = " * ".join(
                    (f"(1 - {self.factor_names[r]})" if self.factor_names else f"(1-f_{r})") for r in S
                )
            else:
                label = "1"
            lines.append(f"  {label}:  {poly_to_string(P, self.varnames, limit=6)}")
        return "\n".join(lines)


def certificate_subsets(n_factors: int, order: int) -> List[Tuple[int, ...]]:
    """All subsets of factor indices of size 1..order (the empty set is implicit)."""
    out: List[Tuple[int, ...]] = []
    for k in range(1, order + 1):
        out.extend(combinations(range(n_factors), k))
    return out


def search_certificate(
    N: Poly,
    factors: Sequence[Poly],
    nvars: int,
    order: int = 1,
    max_degree: Optional[int] = None,
    varnames: Tuple[str, ...] = (),
    factor_names: Tuple[str, ...] = (),
    subsets: Optional[Sequence[Tuple[int, ...]]] = None,
    tol: float = 1e-9,
) -> Optional[Certificate]:
    """
    Search for an order-``order`` denominator certificate for ``N / prod (1-f_r)``.

    ``max_degree`` caps the total degree of the unknown ``P_S`` for nonempty
    ``S``; ``P_empty`` is unconstrained in degree because it is solved for.
    Returns a verified :class:`Certificate`, or ``None`` if no certificate of
    this shape and degree was found.
    """
    varnames = tuple(varnames) or tuple(f"x{i}" for i in range(nvars))
    for r, f in enumerate(factors):
        if not is_nonneg(f):
            raise ValueError(f"factor {r} has a negative coefficient; not a valid denominator")
        if any(sum(e) == 0 for e in f):
            raise ValueError(f"factor {r} has a nonzero constant term")

    # Order 0: nothing to solve, N must simply be nonnegative.
    if order == 0:
        if is_nonneg(N):
            return Certificate({(): dict(N)}, nvars, varnames, 0, 0, tuple(factor_names))
        logger.info("order 0 fails: N has %d negative coefficients", len(negative_terms(N)))
        return None

    if max_degree is None:
        max_degree = total_degree(N)

    subs = list(subsets) if subsets is not None else certificate_subsets(len(factors), order)
    A_ub, b_ub, columns, g_of = _build_program(N, factors, nvars, subs, max_degree)

    logger.info(
        "order %d, deg <= %d -> %d unknowns, %d constraints",
        order,
        max_degree,
        A_ub.shape[1],
        A_ub.shape[0],
    )

    # Minimising total mass keeps the certificate small and sparse, which in
    # turn keeps the rational reconstruction below simple.
    res = linprog(
        c=np.ones(A_ub.shape[1]),
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=(0, None),
        method="highs",
    )
    if not res.success:
        logger.info("LP infeasible at order %d, deg <= %d (%s)", order, max_degree, res.message)
        return None

    return _reconstruct(
        res.x,
        columns,
        N,
        factors,
        g_of,
        nvars,
        varnames,
        order,
        max_degree,
        tuple(factor_names),
        tol,
    )


def _build_program(
    N: Poly,
    factors: Sequence[Poly],
    nvars: int,
    subs: Sequence[Tuple[int, ...]],
    basis_degree: int,
    ceiling: Optional[int] = None,
):
    """
    Assemble the feasibility LP ``A_ub c <= b_ub``, ``c >= 0``.

    Row ``nu`` reads ``sum_S sum_mu c[S,mu] * g_S[nu - mu] <= N[nu]``, i.e.
    exactly the statement that the remainder ``P_empty`` has a nonnegative
    coefficient on ``nu``.  Column ``k*len(basis) + i`` is the coefficient of
    ``basis[i]`` in ``P_{subs[k]}``.
    """
    g_of: Dict[Tuple[int, ...], Poly] = {
        S: poly_mul_many([one_minus(factors[r], nvars) for r in S], nvars) for S in subs
    }
    if ceiling is None:
        ceiling = max(
            total_degree(N),
            basis_degree + max((total_degree(g) for g in g_of.values()), default=0),
        )

    basis = monomials_up_to(nvars, basis_degree)
    rows = monomials_up_to(nvars, ceiling)
    row_index = {e: i for i, e in enumerate(rows)}
    columns = [(S, mu) for S in subs for mu in basis]

    # For a fixed exponent e, mu |-> mu + e is injective on the basis, so each
    # term of each g_S contributes one whole diagonal.  Precomputing those
    # index vectors keeps assembly vectorised; the naive triple loop is what
    # makes A_5 and A_6 unreachable.
    nbasis = len(basis)
    basis_arr = np.array(basis, dtype=np.int64)
    shift_rows: Dict[Tuple[int, ...], np.ndarray] = {}
    for g in g_of.values():
        for e in g:
            if e not in shift_rows:
                shifted = basis_arr + np.array(e, dtype=np.int64)
                idx = np.array([row_index.get(tuple(m), -1) for m in shifted], dtype=np.int64)
                shift_rows[e] = idx

    col_arange = np.arange(nbasis, dtype=np.int64)
    ri_blocks, ci_blocks, data_blocks = [], [], []
    for k, S in enumerate(subs):
        offset = k * nbasis
        for e, c in g_of[S].items():
            idx = shift_rows[e]
            mask = idx >= 0  # terms pushed above the ceiling are dropped
            if not mask.any():
                continue
            ri_blocks.append(idx[mask])
            ci_blocks.append(col_arange[mask] + offset)
            data_blocks.append(np.full(int(mask.sum()), float(c)))

    A_ub = coo_matrix(
        (
            np.concatenate(data_blocks) if data_blocks else np.zeros(0),
            (
                np.concatenate(ri_blocks) if ri_blocks else np.zeros(0, dtype=np.int64),
                np.concatenate(ci_blocks) if ci_blocks else np.zeros(0, dtype=np.int64),
            ),
        ),
        shape=(len(rows), len(columns)),
    ).tocsr()
    b_ub = np.array([float(N.get(nu, 0)) for nu in rows])
    return A_ub, b_ub, columns, g_of


def projection_is_feasible(
    N: Poly,
    factors: Sequence[Poly],
    nvars: int,
    order: int,
    probe_degree: int,
    subsets: Optional[Sequence[Tuple[int, ...]]] = None,
) -> bool:
    """
    Feasibility of the degree-``probe_degree`` projection of the certificate LP.

    A constraint on a monomial ``nu`` of total degree ``T`` involves only the
    ``P_S`` coefficients of degree ``<= T``, because every ``g_S`` has constant
    term 1 and no negative-degree terms.  Truncating both the constraints and
    the unknowns at ``T`` is therefore an *exact projection* of the full
    feasible set, not a further restriction.

    Consequently ``False`` is a proof: no order-``order`` certificate exists at
    **any** degree.  ``True`` is only the absence of an obstruction at this
    depth.  Both verdicts are modulo the LP solver, which works in floating
    point over coefficients rounded from exact rationals; a solver that fails to
    resolve raises rather than being read as either answer.
    """
    subs = list(subsets) if subsets is not None else certificate_subsets(len(factors), order)
    A_ub, b_ub, _, _ = _build_program(N, factors, nvars, subs, probe_degree, ceiling=probe_degree)
    res = linprog(
        c=np.zeros(A_ub.shape[1]),
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=(0, None),
        method="highs",
    )
    # HiGHS reports 0 optimal, 1 iteration limit, 2 infeasible, 3 unbounded,
    # 4 numerical difficulties.  Only 2 is the statement we are entitled to
    # publish; `res.success` is false for all of 1, 3 and 4 as well, and
    # returning False for those would report a solver bailout as a theorem.
    if res.status == 2:
        return False
    if res.status == 0:
        return True
    raise RuntimeError(
        f"the projection LP did not resolve: status {res.status} ({res.message}). "
        "Neither feasibility nor infeasibility may be concluded from this."
    )


def minimum_order(
    N: Poly,
    factors: Sequence[Poly],
    nvars: int,
    probe_degree: int = 2,
    max_order: Optional[int] = None,
) -> Optional[int]:
    """
    Least ``k`` whose degree-``probe_degree`` projection is feasible.

    A rigorous lower bound on the order of *any* denominator certificate for
    ``N / prod (1 - f_r)``: every smaller order is provably impossible at every
    degree.  Returns ``None`` if even ``max_order`` is obstructed.
    """
    top = len(factors) if max_order is None else max_order
    for k in range(0, top + 1):
        if k == 0:
            if is_nonneg(N):
                return 0
            continue
        if projection_is_feasible(N, factors, nvars, k, probe_degree):
            return k
    return None


def _reconstruct(
    x: np.ndarray,
    columns: Sequence[Tuple[Tuple[int, ...], Tuple[int, ...]]],
    N: Poly,
    factors: Sequence[Poly],
    g_of: Dict[Tuple[int, ...], Poly],
    nvars: int,
    varnames: Tuple[str, ...],
    order: int,
    max_degree: int,
    factor_names: Tuple[str, ...],
    tol: float,
) -> Optional[Certificate]:
    """
    Turn the LP's floating-point answer into an exactly verified certificate.

    For each candidate denominator ``q`` the nonempty parts are snapped to
    multiples of ``1/q``; ``P_empty`` is then recomputed exactly and the result
    accepted only if it is coefficientwise nonnegative.  Because ``P_empty``
    absorbs the whole remainder, the identity holds by construction -- the LP
    only ever supplies a guess.
    """
    active = [(j, float(x[j])) for j in range(len(x)) if x[j] > tol]
    logger.info("LP feasible; %d active unknowns, trying exact reconstruction", len(active))

    for q in _DENOMINATORS:
        parts: Dict[Tuple[int, ...], Poly] = {}
        for j, val in active:
            S, mu = columns[j]
            snapped = Fraction(round(val * q), q)
            if snapped > 0:
                parts.setdefault(S, {})[mu] = parts.setdefault(S, {}).get(mu, 0) + snapped

        acc: Poly = {}
        for S, P in parts.items():
            acc = poly_add(acc, poly_mul(P, g_of[S]))
        remainder = poly_sub(N, acc)

        if is_nonneg(remainder):
            parts[()] = remainder
            cert = Certificate(
                parts,
                nvars,
                varnames,
                order,
                max_degree,
                factor_names,
                notes=[f"nonempty parts snapped to multiples of 1/{q}"],
            )
            if not cert.is_valid(N, factors):  # belt and braces; should never fire
                logger.warning("reconstruction with 1/%d passed nonnegativity but failed verify", q)
                continue
            logger.info("verified exact certificate (coefficients in (1/%d)Z)", q)
            return cert

        logger.debug(
            "1/%d snap leaves %d negative coefficients in the remainder",
            q,
            len(negative_terms(remainder)),
        )

    logger.info("LP was feasible but no exact rational reconstruction succeeded; " "reporting no certificate")
    return None
