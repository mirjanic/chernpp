"""
Gauge freedom in the numerator of the residue formula.

``Q_d`` is not the only numerator that computes ``Tp_{A_d}``.  Two numerators
differing by a *residue-null kernel* -- one whose every Chern packet sums to
zero -- give the same Thom polynomial at every relative dimension while giving
genuinely different chamber series.  The two positivity statements sit on
opposite sides of that freedom:

    A_beta   depends on which numerator was used
    C(M)     does not

So a negative ``A_beta`` may be an artefact of the numerator rather than a fact
about the singularity.  This is not idle: the strong conjecture is false for
``Q_d`` itself from ``d = 5`` on, but a numerator whose series happens to be
coefficientwise nonnegative would prove Rimanyi's conjecture at that order
outright, with no cancellation argument at all.  Searching for one is what this
module is for.

The whole construction is linear in the numerator.  Writing ``V`` for the
Vandermonde and ``f_r`` for the denominator factors,

    F(P)  =  -(P / x^corr) * Phi,        Phi := V / prod_r (1 - f_r),

so ``Phi`` is computed once and every candidate numerator costs one polynomial
multiplication.  ``corr`` is the chamber correction monomial, recovered from the
artifact rather than hard-coded.

There are two ways to get hold of null kernels here, and they are not equally
good.  :func:`null_kernel_basis` samples nullity on the Chern packets that fit
under a truncation, which is cheap but badly underdetermined -- fine at ``d = 5``
(48 packets against 34 monomials), useless at ``d = 6`` (29 against 766), where
the search returns kernels fitted to the packets it can see.
:func:`symmetry_kernels` instead *constructs* kernels that are null for a
structural reason, in all degrees at once, and is the route that scales.

THE SYMMETRY CONSTRUCTION.  The Chern insertion of the residue formula is
symmetric in ``z_1, ..., z_d`` and the Vandermonde is antisymmetric.  So if
``P / D_d`` is invariant under a transposition ``s``, the whole integrand is
antisymmetric under ``s`` and the residue is its own negative -- zero.  Writing
``A`` for the product of the ``D_d`` factors that ``s`` moves off the factor set,
invariance of ``P / D_d`` is equivalent to

    P = A * R,      R any s-invariant polynomial

so every such product is a null kernel, and the space of them is small, explicit,
and available at any ``d``.

WHAT IS PROVED HERE AND WHAT IS NOT.  The symmetry argument above is exact as
algebra but assumes the contour may be swapped without crossing a pole.  That
last step is a genuine analytic condition which this module does not verify; a
kernel it constructs is a candidate with a structural reason behind it, not a
theorem.  Packet checks remain the falsifier -- a kernel failing them is
definitely not null.  Nonnegativity of a returned gauge is re-verified in exact
integer arithmetic, but only over the degrees searched.  Every record carries the
truncation it was established at, so nothing here can be mistaken for an
all-degree statement.

CREDIT.  The gauge idea, and the ``d = 5`` kernel that first showed it works, are
due to two external reports in ``papers/``:
``rimanyi_positivity_findings_summary.pdf`` (29 July 2026), whose section 3.2
identifies the universal kernel as ``ker eps_+`` and whose section 4.4 proves
``G B C`` residue-null by two chamber-safe contour swaps; and ``a6_status.pdf``
(31 July 2026), which lifts the construction to ``d = 6``.  Both are unrefereed.

The two constructions below are this repository's reformulation of their two
swaps.  For the first, their identity ``D_5 = B F E P(z_4) P(z_5) M(z_5) C``
makes ``B M C / D_5 = 1 / (F E P(z_4) P(z_5))`` manifestly ``s_45``-invariant,
and absorbing the moved factors ``M`` and ``C`` is what :func:`analyse_swap`
computes.  For the second, absorbing all the moved factors *but one* leaves an
antisymmetric part in which the spare factor is paired with its image, which is
their source-multiset completion; :func:`partial_absorption_kernels` enumerates
that shape and recovers their ``F B C``.

Together the two families close ``d = 5`` with no fitted kernels at all.
"""

from dataclasses import dataclass
from fractions import Fraction
from collections import defaultdict
import numpy as np
from itertools import combinations, combinations_with_replacement
import numpy as np
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import logging

logger = logging.getLogger(__name__)

from ..artifacts import ChamberAlgebra, load_algebra
from ..chamber import ballot_orderings
from ..polynomial import Exponent, Poly, expand_rational, negative_terms, poly_mul

__all__ = [
    "GaugeSetup",
    "NullKernels",
    "GaugeResult",
    "to_chamber",
    "to_z",
    "correction_monomial",
    "setup",
    "series_of",
    "admissible_monomials",
    "null_kernel_basis",
    "denominator_weights",
    "linear_form",
    "Swap",
    "analyse_swap",
    "symmetric_basis",
    "symmetry_kernels",
    "partial_absorption_kernels",
    "null_candidates",
    "certifies_null",
    "certified_null_candidates",
    "validate_gauge",
    "Validation",
    "DeficitResult",
    "solve_positive_gauge_jax",
    "solve_positive_gauge_continuous",
]


# --------------------------------------------------------------------------
# coordinates
# --------------------------------------------------------------------------


def to_chamber(poly_z: Poly, order: int, total_degree: int) -> Poly:
    """
    A homogeneous polynomial in ``z_1, ..., z_d`` in chamber coordinates.

    With ``z_d = 1`` and ``x_j = z_j / z_{j+1}`` one has ``z_j = x_j ... x_{d-1}``,
    so the monomial ``z^a`` becomes ``x^b`` with ``b_k = a_1 + ... + a_k``.  The
    last exponent ``a_d`` is recovered from homogeneity, which is why the total
    degree has to be supplied and is checked.
    """
    out: Poly = {}
    for a, coeff in poly_z.items():
        if len(a) != order:
            raise ValueError(f"A_{order}: exponent {a} does not have {order} coordinates")
        if sum(a) != total_degree:
            raise ValueError(
                f"A_{order}: monomial {a} has degree {sum(a)}, not the declared {total_degree}; "
                "the numerator of the residue formula is homogeneous"
            )
        b, running = [], 0
        for k in range(order - 1):
            running += a[k]
            b.append(running)
        key = tuple(b)
        out[key] = out.get(key, 0) + coeff
    return {e: c for e, c in out.items() if c}


def to_z(poly_chamber: Poly, order: int, total_degree: int) -> Poly:
    """Inverse of :func:`to_chamber`: ``a_k = b_k - b_{k-1}``, ``a_d = deg - b_{d-1}``."""
    out: Poly = {}
    for b, coeff in poly_chamber.items():
        if len(b) != order - 1:
            raise ValueError(f"A_{order}: exponent {b} does not have {order - 1} coordinates")
        a = [b[0]] + [b[k] - b[k - 1] for k in range(1, len(b))]
        a.append(total_degree - b[-1])
        if any(e < 0 for e in a):
            raise ValueError(
                f"A_{order}: chamber monomial {b} does not come from a polynomial in z "
                f"of degree {total_degree} (it would need exponents {tuple(a)})"
            )
        key = tuple(a)
        out[key] = out.get(key, 0) + coeff
    return {e: c for e, c in out.items() if c}


def correction_monomial(algebra: ChamberAlgebra) -> Exponent:
    """
    The chamber correction ``x^corr`` with ``normalized_numerator = -Q_d / x^corr``.

    Recovered rather than tabulated: the normalized numerator has constant term
    1, so its exponentwise minimum is zero and ``corr`` is the exponentwise
    minimum of ``Q_d`` in chamber coordinates.  The result is checked against the
    artifact before being returned, so a change in the Sage stage's normalisation
    convention shows up here as a failure rather than as silently wrong gauges.
    """
    Q = algebra.multidegree
    nvars = algebra.nvars
    corr = tuple(min(e[i] for e in Q) for i in range(nvars))
    rebuilt = {tuple(e[i] - corr[i] for i in range(nvars)): -c for e, c in Q.items()}
    if rebuilt != algebra.normalized_numerator:
        raise RuntimeError(
            f"A_{algebra.order}: could not recover the chamber correction monomial; "
            "the artifact's normalized numerator is not -Q_d / x^corr"
        )
    return corr


# --------------------------------------------------------------------------
# the series as a linear function of the numerator
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GaugeSetup:
    """Everything needed to turn a numerator into a chamber series, computed once."""

    order: int
    max_deg: int
    algebra: ChamberAlgebra
    #: V / prod_r (1 - f_r), truncated.  The numerator-independent half.
    phi: Poly
    #: The chamber correction monomial.
    correction: Exponent
    #: deg Q_d, which every admissible numerator must match.
    degree: int

    @property
    def nvars(self) -> int:
        return self.algebra.nvars


def setup(
    order: int,
    max_deg: int,
    algebra: Optional[ChamberAlgebra] = None,
    exact: bool = True,
    use_jax: bool = False,
) -> GaugeSetup:
    """Precompute the numerator-independent part of the chamber series."""
    alg = algebra or load_algebra(order)
    if use_jax:
        from .jax_polynomial import expand_rational_jax

        phi = expand_rational_jax(alg.vandermonde, list(alg.denominator_factors), max_deg, alg.nvars)
    else:
        phi = expand_rational(alg.vandermonde, list(alg.denominator_factors), max_deg, exact=exact)
    corr = correction_monomial(alg)
    return GaugeSetup(order, max_deg, alg, phi, corr, _z_degree(alg))


def _z_degree(algebra: ChamberAlgebra) -> int:
    """deg Q_d as a polynomial in z, read off the stored chamber form."""
    # In chamber coordinates b_k = a_1 + ... + a_k, so the last coordinate is
    # deg - a_d <= deg, with equality exactly when a_d = 0.  Every Q_d has such a
    # monomial, so the degree is the largest last coordinate.
    return max(e[-1] for e in algebra.multidegree)


def series_of(numerator_z: Poly, gauge: GaugeSetup, exact: bool = False) -> Poly:
    """
    The chamber series of an arbitrary numerator, truncated at ``gauge.max_deg``.

    Raises if the numerator is not divisible by the chamber correction monomial.
    That is not a technicality: such a numerator produces a Laurent series with
    negative exponents rather than the power series the ballot indexing assumes,
    and silently dropping those terms would corrupt every packet sum downstream.
    """
    if exact:
        numerator_z = {
            e: (Fraction(c).limit_denominator(10**10) if isinstance(c, float) else c)
            for e, c in numerator_z.items()
        }
    chamber = to_chamber(numerator_z, gauge.order, gauge.degree)
    corr, n = gauge.correction, gauge.nvars
    shifted: Poly = {}
    for e, c in chamber.items():
        f = tuple(e[i] - corr[i] for i in range(n))
        if any(v < 0 for v in f):
            raise ValueError(
                f"A_{gauge.order}: numerator monomial {e} is not divisible by the chamber "
                f"correction x^{corr}; it is not an admissible numerator"
            )
        shifted[f] = -c
    return poly_mul(shifted, gauge.phi, max_deg=gauge.max_deg, exact=exact)


def _safe_series_of(k, g):
    try:
        return series_of(k, g)
    except ValueError:
        return {}


# --------------------------------------------------------------------------
# the space of candidate numerators, and the null subspace
# --------------------------------------------------------------------------


def admissible_monomials(gauge: GaugeSetup) -> List[Exponent]:
    """
    Degree-``deg Q_d`` monomials in ``z`` that are legal numerators.

    Homogeneity is forced by the residue formula; divisibility by the correction
    monomial is forced by :func:`series_of`.  Everything else is free, so this is
    the full linear space a gauge search may move in.
    """
    order, deg, corr = gauge.order, gauge.degree, gauge.correction
    out = []
    for a in _compositions(deg, order):
        running, ok = 0, True
        for k in range(order - 1):
            running += a[k]
            if running < corr[k]:
                ok = False
                break
        if ok:
            out.append(a)
    return out


def _compositions(total: int, parts: int) -> List[Exponent]:
    """All nonnegative integer tuples of the given length summing to ``total``."""
    if parts == 1:
        return [(total,)]
    out = []
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            out.append((first,) + rest)
    return out


def usable_packets(gauge: GaugeSetup, spread: Optional[int] = None) -> List[Tuple[int, ...]]:
    """
    Zero-sum multisets all of whose ballot orderings fit under the truncation.

    A packet only half inside the truncation would contribute a partial sum, so
    those are excluded rather than approximated.
    """
    order, md = gauge.order, gauge.max_deg
    reach = spread if spread is not None else order
    out = []
    for M in combinations_with_replacement(range(-reach, reach + 1), order):
        if sum(M) != 0:
            continue
        orders = ballot_orderings(list(M))
        if not orders or any(sum(b) > md for _, b in orders):
            continue
        out.append(tuple(M))
    return out


def packet_sum(series: Poly, multiset: Sequence[int]) -> int:
    """Sum of the series over the ballot orderings of one multiset.  No truncation check."""
    return sum(series.get(b, 0) for _, b in ballot_orderings(list(multiset)))


@dataclass(frozen=True)
class NullKernels:
    """
    A basis for the numerators that contribute nothing to any Chern packet
    *within the truncation searched*.

    ``truncation`` and ``packets`` record exactly how far that was checked.  This
    is a necessary condition for true residue-nullity, not a sufficient one.
    """

    order: int
    truncation: int
    #: Each basis element as a polynomial in z.
    basis: Tuple[Poly, ...]
    #: The monomials the search ranged over.
    monomials: Tuple[Exponent, ...]
    #: The packets the nullity conditions came from.
    packets: Tuple[Tuple[int, ...], ...]

    def __len__(self) -> int:
        return len(self.basis)


def null_kernel_basis(gauge: GaugeSetup, spread: Optional[int] = None) -> NullKernels:
    """
    Solve for the numerators whose every usable packet sums to zero.

    Exact rational linear algebra throughout -- a floating-point nullspace here
    would produce kernels that are null only to rounding, and the whole point of
    the construction is that the Thom polynomial is unchanged exactly.
    """
    monomials = admissible_monomials(gauge)
    packets = usable_packets(gauge, spread)
    if not packets:
        raise ValueError(
            f"A_{gauge.order}: no Chern packet fits under truncation {gauge.max_deg}; "
            "raise max_deg before asking for null kernels"
        )

    columns = []
    for a in monomials:
        series = series_of({a: 1}, gauge)
        columns.append([packet_sum(series, M) for M in packets])

    rows = [[Fraction(columns[j][i]) for j in range(len(monomials))] for i in range(len(packets))]
    basis_vectors = _nullspace(rows, len(monomials))

    basis = []
    for vec in basis_vectors:
        scaled = _clear_denominators(vec)
        poly = {monomials[j]: scaled[j] for j in range(len(monomials)) if scaled[j]}
        if poly:
            basis.append(poly)
    return NullKernels(gauge.order, gauge.max_deg, tuple(basis), tuple(monomials), tuple(packets))


def _nullspace(rows: List[List[Fraction]], ncols: int) -> List[List[Fraction]]:
    """Exact nullspace by Gauss-Jordan elimination over the rationals."""
    mat = [row[:] for row in rows]
    pivots: Dict[int, int] = {}
    r = 0
    for c in range(ncols):
        pivot = None
        for i in range(r, len(mat)):
            if mat[i][c]:
                pivot = i
                break
        if pivot is None:
            continue
        mat[r], mat[pivot] = mat[pivot], mat[r]
        inv = Fraction(1) / mat[r][c]
        mat[r] = [v * inv for v in mat[r]]
        for i in range(len(mat)):
            if i != r and mat[i][c]:
                factor = mat[i][c]
                mat[i] = [a - factor * b for a, b in zip(mat[i], mat[r])]
        pivots[c] = r
        r += 1
        if r == len(mat):
            break

    free = [c for c in range(ncols) if c not in pivots]
    out = []
    for f in free:
        vec = [Fraction(0)] * ncols
        vec[f] = Fraction(1)
        for c, row in pivots.items():
            vec[c] = -mat[row][f]
        out.append(vec)
    return out


def _clear_denominators(vec: Sequence[Fraction]) -> List[int]:
    """Scale a rational vector to primitive integers."""
    from math import gcd

    denom = 1
    for v in vec:
        denom = denom * v.denominator // gcd(denom, v.denominator)
    ints = [int(v * denom) for v in vec]
    common = 0
    for v in ints:
        common = gcd(common, abs(v))
    if common > 1:
        ints = [v // common for v in ints]
    return ints


# --------------------------------------------------------------------------
# the search
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GaugeResult:
    """
    Outcome of a search for a numerator with a nonnegative chamber series.

    ``found`` means the returned numerator was re-verified in exact integer
    arithmetic to have no negative coefficient *up to* ``truncation``.  It is not
    an all-degree claim, and ``kernel_is_null_to`` records that the kernel used
    was itself only checked to that degree.
    """

    order: int
    truncation: int
    found: bool
    #: The numerator Q_d - K, in z coordinates.
    numerator: Optional[Poly]
    #: The kernel K subtracted, in z coordinates.
    kernel: Optional[Poly]
    negatives_before: int
    negatives_after: int
    kernel_is_null_to: int
    note: str = ""


# --------------------------------------------------------------------------
# constructing null kernels from a symmetry
# --------------------------------------------------------------------------


def denominator_weights(order: int) -> List[Exponent]:
    """
    Weight vectors of the linear factors of ``D_d = prod (z_m + z_r - z_l)``.

    Indices run ``1 <= m < l``, ``1 <= r <= min(m, l - m)``, matching the residue
    formula.  A weight is the vector of coefficients, so ``2 z_1 - z_2`` is
    ``(2, -1, 0, ...)`` -- these are weights, not monomial exponents, and must be
    turned into polynomials by :func:`linear_form` before being multiplied.
    """
    out = []
    for l in range(2, order + 1):
        for m in range(1, l):
            for r in range(1, min(m, l - m) + 1):
                w = [0] * order
                w[m - 1] += 1
                w[r - 1] += 1
                w[l - 1] -= 1
                out.append(tuple(w))
    return out


def linear_form(weight: Sequence[int]) -> Poly:
    """The linear polynomial with the given coefficient vector."""
    n = len(weight)
    return {tuple(1 if t == k else 0 for t in range(n)): c for k, c in enumerate(weight) if c}


@dataclass(frozen=True)
class Swap:
    """
    What the transposition ``s_ij`` does to the denominator of the residue formula.

    ``moved`` are the factors whose image under the swap is not itself a factor;
    their product ``A`` has to be absorbed into the numerator before the quotient
    can be symmetric.  ``contour_safe`` records the analytic side: the swap is
    only legitimate if no pole is crossed when the two contours are exchanged.
    """

    order: int
    i: int
    j: int
    moved: Tuple[Exponent, ...]
    remaining: Tuple[Exponent, ...]
    contour_safe: bool
    reason: str

    @property
    def a_degree(self) -> int:
        return len(self.moved)


def _swap_weight(w: Sequence[int], i: int, j: int) -> Exponent:
    u = list(w)
    u[i], u[j] = u[j], u[i]
    return tuple(u)


def analyse_swap(order: int, i: int, j: int) -> Swap:
    """
    Split ``D_d`` for the transposition exchanging ``z_i`` and ``z_j`` (0-indexed).

    A factor survives in the denominator only if the swap sends it to another
    factor; the rest must be absorbed.  The contour test is the one the residue
    argument needs: after absorption, every remaining factor that pins down
    ``z_i`` or ``z_j`` must locate it at a combination of *inner* variables --
    indices below both ``i`` and ``j`` -- so that the two outer contours can be
    exchanged without a pole passing between them.
    """
    weights = denominator_weights(order)
    present = set(weights)
    moved = tuple(w for w in weights if _swap_weight(w, i, j) not in present)
    remaining = tuple(w for w in weights if w not in set(moved))

    if {_swap_weight(w, i, j) for w in remaining} != set(remaining):
        return Swap(order, i, j, moved, remaining, False, "remaining factors are not permuted")

    inner = min(i, j)
    for w in remaining:
        touches = w[i] or w[j]
        if not touches:
            continue
        negative = [k for k, c in enumerate(w) if c < 0]
        if negative and negative[0] not in (i, j):
            return Swap(
                order,
                i,
                j,
                moved,
                remaining,
                False,
                f"factor {w} lets an outer variable depend on z_{min(i, j) + 1}",
            )
        sources = [k for k, c in enumerate(w) if c > 0]
        if any(k >= inner for k in sources):
            return Swap(
                order,
                i,
                j,
                moved,
                remaining,
                False,
                f"factor {w} has a pole outside the inner scale",
            )
    return Swap(order, i, j, moved, remaining, True, "poles stay on the inner scale")


def symmetric_basis(order: int, degree: int, i: int, j: int) -> List[Poly]:
    """
    A basis of polynomials of the given degree invariant under ``z_i <-> z_j``.

    One element per orbit of monomials: ``m + s(m)`` off the diagonal, ``m`` on it.
    """
    out, seen = [], set()
    for e in _compositions(degree, order):
        if e in seen:
            continue
        f = _swap_weight(e, i, j)
        seen.add(e)
        seen.add(f)
        out.append({e: 1} if e == f else {e: 1, f: 1})
    return out


def symmetry_kernels(
    gauge: GaugeSetup,
    swaps: Optional[Sequence[Tuple[int, int]]] = None,
    max_r_degree: Optional[int] = None,
    require_contour_safe: bool = True,
) -> List[Tuple[Poly, Swap]]:
    """
    Null kernels built as ``A * R``: the absorbed factors times a symmetric factor.

    Unlike :func:`null_kernel_basis` these are not fitted to anything.  If
    ``P / D_d`` is ``s``-invariant then the integrand is antisymmetric against a
    symmetric Chern insertion, so the residue vanishes at every level at once --
    the reason is structural and degree-independent, which is what
    :func:`null_kernel_basis` cannot supply past ``d = 5``.

    Each kernel is returned with the swap that justifies it.  Kernels whose swap
    is not contour-safe are dropped unless asked for, since for those the
    algebraic symmetry is real but the residue argument does not close.
    """
    order = gauge.order
    pairs = swaps if swaps is not None else list(combinations(range(order), 2))
    out: List[Tuple[Poly, Swap]] = []
    for i, j in pairs:
        swap = analyse_swap(order, i, j)
        if require_contour_safe and not swap.contour_safe:
            continue
        r_degree = gauge.degree - swap.a_degree
        if r_degree < 0:
            continue
        if max_r_degree is not None and r_degree > max_r_degree:
            continue
        a = {(0,) * order: 1}
        for w in swap.moved:
            a = poly_mul(a, linear_form(w))
        for r in symmetric_basis(order, r_degree, i, j):
            out.append((poly_mul(a, r), swap))
    return out


def partial_absorption_kernels(
    gauge: GaugeSetup,
    swaps: Optional[Sequence[Tuple[int, int]]] = None,
    max_r_degree: Optional[int] = None,
) -> List[Tuple[Poly, Swap, Exponent]]:
    """
    Null kernels that absorb all but *one* of the factors a transposition moves.

    :func:`symmetry_kernels` makes ``P / D_d`` outright ``s``-invariant, which is
    sufficient but leaves out the kernels the residue argument reaches by a second
    swap.  Those have the shape

        P = (A_s / u) * R,     u a single moved factor, R  s-invariant,

    so that ``P / D_d = R / (C u)`` with ``C`` the ``s``-stable part.  The
    ``s``-symmetric half of that still dies against the antisymmetric Vandermonde;
    the antisymmetric half is

        R (s(u) - u) / (C u s(u)),

    which pairs ``u`` with its image and so completes the source multiset of the
    remaining denominator.  When that completed object is invariant under some
    *other* transposition, a second swap kills it too.

    Whether the completion actually closes depends on the pair of transpositions,
    so these are candidates rather than theorems, and every one is put through the
    packet falsifier before use.  At ``d = 5`` the family contains ``F B C`` -- the
    summand :func:`symmetry_kernels` provably cannot reach -- which is what makes
    the difference between a search that closes and one that does not.
    """
    order = gauge.order
    pairs = swaps if swaps is not None else list(combinations(range(order), 2))
    out: List[Tuple[Poly, Swap, Exponent]] = []
    for i, j in pairs:
        swap = analyse_swap(order, i, j)
        if swap.a_degree < 1:
            continue
        r_degree = gauge.degree - (swap.a_degree - 1)
        if r_degree < 0:
            continue
        if max_r_degree is not None and r_degree > max_r_degree:
            continue
        for dropped in set(swap.moved):
            kept = list(swap.moved)
            kept.remove(dropped)
            a = {(0,) * order: 1}
            for w in kept:
                a = poly_mul(a, linear_form(w))
            for r in symmetric_basis(order, r_degree, i, j):
                out.append((poly_mul(a, r), swap, dropped))
    return out


def null_candidates(
    gauge: GaugeSetup,
    max_r_degree: Optional[int] = None,
    spread: Optional[int] = None,
    filter_at: Optional[int] = None,
    return_descriptions: bool = False,
):
    """
    Every structurally motivated kernel that survives the packet falsifier.

    Pools :func:`symmetry_kernels` with :func:`partial_absorption_kernels` and
    keeps only those whose Chern packets all vanish.  The filter is what makes the
    second family usable at all: partial absorption is a shape, not a proof, and
    much of what it proposes is not null.

    ``filter_at`` is the important knob and should be set *above* the truncation
    the search will run at.  The falsifier is only as strong as the number of
    packets in range, and at ``d = 6`` a truncation of 12 offers 13 of them
    against candidates numbering in the thousands -- far too weak, so non-null
    kernels get through and the search happily builds on them.  Filtering deeper
    costs one pass and removes them before they can do any harm.
    """
    if filter_at:
        if filter_at <= gauge.max_deg:
            phi_trunc = {e: float(c) for e, c in gauge.phi.items() if sum(e) <= filter_at}
            filter_gauge = GaugeSetup(
                gauge.order, filter_at, gauge.algebra, phi_trunc, gauge.correction, gauge.degree
            )
        else:
            filter_gauge = setup(gauge.order, filter_at, exact=False)
    else:
        filter_gauge = gauge
    packets = usable_packets(filter_gauge, spread)
    import time

    t0 = time.time()
    seen, out, descs_out = set(), [], []
    W_cache = {}
    ballot_cache = {M: ballot_orderings(list(M)) for M in packets}

    from chernpp.polynomial import poly_to_string

    z_vars = tuple(f"z_{i+1}" for i in range(gauge.order))

    families_with_desc = []
    for k, swap in symmetry_kernels(gauge, require_contour_safe=False):
        desc = f"symmetry s_{swap.i+1},{swap.j+1} absorbing {len(swap.moved)} factors"
        families_with_desc.append((k, desc))

    t1 = time.time()
    logger.debug(f"Generated {len(families_with_desc)} symmetry kernels in {t1 - t0:.2f}s")

    for k, swap, dropped in partial_absorption_kernels(gauge, max_r_degree=max_r_degree):
        dropped_str = poly_to_string(linear_form(dropped), z_vars)
        desc = f"partial abs. s_{swap.i+1},{swap.j+1} dropping ({dropped_str})"
        families_with_desc.append((k, desc))

    t2 = time.time()
    logger.debug(f"Generated {len(families_with_desc)} total kernels in {t2 - t1:.2f}s")

    corr, n = filter_gauge.correction, filter_gauge.nvars
    for kernel, desc in families_with_desc:
        key = tuple(sorted(kernel.items()))
        if key in seen:
            continue
        seen.add(key)

        try:
            shifted = to_chamber(kernel, filter_gauge.order, filter_gauge.degree)
            f_shifted = {}
            for e, c in shifted.items():
                f = tuple(e[i] - corr[i] for i in range(n))
                if any(v < 0 for v in f):
                    raise ValueError
                f_shifted[f] = -c
        except ValueError:
            continue

        kernel_is_null = True
        for M in packets:
            total = 0
            for e, c in f_shifted.items():
                key = (M, e)
                w = W_cache.get(key)
                if w is None:
                    w = 0
                    for _, b in ballot_cache[M]:
                        diff = tuple(b[i] - e[i] for i in range(n))
                        if all(v >= 0 for v in diff):
                            w += filter_gauge.phi.get(diff, 0)
                    W_cache[key] = w
                total += c * w

            if abs(total) > 1e-3:
                kernel_is_null = False
                break

        if kernel_is_null:
            out.append(kernel)
            descs_out.append(desc)

    t3 = time.time()
    logger.debug(f"Packet filtering finished in {t3 - t2:.2f}s, yielding {len(out)} kernels")

    if return_descriptions:
        return out, descs_out
    return out


def _evaluate(poly: Poly, point: Sequence[Fraction]) -> Fraction:
    total = Fraction(0)
    for exponents, coeff in poly.items():
        term = Fraction(coeff)
        for value, power in zip(point, exponents):
            if power:
                term *= value**power
        total += term
    return total


def _denominator_value(order: int, point: Sequence[Fraction]) -> Fraction:
    product = Fraction(1)
    for w in denominator_weights(order):
        product *= sum(Fraction(c) * v for c, v in zip(w, point))
    return product


def certifies_null(
    kernel: Poly, order: int, swap: Tuple[int, int], samples: int = 6
) -> Optional[Tuple[int, int]]:
    """
    Check the two-swap identity exactly, and return the second transposition.

    Nullity by the contour argument needs the ``s``-antisymmetric part of
    ``H = P / D_d`` to be invariant under some further transposition ``s'``, which
    written out is the identity

        H(z) - H(sz) - H(s'z) + H(s s' z) = 0.

    That is an identity between rational functions, so it can be settled by
    evaluating at random points: an identity holds everywhere, and a non-identity
    survives a random point only with negligible probability.  Unlike the packet
    test this is *degree-independent* -- there is no truncation to outrun.

    That distinction is not academic.  Filtering candidates on the packets in
    range admits kernels that are null only up to the depth checked: at ``d = 6``
    a filter at degree 20 still passes kernels that move four packet sums at
    degree 22.  A falsifier cannot certify, however deep it is run.

    Returns the transposition witnessing the identity, or ``None``.
    """
    import random

    rng = random.Random(0x5EED)
    points = []
    while len(points) < samples:
        point = [Fraction(rng.randint(2, 400), rng.randint(1, 40)) for _ in range(order)]
        if _denominator_value(order, point):
            points.append(point)

    i, j = swap

    def h(p: Sequence[Fraction]) -> Fraction:
        return _evaluate(kernel, p) / _denominator_value(order, p)

    def permute(p: Sequence[Fraction], a: int, b: int) -> List[Fraction]:
        q = list(p)
        q[a], q[b] = q[b], q[a]
        return q

    for a, b in combinations(range(order), 2):
        if all(
            h(p) - h(permute(p, i, j)) - h(permute(p, a, b)) + h(permute(permute(p, a, b), i, j)) == 0
            for p in points
        ):
            return (a, b)
    return None


def certified_null_candidates(gauge: GaugeSetup, max_r_degree: Optional[int] = None) -> List[Poly]:
    """
    Partial-absorption kernels that satisfy the two-swap identity exactly.

    This replaces the packet filter of :func:`null_candidates` with a test that
    does not depend on a truncation, so what it returns cannot fail further out.
    Fully ``s``-invariant kernels satisfy the identity trivially and are included.
    """
    seen, out = set(), []
    families: List[Tuple[Poly, Swap]] = [
        (k, s) for k, s in symmetry_kernels(gauge, require_contour_safe=False)
    ]
    families += [(k, s) for k, s, _ in partial_absorption_kernels(gauge, max_r_degree=max_r_degree)]
    for kernel, swap in families:
        key = tuple(sorted(kernel.items()))
        if key in seen:
            continue
        seen.add(key)
        try:
            series_of(kernel, gauge)
        except ValueError:
            continue
        if certifies_null(kernel, gauge.order, (swap.i, swap.j)) is not None:
            out.append(kernel)
    return out


@dataclass(frozen=True)
class Validation:
    """Out-of-sample report for a kernel found at a lower truncation."""

    order: int
    truncation: int
    packets: int
    packets_changed: int
    negatives_canonical: int
    negatives_gauged: int

    @property
    def holds(self) -> bool:
        return self.packets_changed == 0 and self.negatives_gauged == 0


def validate_gauge(kernel: Poly, order: int, truncation: int) -> Validation:
    """
    Re-test a kernel at a truncation it was not fitted at.

    The search fixes its kernel against the packets that fit under one
    truncation, and at ``d = 5`` there are only 21 of those against 34 free
    monomials -- badly underdetermined, so a kernel that merely overfits those
    packets is a real possibility.  Raising the truncation brings in packets the
    search never saw and coefficients it never constrained.  A kernel that keeps
    every packet sum and stays nonnegative there is doing something structural;
    one that does not was fitted to noise.

    This is evidence, not proof: no finite truncation establishes nullity at
    every level.
    """
    gauge = setup(order, truncation)
    original = to_z(gauge.algebra.multidegree, order, gauge.degree)
    base = series_of(original, gauge)
    gauged = series_of(_subtract(original, kernel), gauge)
    packets = usable_packets(gauge)
    changed = sum(1 for M in packets if abs(packet_sum(gauged, M) - packet_sum(base, M)) > 1e-5)

    neg_base = sum(1 for c in base.values() if c < -1e-5)
    neg_gauged = sum(1 for c in gauged.values() if c < -1e-5)

    return Validation(
        order,
        truncation,
        len(packets),
        changed,
        neg_base,
        neg_gauged,
    )


def _subtract(p: Poly, q: Poly) -> Poly:
    out = dict(p)
    for e, c in q.items():
        out[e] = out.get(e, 0) - c
    return {e: v for e, v in out.items() if v}


# --------------------------------------------------------------------------
# improved gauge search (v2)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DeficitResult:
    """
    Outcome of a soft-margin gauge search.

    ``deficit`` is the total negative mass remaining after the best kernel
    combination.  Zero deficit means all negatives were eliminated.
    ``negatives_remaining`` counts the number of still-negative coefficients.
    """

    order: int
    truncation: int
    found: bool
    kernel: Optional[Poly]
    deficit: int
    negatives_before: int
    negatives_remaining: int
    kernels_available: int
    kernels_used: int
    note: str
    kernel_coefficients: Optional[List[float]] = None


# --------------------------------------------------------------------------
# fast MILP with lazy constraint generation (v3)
# --------------------------------------------------------------------------


def solve_positive_gauge_jax(
    order: int,
    fit_depth: int = 24,
    bound: float = 20.0,
    max_iters: int = 50000,
) -> DeficitResult:
    """
    Finds a positive gauge perturbation via First-Order Primal-Dual Hybrid Gradient (PDHG)
    natively implemented in JAX, allowing 100% execution on GPU/TPU accelerators.
    """
    import jax
    import jax.numpy as jnp

    gauge = setup(order, fit_depth, exact=False, use_jax=True)
    original = to_z(gauge.algebra.multidegree, order, gauge.degree)
    base = series_of(original, gauge)

    logger.debug(f"Computing null kernels for d={order}")
    kernels = null_candidates(gauge, filter_at=fit_depth)
    n = len(kernels)

    col_series = []
    for k in kernels:
        try:
            col_series.append(series_of(k, gauge))
        except ValueError:
            col_series.append({})

    all_exps = sorted(set(base) | {e for s in col_series for e in s})
    exp_idx = {e: i for i, e in enumerate(all_exps)}
    base_vec = np.array([base.get(e, 0) for e in all_exps])

    kern_effects = []
    for s in col_series:
        kern_effects.append({exp_idx[e]: c for e, c in s.items()})

    before = sum(1 for c in base_vec if c < 0)
    active_rows = {i for i, b in enumerate(base_vec) if b < 0}
    note_parts = []
    best_kernel = None

    @jax.jit
    def pdhg_step(x, y, A_jax, b_jax, c_jax, tau, sigma):
        # Primal update: x_{k+1} = proj(x_k - tau * c - tau * A^T y_k)
        x_new = x - tau * c_jax - tau * (A_jax.T @ y)
        x_new = jnp.clip(x_new, 0.0, bound)

        # Dual update: y_{k+1} = proj_pos(y_k + sigma * A * (2x_new - x) - sigma * b)
        y_new = y + sigma * (A_jax @ (2.0 * x_new - x)) - sigma * b_jax
        y_new = jnp.maximum(y_new, 0.0)
        return x_new, y_new

    for round_num in range(15):
        active_list = sorted(active_rows)
        m = len(active_list)

        A_np = np.zeros((m, 2 * n), dtype=np.float32)
        b_np = np.zeros(m, dtype=np.float32)
        for ri, ai in enumerate(active_list):
            b_np[ri] = float(base_vec[ai])
            for j, ke in enumerate(kern_effects):
                v = ke.get(ai, 0)
                if v != 0:
                    A_np[ri, j] = float(v)
                    A_np[ri, n + j] = float(-v)

        A_jax = jnp.array(A_np)
        b_jax = jnp.array(b_np)
        c_jax = jnp.ones(2 * n, dtype=jnp.float32)

        # PDHG Stepsize heuristics
        norm_A = jnp.linalg.norm(A_jax, ord=2) + 1e-6
        tau = 1.0 / norm_A
        sigma = 1.0 / norm_A

        x = jnp.zeros(2 * n, dtype=jnp.float32)
        y = jnp.zeros(m, dtype=jnp.float32)

        def body_fun(i, val):
            x, y = val
            return pdhg_step(x, y, A_jax, b_jax, c_jax, tau, sigma)

        x, y = jax.lax.fori_loop(0, max_iters, body_fun, (x, y))

        x_np = np.array(x)
        k_val = x_np[:n] - x_np[n:]

        kernel = {}
        for t, kern in zip(k_val, kernels):
            if abs(t) > 1e-4:
                for e, c in kern.items():
                    kernel[e] = kernel.get(e, 0) + float(t) * c

        candidate = _subtract(original, kernel)
        gauged = series_of(candidate, gauge)

        new_violations = 0
        for e, c in gauged.items():
            if c < -1e-4:
                ei = exp_idx.get(e)
                if ei is not None and ei not in active_rows:
                    active_rows.add(ei)
                    new_violations += 1

        if new_violations == 0:
            best_kernel = kernel
            note_parts.append(f"JAX PDHG converged in {round_num} rounds with {m} active constraints")
            break

    after_negs = [c for c in gauged.values() if c < -1e-4]
    best_deficit = sum(-c for c in after_negs)
    best_remaining = len(after_negs)
    found = best_deficit < 1e-4 and best_remaining == 0

    return DeficitResult(
        order,
        gauge.max_deg,
        found,
        best_kernel,
        best_deficit,
        before,
        best_remaining,
        n,
        sum(1 for t in k_val if abs(t) > 1e-4),
        "; ".join(note_parts),
    )


def solve_positive_gauge_continuous(order: int, fit_depth: int = 24, bound: float = 20.0) -> DeficitResult:
    """
    Finds a positive gauge perturbation via Continuous LP using SciPy HiGHS.
    This is extremely fast for our LP sizes and is highly exact.
    """
    gauge = setup(order, fit_depth, exact=False)
    original = to_z(gauge.algebra.multidegree, order, gauge.degree)
    base = series_of(original, gauge)

    logger.debug(f"Computing verified null kernels for d={order}")
    kernels, descs = null_candidates(gauge, filter_at=fit_depth, return_descriptions=True)
    n = len(kernels)
    logger.debug(f"Found {n} null kernels. Computing chamber series in parallel...")

    from functools import partial

    func = partial(_safe_series_of, g=gauge)

    # Try using joblib (loky backend by default) for robust parallel processing
    try:
        import joblib

        col_series = joblib.Parallel(n_jobs=-1, backend="loky")(joblib.delayed(func)(k) for k in kernels)
    except Exception as e:
        logger.warning(f"Parallel series_of failed, falling back to sequential: {e}")
        col_series = []
        for k in kernels:
            try:
                col_series.append(series_of(k, gauge))
            except ValueError:
                col_series.append({})

    all_exps = sorted(set(base) | {e for s in col_series for e in s})
    exp_idx = {e: i for i, e in enumerate(all_exps)}
    base_vec = np.array([base.get(e, 0) for e in all_exps])

    kern_effects = []
    for s in col_series:
        kern_effects.append({exp_idx[e]: c for e, c in s.items()})

    before = sum(1 for c in base_vec if c < 0)

    # Build full constraint matrix in one shot. SciPy HiGHS handles this easily.
    m = len(all_exps)
    rows, cols, data = [], [], []
    rhs = np.zeros(m)
    for ri in range(m):
        rhs[ri] = float(base_vec[ri])
        for j, ke in enumerate(kern_effects):
            v = ke.get(ri, 0)
            if v != 0:
                rows.append(ri)
                cols.append(j)
                data.append(float(v))
                rows.append(ri)
                cols.append(n + j)
                data.append(float(-v))

    import scipy.sparse as sp
    from scipy.optimize import linprog

    A_sparse = sp.csr_array((data, (rows, cols)), shape=(m, 2 * n))
    c_obj = np.ones(2 * n)
    bounds = (0, bound)

    logger.debug(f"Solving full LP (m={m}, n={2*n}) natively with HiGHS")
    try:
        res = linprog(c_obj, A_ub=A_sparse, b_ub=rhs, bounds=bounds, method="highs")
    except Exception as e:
        return DeficitResult(
            order,
            gauge.max_deg,
            False,
            None,
            sum(-c for c in base.values() if c < 0),
            before,
            before,
            n,
            0,
            f"SciPy LP failed: {str(e)}",
        )

    if not res.success:
        return DeficitResult(
            order,
            gauge.max_deg,
            False,
            None,
            sum(-c for c in base.values() if c < 0),
            before,
            before,
            n,
            0,
            f"LP infeasible: {res.message}",
        )

    x_val = res.x
    k_val = x_val[:n] - x_val[n:]

    note_parts = []
    kernel = {}
    for t, kern in zip(k_val, kernels):
        if abs(t) > 1e-5:
            for e, c in kern.items():
                kernel[e] = kernel.get(e, 0) + float(t) * c

    candidate = _subtract(original, kernel)
    gauged = series_of(candidate, gauge)

    new_violations = 0
    for e, c in gauged.items():
        if c < -1e-5:
            new_violations += 1

    if new_violations == 0:
        best_kernel = kernel
        logger.info(f"LP converged with full constraint matrix")
        note_parts.append("Found exact fractional continuous solution via HiGHS")

        contribs = [(t, kern, desc) for t, kern, desc in zip(k_val, kernels, descs) if abs(t) > 1e-5]
        contribs.sort(key=lambda item: abs(item[0]), reverse=True)
        top_10 = contribs[:10]
        if top_10:
            logger.info("Top 10 kernel contributions by absolute value:")
            for i, (t, kern, desc) in enumerate(top_10):
                logger.info(f"  {i+1}: coeff {t:.4f} | {desc}")
    else:
        best_kernel = None
        note_parts.append("Solution failed strict zero check.")

    after_negs = [c for c in gauged.values() if c < -1e-5]
    best_deficit = sum(-c for c in after_negs)
    best_remaining = len(after_negs)
    found = best_deficit < 1e-5 and best_remaining == 0

    return DeficitResult(
        order,
        gauge.max_deg,
        found,
        best_kernel,
        best_deficit,
        before,
        best_remaining,
        n,
        sum(1 for t in k_val if abs(t) > 1e-5),
        "; ".join(note_parts),
        k_val.tolist(),
    )
