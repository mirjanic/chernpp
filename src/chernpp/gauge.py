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

The symmetry construction below is this repository's reformulation of the first
of those two swaps.  Their identity ``D_5 = B F E P(z_4) P(z_5) M(z_5) C`` makes
``B M C / D_5 = 1 / (F E P(z_4) P(z_5))`` manifestly ``s_45``-invariant, and
absorbing the moved factors ``M`` and ``C`` is exactly what :func:`analyse_swap`
computes.  Their *second* swap -- antisymmetrise, then complete the source
multiset so a different transposition applies -- is not implemented here, and
:func:`symmetry_kernels` is correspondingly incomplete: it reaches their ``M B C``
but not their ``F B C``.
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, combinations_with_replacement
from typing import Dict, List, Optional, Sequence, Tuple

from .artifacts import ChamberAlgebra, load_algebra
from .chamber import ballot_orderings
from .polynomial import Exponent, Poly, expand_rational, negative_terms, poly_mul

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
    "search_positive_gauge",
    "search_over_kernels",
    "solve_positive_gauge",
    "Solution",
    "validate_gauge",
    "Validation",
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


def setup(order: int, max_deg: int, algebra: Optional[ChamberAlgebra] = None) -> GaugeSetup:
    """Precompute the numerator-independent part of the chamber series."""
    alg = algebra or load_algebra(order)
    phi = expand_rational(alg.vandermonde, list(alg.denominator_factors), max_deg)
    corr = correction_monomial(alg)
    return GaugeSetup(order, max_deg, alg, phi, corr, _z_degree(alg))


def _z_degree(algebra: ChamberAlgebra) -> int:
    """deg Q_d as a polynomial in z, read off the stored chamber form."""
    # In chamber coordinates b_k = a_1 + ... + a_k, so the last coordinate is
    # deg - a_d <= deg, with equality exactly when a_d = 0.  Every Q_d has such a
    # monomial, so the degree is the largest last coordinate.
    return max(e[-1] for e in algebra.multidegree)


def series_of(numerator_z: Poly, gauge: GaugeSetup) -> Poly:
    """
    The chamber series of an arbitrary numerator, truncated at ``gauge.max_deg``.

    Raises if the numerator is not divisible by the chamber correction monomial.
    That is not a technicality: such a numerator produces a Laurent series with
    negative exponents rather than the power series the ballot indexing assumes,
    and silently dropping those terms would corrupt every packet sum downstream.
    """
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
    return poly_mul(shifted, gauge.phi, max_deg=gauge.max_deg)


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


def search_over_kernels(
    gauge: GaugeSetup,
    kernels: Sequence[Poly],
    spread: Optional[int] = None,
    bound: int = 8,
    time_limit: Optional[float] = None,
) -> GaugeResult:
    """
    Search a supplied list of null kernels for a combination that kills every
    negative coefficient.

    This is the counterpart of :func:`search_positive_gauge` for kernels that are
    already null for a structural reason -- typically from
    :func:`symmetry_kernels`.  Because nullity does not have to be imposed, there
    are no equality constraints and the free parameters are only as many as there
    are kernels, so the programme stays determined even at ``d = 7`` where the
    monomial search is hopeless.

    Nullity is still re-verified exactly on every usable packet.  Constructing a
    kernel from a symmetry is a reason to believe it is null, not a licence to
    skip the check.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    packets = usable_packets(gauge, spread)
    original = to_z(gauge.algebra.multidegree, gauge.order, gauge.degree)
    base = series_of(original, gauge)
    before = len(negative_terms(base))

    usable: List[Poly] = []
    series: List[Poly] = []
    for k in kernels:
        try:
            s = series_of(k, gauge)
        except ValueError:
            continue
        usable.append(k)
        series.append(s)
    if not usable:
        return GaugeResult(
            gauge.order,
            gauge.max_deg,
            False,
            None,
            None,
            before,
            before,
            gauge.max_deg,
            "no supplied kernel is an admissible numerator",
        )

    exponents = sorted(set(base) | {e for s in series for e in s})
    n = len(usable)
    ub = np.array([[float(s.get(e, 0)) for s in series] for e in exponents])
    res = milp(
        c=np.ones(2 * n),
        constraints=[
            LinearConstraint(
                np.hstack([ub, -ub]),
                -np.inf,
                np.array([float(base.get(e, 0)) for e in exponents]),
            )
        ],
        integrality=np.ones(2 * n),
        bounds=Bounds(0, bound),
        options={"time_limit": time_limit} if time_limit else {},
    )
    if res.x is None:
        return GaugeResult(
            gauge.order,
            gauge.max_deg,
            False,
            None,
            None,
            before,
            before,
            gauge.max_deg,
            f"no integer combination of {n} kernels with |t| <= {bound}: {res.message.strip()}",
        )

    coeffs = np.round(res.x[:n] - res.x[n:]).astype(int)
    kernel: Poly = {}
    for t, k in zip(coeffs, usable):
        if not t:
            continue
        for e, c in k.items():
            kernel[e] = kernel.get(e, 0) + int(t) * c
    kernel = {e: c for e, c in kernel.items() if c}
    if not kernel:
        return GaugeResult(
            gauge.order,
            gauge.max_deg,
            before == 0,
            original if before == 0 else None,
            None,
            before,
            before,
            gauge.max_deg,
            "solver returned the zero kernel",
        )

    candidate = _subtract(original, kernel)
    gauged = series_of(candidate, gauge)
    if any(packet_sum(gauged, M) != packet_sum(base, M) for M in packets):
        return GaugeResult(
            gauge.order,
            gauge.max_deg,
            False,
            None,
            None,
            before,
            before,
            gauge.max_deg,
            "combination is not null under exact recomputation",
        )
    after = len(negative_terms(gauged))
    return GaugeResult(
        gauge.order,
        gauge.max_deg,
        after == 0,
        candidate if after == 0 else None,
        kernel,
        before,
        after,
        gauge.max_deg,
        f"{int(np.count_nonzero(coeffs))} of {n} kernels used; exactly verified over "
        f"{len(exponents)} coefficients and {len(packets)} packets",
    )


def search_positive_gauge(
    gauge: GaugeSetup,
    spread: Optional[int] = None,
    bound: int = 40,
    time_limit: Optional[float] = None,
) -> GaugeResult:
    """
    Look for a null kernel ``K`` making the series of ``Q_d - K`` nonnegative.

    Posed as a mixed-integer programme in the *monomial* coordinates of the
    kernel, with nullity as equality constraints and one inequality per chamber
    monomial.  Two choices here were forced by experiment rather than taste:

    Monomial coordinates, not a basis of the null space.  A row-reduced basis of
    the null space has entries reaching ``8e9`` at ``d = 5`` alone, and the
    resulting programme is so badly scaled that its solution cannot be recovered
    exactly.  In monomial coordinates the numbers stay the size of the series
    coefficients.

    Integer variables, not a relaxation.  ``Q_d`` has integer coefficients and so
    must ``Q_d - K``.  The continuous relaxation happily returns a fractional
    vertex -- at ``d = 5`` it gives 21 nonzero coordinates around ``0.5`` -- and
    no rounding of that point is feasible.  Asking for integers directly returns
    a kernel that verifies immediately.

    The L1 objective picks a sparse kernel among the many that work, which is
    what makes the answer readable.

    Whatever the solver returns is recomputed in exact integer arithmetic, and
    reported as found only if the exact series has no negative coefficient *and*
    every exact packet sum is unchanged.  The solver is a search heuristic; the
    exact recomputation is the witness.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    monomials = admissible_monomials(gauge)
    packets = usable_packets(gauge, spread)
    if not packets:
        raise ValueError(
            f"A_{gauge.order}: no Chern packet fits under truncation {gauge.max_deg}; "
            "raise max_deg before searching for a gauge"
        )

    original = to_z(gauge.algebra.multidegree, gauge.order, gauge.degree)
    base = series_of(original, gauge)
    before = len(negative_terms(base))
    column_series = [series_of({a: 1}, gauge) for a in monomials]
    exponents = sorted(set(base) | {e for s in column_series for e in s})
    n = len(monomials)

    # k = kplus - kminus with both halves nonnegative, so that |k| is linear.
    ub = np.array([[float(s.get(e, 0)) for s in column_series] for e in exponents])
    eq = np.array([[float(packet_sum(s, M)) for s in column_series] for M in packets])
    constraints = [
        LinearConstraint(np.hstack([ub, -ub]), -np.inf, np.array([float(base.get(e, 0)) for e in exponents])),
        LinearConstraint(np.hstack([eq, -eq]), 0, 0),
    ]
    options = {"time_limit": time_limit} if time_limit else {}
    res = milp(
        c=np.ones(2 * n),
        constraints=constraints,
        integrality=np.ones(2 * n),
        bounds=Bounds(0, bound),
        options=options,
    )
    if res.x is None:
        return GaugeResult(
            gauge.order,
            gauge.max_deg,
            False,
            None,
            None,
            before,
            before,
            gauge.max_deg,
            f"no integer gauge with |k| <= {bound}: {res.message.strip()}",
        )

    raw = np.round(res.x[:n] - res.x[n:]).astype(int)
    kernel = {monomials[j]: int(raw[j]) for j in range(n) if raw[j]}
    if not kernel:
        return GaugeResult(
            gauge.order,
            gauge.max_deg,
            before == 0,
            original if before == 0 else None,
            None,
            before,
            before,
            gauge.max_deg,
            "solver returned the zero kernel",
        )

    candidate = _subtract(original, kernel)
    series = series_of(candidate, gauge)
    if any(packet_sum(series, M) != packet_sum(base, M) for M in packets):
        return GaugeResult(
            gauge.order,
            gauge.max_deg,
            False,
            None,
            None,
            before,
            before,
            gauge.max_deg,
            "solver returned a kernel that is not null under exact recomputation",
        )
    after = len(negative_terms(series))
    return GaugeResult(
        gauge.order,
        gauge.max_deg,
        after == 0,
        candidate if after == 0 else None,
        kernel,
        before,
        after,
        gauge.max_deg,
        f"exactly verified over {len(exponents)} coefficients and {len(packets)} packets",
    )


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


@dataclass(frozen=True)
class Solution:
    """
    A positive gauge that survived validation at a truncation it was not fitted at.

    ``found`` is only true when the kernel is null *and* the series is nonnegative
    at every checked truncation, the largest of which was never seen by the fit.
    """

    order: int
    found: bool
    numerator: Optional[Poly]
    kernel: Optional[Poly]
    fitted_at: int
    validated_at: Tuple[int, ...]
    kernels_available: int
    kernels_used: int
    negatives_canonical: int
    note: str


def _solve_at(
    gauge: GaugeSetup, bound: int, time_limit: Optional[float]
) -> Tuple[Optional[Poly], int, int, str]:
    """
    One mixed-integer fit at a fixed truncation.  Returns (kernel, columns, negatives, note).

    The columns are of two kinds and they are pooled in a single programme rather
    than by concatenating bases -- a row-reduced basis of the fitted null space has
    entries around ``8e9`` and poisons the conditioning, whereas as *columns* the
    same information stays the size of the series coefficients.

    Symmetry columns are null whatever the truncation.  Monomial columns are only
    admitted when the packets in range outnumber the free monomials, i.e. when
    nullity is actually determined by the data; at ``d = 6`` that test fails by a
    factor of twenty-six and admitting them would just let the solver zero the
    packets it can see.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    packets = usable_packets(gauge)
    monomials = admissible_monomials(gauge)
    columns = [k for k, _ in symmetry_kernels(gauge, require_contour_safe=False)]
    determined = len(packets) > len(monomials)
    if determined:
        columns = columns + [{a: 1} for a in monomials]

    usable, series = [], []
    for c in columns:
        try:
            s = series_of(c, gauge)
        except ValueError:
            continue
        usable.append(c)
        series.append(s)

    original = to_z(gauge.algebra.multidegree, gauge.order, gauge.degree)
    base = series_of(original, gauge)
    negatives = len(negative_terms(base))
    if not usable:
        return None, 0, negatives, "no usable column"

    exponents = sorted(set(base) | {e for s in series for e in s})
    n = len(usable)
    ub = np.array([[float(s.get(e, 0)) for s in series] for e in exponents])
    eq = np.array([[float(packet_sum(s, M)) for s in series] for M in packets])
    res = milp(
        c=np.ones(2 * n),
        constraints=[
            LinearConstraint(
                np.hstack([ub, -ub]),
                -np.inf,
                np.array([float(base.get(e, 0)) for e in exponents]),
            ),
            LinearConstraint(np.hstack([eq, -eq]), 0, 0),
        ],
        integrality=np.ones(2 * n),
        bounds=Bounds(0, bound),
        options={"time_limit": time_limit} if time_limit else {},
    )
    kind = "symmetry + monomial" if determined else "symmetry only"
    if res.x is None:
        return None, n, negatives, f"{kind}: infeasible ({res.message.strip()[:60]})"

    coeffs = np.round(res.x[:n] - res.x[n:]).astype(int)
    kernel: Poly = {}
    for t, c in zip(coeffs, usable):
        if not t:
            continue
        for e, v in c.items():
            kernel[e] = kernel.get(e, 0) + int(t) * v
    kernel = {e: v for e, v in kernel.items() if v}
    return kernel or None, n, negatives, f"{kind}, {n} columns"


def solve_positive_gauge(
    order: int,
    fit_degrees: Sequence[int] = (12, 14, 16, 18),
    validate_steps: int = 2,
    bound: int = 12,
    time_limit: Optional[float] = 900,
) -> Solution:
    """
    Find a numerator for ``A_d`` whose chamber series has no negative coefficient.

    Fits at increasing truncation and returns only an answer that *survives a
    truncation it was not fitted at*.  That escalation is the substance of the
    routine, because nullity and nonnegativity fail differently: a symmetry kernel
    stays null out of sample whatever happens, but its positivity is easily an
    artefact of where the fit stopped.  Only the deeper check separates the two,
    and a run that never validates reports the failure rather than the last thing
    the solver happened to like.
    """
    last = "no fit degree produced a candidate"
    negatives = columns = 0
    for fit in fit_degrees:
        gauge = setup(order, fit)
        kernel, columns, negatives, note = _solve_at(gauge, bound, time_limit)
        if kernel is None:
            last = f"fit at {fit}: {note}"
            continue
        checks = tuple(fit + 2 * s for s in range(1, validate_steps + 1))
        reports = [validate_gauge(kernel, order, t) for t in checks]
        if all(v.holds for v in reports):
            numerator = _subtract(to_z(gauge.algebra.multidegree, order, gauge.degree), kernel)
            return Solution(
                order,
                True,
                numerator,
                kernel,
                fit,
                checks,
                columns,
                len(kernel),
                negatives,
                f"{note}; validated at {checks}",
            )
        worst = next(v for v in reports if not v.holds)
        last = (
            f"fit at {fit} ({note}) verified in sample but failed at {worst.truncation}: "
            f"{worst.packets_changed} packet(s) moved, {worst.negatives_gauged} of "
            f"{worst.negatives_canonical} negatives left"
        )
    return Solution(order, False, None, None, fit_degrees[-1], (), columns, 0, negatives, last)


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
    changed = sum(1 for M in packets if packet_sum(gauged, M) != packet_sum(base, M))
    return Validation(
        order,
        truncation,
        len(packets),
        changed,
        len(negative_terms(base)),
        len(negative_terms(gauged)),
    )


def _subtract(p: Poly, q: Poly) -> Poly:
    out = dict(p)
    for e, c in q.items():
        out[e] = out.get(e, 0) - c
    return {e: v for e, v in out.items() if v}
