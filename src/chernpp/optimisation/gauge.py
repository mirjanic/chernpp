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


class FastChamberMultiplier:
    def __init__(self, gauge):
        import numpy as np
        from .jax_polynomial import get_exponent_mapping, precompute_shifts

        self.gauge = gauge
        self.max_deg = gauge.max_deg
        self.nvars = gauge.nvars

        self.exps_arr, self.base, self.sorted_keys, self.sorted_keys_idx = get_exponent_mapping(
            self.nvars, self.max_deg
        )
        self.degrees = self.exps_arr.sum(axis=1)
        self.flat_keys = self.exps_arr.astype(np.int64).dot(self.base.astype(np.int64))
        self.N = len(self.exps_arr)

        # dense phi
        self.phi_arr = np.zeros(self.N + 1, dtype=np.float64)
        phi_exps = []
        phi_coeffs = []
        for e, c in gauge.phi.items():
            phi_exps.append(tuple(e) + (0,) * (self.nvars - len(e)))
            phi_coeffs.append(float(c))

        if phi_exps:
            k = np.array(phi_exps).dot(self.base)
            p = np.searchsorted(self.sorted_keys, k)
            # handle out of bounds gracefully
            p = np.minimum(p, len(self.sorted_keys) - 1)
            valid = self.sorted_keys[p] == k
            self.phi_arr[self.sorted_keys_idx[p[valid]]] = np.array(phi_coeffs, dtype=np.float64)[valid]

        self.b = self.max_deg + 1
        max_key = int(self.b**self.nvars)
        self.direct_lookup = None
        if max_key < 5 * 10**8:
            import logging

            logging.getLogger(__name__).debug(
                f"Allocating {max_key * 4 / 10**6:.1f} MB direct lookup array for fast shifts"
            )
            self.direct_lookup = np.full(max_key, -1, dtype=np.int32)
            self.direct_lookup[self.sorted_keys] = self.sorted_keys_idx

        self.shift_maps = {}

    def get_shift_map(self, shift_e):
        # Cached per instance rather than with lru_cache on the method: that
        # keys on ``self`` and keeps every multiplier it has ever seen alive.
        import numpy as np

        cached = self.shift_maps.get(shift_e)
        if cached is not None:
            return cached
        pad_e = tuple(shift_e) + (0,) * (self.nvars - len(shift_e))
        shift_arr = np.array(pad_e, dtype=np.int64)

        deg_mask = self.degrees + shift_arr.sum() <= self.max_deg
        shift_key = shift_arr.dot(self.base.astype(np.int64))

        valid_keys = self.flat_keys[deg_mask] + shift_key

        if self.direct_lookup is not None:
            original_indices = self.direct_lookup[valid_keys]
            valid_idx = original_indices != -1
            original_indices = original_indices[valid_idx]
            src_indices = np.where(deg_mask)[0][valid_idx]
            out = (original_indices.astype(np.int32, copy=False), src_indices.astype(np.int32, copy=False))
            self.shift_maps[shift_e] = out
            return out
        else:
            pos = np.searchsorted(self.sorted_keys, valid_keys)
            valid_idx = pos < len(self.sorted_keys)
            valid_idx[valid_idx] = self.sorted_keys[pos[valid_idx]] == valid_keys[valid_idx]

            original_indices = self.sorted_keys_idx[pos[valid_idx]]
            src_indices = np.where(deg_mask)[0][valid_idx]
            out = (original_indices.astype(np.int32, copy=False), src_indices.astype(np.int32, copy=False))
            self.shift_maps[shift_e] = out
            return out

    def multiply(self, numerator_z):
        import numpy as np

        # Inadmissibility is the one expected failure and is reported as None;
        # anything else is a bug and must not be turned into a quiet wrong answer.
        try:
            chamber = to_chamber(numerator_z, self.gauge.order, self.gauge.degree)
        except ValueError:
            return None
        corr = self.gauge.correction
        n = self.gauge.nvars

        res_arr = np.zeros(self.N + 1, dtype=np.float64)
        valid = True
        for e, c in chamber.items():
            f = tuple(e[i] - corr[i] for i in range(n))
            if any(v < 0 for v in f):
                valid = False
                break
            dest_idx, src_idx = self.get_shift_map(f)
            # -c because shifted logic negates chamber
            res_arr[dest_idx] += (-float(c)) * self.phi_arr[: self.N][src_idx]

        if not valid:
            return None
        return res_arr[: self.N]


from fractions import Fraction
from itertools import combinations, combinations_with_replacement
import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple
import logging

logger = logging.getLogger(__name__)

from ..artifacts import ChamberAlgebra, load_algebra
from ..chamber import ballot_orderings
from ..polynomial import Exponent, Poly, expand_rational, poly_mul

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


def _is_admissible(numerator_z: Poly, gauge: GaugeSetup) -> bool:
    try:
        chamber = to_chamber(numerator_z, gauge.order, gauge.degree)
    except ValueError:
        return False
    corr, n = gauge.correction, gauge.nvars
    for e in chamber.keys():
        if any(e[i] - corr[i] < 0 for i in range(n)):
            return False
    return True


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


def packet_sum(series: Poly, multiset: Sequence[int]):
    """
    Sum of the series over the ballot orderings of one multiset.  No truncation check.

    Exact whenever the series is: the return type follows the coefficients, so an
    integer series gives an integer and a rational one a ``Fraction``.
    """
    return sum(series.get(b, 0) for _, b in ballot_orderings(list(multiset)))


#: Above this magnitude float64 stops representing integers exactly, and every
#: comparison built on it silently becomes a guess.
FLOAT_INT_CEILING = 2**53


def _guard_float_exactness(values, context: str) -> None:
    """
    Refuse to draw exact conclusions from floats that may have lost integer precision.

    The chamber coefficients are integers, and float64 holds integers exactly below
    ``2**53``.  Within that range ``== 0`` on a float is a genuine equality test, not
    a tolerance, which is what lets the fast path stay both fast and exact.  Past it
    the arithmetic degrades silently, so this raises rather than letting a rounded
    packet sum pass for a vanishing one -- the same posture as the ``int64`` overflow
    guard in the Chern evaluator.
    """
    if len(values) == 0:
        return
    worst = float(np.max(np.abs(np.asarray(values, dtype=np.float64))))
    if worst >= FLOAT_INT_CEILING:
        raise OverflowError(
            f"{context}: coefficient magnitude {worst:.3e} reaches the float64 integer "
            f"ceiling {float(FLOAT_INT_CEILING):.3e}; exact comparisons are no longer "
            "valid here, so recompute in exact arithmetic instead of trusting this"
        )


def _as_exact(poly: Poly, what: str) -> Dict[Exponent, Fraction]:
    """
    Coefficients as exact rationals, refusing floats that are not whole numbers.

    A float that is not an integer carries no record of the rational it came from,
    and guessing one with ``limit_denominator`` would silently substitute a
    different kernel for the one being checked.  Callers holding a fractional
    solution should carry it as ``Fraction`` from the point it was produced.
    """
    out: Dict[Exponent, Fraction] = {}
    for e, c in poly.items():
        if isinstance(c, float):
            if not c.is_integer():
                raise ValueError(
                    f"{what}: coefficient {c!r} is a non-integer float, so this check "
                    "cannot be made exact; carry the solution as Fraction instead"
                )
            c = int(c)
        value = Fraction(c)
        if value:
            out[e] = value
    return out


def _subtract(p: Poly, q: Poly) -> Poly:
    """``p - q``, dropping cancelled terms."""
    out = dict(p)
    for e, c in q.items():
        out[e] = out.get(e, 0) - c
    return {e: v for e, v in out.items() if v}


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

        _guard_float_exactness(
            list(filter_gauge.phi.values()), f"A_{filter_gauge.order} Phi at depth {filter_at}"
        )
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

            # A packet sum is an integer that must vanish exactly.  Everything
            # here is integral and guarded below the float64 integer ceiling, so
            # this is a genuine equality rather than a tolerance -- a kernel that
            # shifts a packet by one unit is not null, however small one unit
            # looks beside coefficients of size 1e6.
            if total != 0:
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


def certified_null_candidates(
    gauge: GaugeSetup, max_r_degree: Optional[int] = None, return_descriptions: bool = False
):
    """
    Partial-absorption kernels that satisfy the two-swap identity exactly.

    This replaces the packet filter of :func:`null_candidates` with a test that
    does not depend on a truncation, so what it returns cannot fail further out.
    Fully ``s``-invariant kernels satisfy the identity trivially and are included.
    """
    seen, out, descs_out = set(), [], []
    from chernpp.polynomial import poly_to_string

    z_vars = tuple(f"z_{i+1}" for i in range(gauge.order))

    families_with_desc = []
    for k, swap in symmetry_kernels(gauge, require_contour_safe=False):
        desc = f"symmetry s_{swap.i+1},{swap.j+1} absorbing {len(swap.moved)} factors"
        families_with_desc.append((k, swap, desc))

    for k, swap, dropped in partial_absorption_kernels(gauge, max_r_degree=max_r_degree):
        dropped_str = poly_to_string(linear_form(dropped), z_vars)
        desc = f"partial abs. s_{swap.i+1},{swap.j+1} dropping ({dropped_str})"
        families_with_desc.append((k, swap, desc))

    for kernel, swap, desc in families_with_desc:
        key = tuple(sorted(kernel.items()))
        if key in seen:
            continue
        seen.add(key)
        if not _is_admissible(kernel, gauge):
            continue
        if certifies_null(kernel, gauge.order, (swap.i, swap.j)) is not None:
            out.append(kernel)
            descs_out.append(desc)

    if return_descriptions:
        return out, descs_out
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
    every level.  What it is not is approximate.  Packet sums are quantities that
    must vanish *exactly*, and a kernel that shifts one by a single unit is not
    null however small that unit looks beside coefficients of size ``1e6``.  Both
    branches below therefore compare exactly; they differ only in how they can
    afford to.
    """
    exact_kernel = _as_exact(kernel, "kernel passed to validate_gauge")
    integral = all(v.denominator == 1 for v in exact_kernel.values())

    if integral:
        # Every coefficient in sight is an integer, and float64 holds those
        # exactly below the guarded ceiling, so the fast path loses nothing.
        gauge = setup(order, truncation, exact=False)
        original = to_z(gauge.algebra.multidegree, order, gauge.degree)
        multiplier = FastChamberMultiplier(gauge)
        base_arr = multiplier.multiply(original)
        kernel_arr = multiplier.multiply({e: int(v) for e, v in exact_kernel.items()})
        if base_arr is None:
            raise ValueError(f"A_{order}: the canonical numerator is not admissible")
        if kernel_arr is None:
            raise ValueError(f"A_{order}: the kernel is not an admissible numerator")
        gauged_arr = base_arr - kernel_arr
        _guard_float_exactness(base_arr, f"A_{order} canonical series at depth {truncation}")
        _guard_float_exactness(gauged_arr, f"A_{order} gauged series at depth {truncation}")

        neg_base = int(np.sum(base_arr < 0))
        neg_gauged = int(np.sum(gauged_arr < 0))
        exps = multiplier.exps_arr
        base = {tuple(int(x) for x in exps[i]): int(base_arr[i]) for i in np.nonzero(base_arr)[0]}
        gauged = {tuple(int(x) for x in exps[i]): int(gauged_arr[i]) for i in np.nonzero(gauged_arr)[0]}
    else:
        # A fractional gauge cannot be checked in floats at all, so pay for exact
        # rational arithmetic rather than reporting a rounded verdict.
        gauge = setup(order, truncation, exact=True)
        original = to_z(gauge.algebra.multidegree, order, gauge.degree)
        base = series_of(original, gauge, exact=True)
        gauged = series_of(_subtract(original, exact_kernel), gauge, exact=True)
        neg_base = sum(1 for c in base.values() if c < 0)
        neg_gauged = sum(1 for c in gauged.values() if c < 0)

    packets = usable_packets(gauge)
    changed = sum(1 for M in packets if packet_sum(gauged, M) != packet_sum(base, M))

    return Validation(order, truncation, len(packets), changed, neg_base, neg_gauged)


# --------------------------------------------------------------------------
# improved gauge search (v2)
# --------------------------------------------------------------------------


def _rationalise_kernel(
    coefficients: Sequence[float], kernels: Sequence[Poly], max_denominator: int = 10**6
) -> Optional[Poly]:
    """
    Recover the LP's fractional solution as exact rationals, or admit defeat.

    The constraint matrix has integer entries, so every vertex of the feasible
    polyhedron is rational -- but its denominators divide integer subdeterminants
    and can be large, and a float coordinate carries no record of which rational
    produced it.  We therefore accept a candidate only if it reproduces the float
    to within its own representation error, and return ``None`` otherwise rather
    than substituting a nearby kernel for the one the solver actually found.

    ``None`` is not a failure of the search: it says the solution needs exact
    vertex extraction from the active constraint set, which this does not attempt.
    """
    exact: Dict[Exponent, Fraction] = {}
    for value, kernel in zip(coefficients, kernels):
        t = float(value)
        if abs(t) <= 1e-12:
            continue
        approx = Fraction(t).limit_denominator(max_denominator)
        # Accept only if the rational is the float, to float precision.  A vertex
        # with a genuinely ugly denominator fails here, which is the point.
        if abs(float(approx) - t) > 1e-9 * max(1.0, abs(t)):
            return None
        for e, c in kernel.items():
            exact[e] = exact.get(e, Fraction(0)) + approx * Fraction(c)
    return {e: c for e, c in exact.items() if c} or None


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


def solve_positive_gauge_continuous(order: int, fit_depth: int = 24, bound: float = 20.0) -> DeficitResult:
    """
    Finds a positive gauge perturbation via Continuous LP using SciPy HiGHS.
    This is extremely fast for our LP sizes and is highly exact.
    """
    gauge = setup(order, fit_depth, exact=False, use_jax=True)
    original = to_z(gauge.algebra.multidegree, order, gauge.degree)

    logger.debug(f"Computing verified null kernels for d={order}")
    kernels, descs = null_candidates(gauge, filter_at=fit_depth, return_descriptions=True)
    n = len(kernels)

    logger.debug(f"Found {n} null kernels. Fast-multiplying chamber series via FastChamberMultiplier...")

    multiplier = FastChamberMultiplier(gauge)

    import numpy as np

    base = multiplier.multiply(original)
    if base is None:
        from .jax_polynomial import get_exponent_mapping

        exps_arr, _, _, _ = get_exponent_mapping(gauge.nvars, gauge.max_deg)
        base = np.zeros(len(exps_arr), dtype=np.float64)

    base_vec = base
    m = len(base_vec)
    before = np.sum(base_vec < -1e-12)

    indices_list = []
    data_list = []
    indptr = [0]

    for k in range(len(kernels)):
        k_arr = multiplier.multiply(kernels[k])
        if k_arr is not None:
            mask = np.abs(k_arr) > 1e-12
            nnz_rows = np.nonzero(mask)[0].astype(np.int32)
            indices_list.append(nnz_rows)
            data_list.append(k_arr[nnz_rows].astype(np.float64))
            indptr.append(indptr[-1] + len(nnz_rows))
        else:
            indptr.append(indptr[-1])

    rhs = np.array(base_vec, dtype=np.float64)

    total_nnz = indptr[-1]
    indices = np.empty(total_nnz, dtype=np.int32)
    data = np.empty(total_nnz, dtype=np.float64)
    ptr = 0
    # pop to avoid 2x memory usage spike during concatenate
    while indices_list:
        arr_idx = indices_list.pop(0)
        arr_dat = data_list.pop(0)
        l = len(arr_idx)
        indices[ptr : ptr + l] = arr_idx
        data[ptr : ptr + l] = arr_dat
        ptr += l
    del indices_list, data_list
    indptr = np.array(indptr, dtype=np.int32)

    active_row_mask = np.zeros(m, dtype=bool)
    active_row_mask[indices] = True

    inactive_mask = ~active_row_mask

    if np.any(rhs[inactive_mask] < -1e-12):
        return DeficitResult(
            order,
            gauge.max_deg,
            False,
            None,
            float(np.sum(-base_vec[base_vec < 0])),
            before,
            before,
            n,
            0,
            "LP trivially infeasible: negative deficit on exponent with no kernels",
        )

    active_rows = np.where(active_row_mask)[0]
    m_active = len(active_rows)
    rhs_active = rhs[active_rows]

    # Map indices in-place to save memory
    row_map = np.full(m, -1, dtype=np.int32)
    row_map[active_rows] = np.arange(m_active, dtype=np.int32)
    np.take(row_map, indices, out=indices)
    del row_map

    import scipy.sparse as sp
    import highspy
    import gc

    A_sparse_full = sp.csc_array((data, indices, indptr), shape=(m_active, n))
    del data, indices, indptr
    A_sparse_half = A_sparse_full.tocsr()
    del A_sparse_full
    gc.collect()

    logger.debug(f"Solving full LP (m={m_active}, n={2*n}) natively with highspy active set (original m={m})")

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("primal_feasibility_tolerance", 1e-9)

    num_vars = 2 * n
    h.addVars(num_vars, np.zeros(num_vars), np.full(num_vars, bound))
    h.changeObjectiveSense(highspy.ObjSense.kMinimize)
    h.changeColsCost(num_vars, np.arange(num_vars, dtype=np.int32), np.ones(num_vars))

    initial_rows = np.where(rhs_active < -1e-8)[0]
    if len(initial_rows) == 0:
        initial_rows = np.arange(min(100, m_active))

    active_row_indices = initial_rows.copy()
    active_row_mask = np.zeros(m_active, dtype=bool)
    active_row_mask[active_row_indices] = True

    A_sub_half = A_sparse_half[active_row_indices, :]
    A_sub = sp.hstack([A_sub_half, -A_sub_half]).tocsr()
    b_sub = rhs_active[active_row_indices]

    num_sub_rows = A_sub.shape[0]
    h.addRows(
        num_sub_rows,
        np.full(num_sub_rows, -highspy.kHighsInf),
        b_sub,
        A_sub.nnz,
        A_sub.indptr,
        A_sub.indices,
        A_sub.data,
    )

    max_iters = 50
    x_opt = None
    for it in range(max_iters):
        h.run()
        status = h.getModelStatus()

        if status in (highspy.HighsModelStatus.kInfeasible, highspy.HighsModelStatus.kModelError):
            return DeficitResult(
                order,
                gauge.max_deg,
                False,
                None,
                float(np.sum(-base_vec[base_vec < 0])),
                before,
                before,
                n,
                0,
                f"LP infeasible or error: status {status}",
            )

        sol = h.getSolution()
        x_opt = np.array(sol.col_value)

        k_val = x_opt[:n] - x_opt[n:]
        Ax = A_sparse_half @ k_val
        violations = Ax - rhs_active

        violated_idx = np.where(violations > 1e-7)[0]
        new_violations = violated_idx[~active_row_mask[violated_idx]]

        if len(new_violations) == 0:
            active_violations = violations[active_row_mask]
            max_active_violation = np.max(active_violations) if len(active_violations) > 0 else 0
            logger.debug(
                f"Row generation converged in {it+1} iterations. Max active violation: {max_active_violation}"
            )
            break

        if len(new_violations) > 5000:
            worst_indices = np.argsort(violations[new_violations])[-5000:]
            new_violations = new_violations[worst_indices]

        A_new_half = A_sparse_half[new_violations, :]
        A_new = sp.hstack([A_new_half, -A_new_half]).tocsr()
        b_new = rhs_active[new_violations]
        num_new = A_new.shape[0]
        h.addRows(
            num_new,
            np.full(num_new, -highspy.kHighsInf),
            b_new,
            A_new.nnz,
            A_new.indptr,
            A_new.indices,
            A_new.data,
        )

        active_row_indices = np.concatenate([active_row_indices, new_violations])
        active_row_mask[new_violations] = True
        logger.debug(
            f"Row gen iter {it+1}: added {len(new_violations)} constraints, total {len(active_row_indices)}."
        )

        gc.collect()
    else:
        logger.warning(
            f"Row generation hit max iterations ({max_iters}). Output might be slightly inaccurate."
        )

    x_val = x_opt
    k_val = x_val[:n] - x_val[n:]

    note_parts = []
    kernel = {}
    for t, kern in zip(k_val, kernels):
        for e, c in kern.items():
            kernel[e] = kernel.get(e, 0) + float(t) * c

    gauged_arr = np.array(base_vec, dtype=np.float64)
    for t, kern in zip(k_val, kernels):
        if abs(t) > 1e-12:
            k_arr = multiplier.multiply(kern)
            if k_arr is not None:
                gauged_arr -= t * k_arr

    if gauged_arr is not None:
        after_negs = [float(c) for c in gauged_arr[gauged_arr < -1e-5]]
    else:
        after_negs = []

    new_violations = len(after_negs)

    if new_violations == 0:
        # The float solve is a search heuristic; it is not the witness.  Rebuild
        # the kernel over the rationals and recompute the gauged series exactly,
        # because the LP returns fractional coefficients and no tolerance on a
        # rational coefficient can tell a small negative from a zero.
        exact_kernel = _rationalise_kernel(k_val, kernels)
        if exact_kernel is None:
            best_kernel = None
            note_parts.append(
                "LP solution could not be recovered as exact rationals; reporting "
                "unverified rather than trusting the float solve"
            )
        else:
            check = validate_gauge(exact_kernel, order, gauge.max_deg)
            if check.packets_changed == 0 and check.negatives_gauged == 0:
                best_kernel = exact_kernel
                note_parts.append(
                    f"verified exactly at depth {gauge.max_deg}: "
                    f"{check.packets} packet sums unchanged, no negative coefficient"
                )
            else:
                best_kernel = None
                after_negs = [-1.0] * max(check.negatives_gauged, 1)
                note_parts.append(
                    f"float solve looked clean but exact recomputation disagrees: "
                    f"{check.packets_changed} packet sum(s) moved, "
                    f"{check.negatives_gauged} negative coefficient(s)"
                )
        logger.info("LP converged with full constraint matrix")

        contribs = [(t, kern, desc) for t, kern, desc in zip(k_val, kernels, descs)]
        contribs.sort(key=lambda item: abs(item[0]), reverse=True)
        top_10 = contribs[:10]
        if top_10:
            logger.info("Top 10 kernel contributions by absolute value:")
            for i, (t, kern, desc) in enumerate(top_10):
                logger.info(f"  {i+1}: coeff {t:.4f} | {desc}")
    else:
        best_kernel = None
        min_val = np.min(gauged_arr) if gauged_arr is not None else 0
        note_parts.append(f"Solution failed strict zero check. Worst negative: {min_val:.2e}")
        if gauged_arr is not None:
            worst = np.sort(gauged_arr[gauged_arr < -1e-5])[:10]
            worst_str = ", ".join(f"{v:.2e}" for v in worst)
            note_parts.append(f"Largest violations: [{worst_str}]")

    best_deficit = float(sum(-c for c in after_negs))
    best_remaining = len(after_negs)
    # ``found`` means exactly recomputed and verified, never merely within a
    # tolerance of the float solve.  best_kernel is only set above once the
    # rational recomputation has agreed.
    found = best_kernel is not None

    return DeficitResult(
        order,
        gauge.max_deg,
        found,
        best_kernel,
        best_deficit,
        before,
        best_remaining,
        n,
        len(k_val),
        "; ".join(note_parts),
        k_val.tolist(),
    )
