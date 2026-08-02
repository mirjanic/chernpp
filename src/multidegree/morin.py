"""
The Morin A_d model: ambient space, torus weights, reference point, chamber data.

Everything specific to the A_d family lives here, separated from the algorithms
in :mod:`multidegree.basic_equations` that compute a multidegree for it.  A different
singularity family would supply its own module of this shape; a backend that
only knows how to compute multidegrees does not need to change.

The ambient space is

    N_d = span{ q^{mr}_l : 1 <= m <= r, m + r <= l <= d }
        subset Hom(C^d, Sym^2 C^d),

where the coordinate q^{mr}_l carries T_d-weight z_m + z_r - z_l.  The reference
point eps_ref has q^{mr}_l = 1 exactly when m + r = l, and the object of
interest is the multidegree of the Borel orbit closure through it.
"""

from sage.all import *

from . import get_logger

logger = get_logger(__name__)

FAMILY = "morin-a"


def weight_indices(d):
    """
    The index set of N_d: triples (m, r, l) with 1 <= m <= r and m + r <= l <= d.

    Its size is dim N_d -- 7, 13 and 22 for d = 4, 5, 6 -- and equals the number
    of denominator factors in the residue formula.
    """
    return [(m, r, l) for l in range(2, d + 1) for m in range(1, l) for r in range(m, l) if m + r <= l]


def expected_multidegree_degree(d):
    """
    deg Q_d = dim N_d - binomial(d, 2), forced by homogeneity of the residue
    formula: 1, 3 and 7 for d = 4, 5, 6.
    """
    return len(weight_indices(d)) - binomial(d, 2)


def variable_name(m, r, l):
    return f"q_{l}_{m}_{r}"


def defect_zero_names(d):
    """
    The coordinates ``q^{mr}_{m+r}``, on which ``eps_ref`` is 1.

    These are the ones saturation removes components inside, and -- because
    degrevlex is cheapest in its last variables -- the ones to place last.
    """
    return [variable_name(m, r, l) for (m, r, l) in weight_indices(d) if m + r == l]


def multivariate_ring(base_field, names, order=None):
    """
    A polynomial ring that stays multivariate however few variables it has.

    Handed a single name, ``PolynomialRing`` returns a *univariate* ring, whose
    ``dict()`` is keyed by integers rather than exponent tuples; handed none, it
    refuses outright.  Passing the count explicitly forces the multivariate
    constructor, so ``d = 1`` and ``d = 2`` go through the same code as ``d = 7``.
    """
    names = list(names)
    kwargs = {"order": order} if order else {}
    return PolynomialRing(base_field, len(names), names or "x", **kwargs)


def groebner_order(d):
    """
    The variable order the saturation is cheapest in: by level ``l``, descending.

    Two facts about degrevlex drive this.  It is cheapest in its *last*
    variables, so the defect-zero coordinates -- the ones being saturated away --
    belong at the end.  And within each block, sorting by ``l`` descending puts
    the coordinates of the deepest level, which carry the most involved
    relations, where the order does the least work.

    The effect is not marginal.  At ``d = 7`` the twelve saturating Groebner
    bases cost 14.0 s in the order the indices are generated in and 2.7 s here,
    for the same ideal and the same answer.  Sorting by the defect ``l - m - r``
    instead gives 4.2 s, and ascending ``l`` gives 15.6 s, so it is the direction
    that matters rather than merely having a rule.
    """
    indices = weight_indices(d)
    level = {variable_name(m, r, l): l for (m, r, l) in indices}
    defect_zero = set(defect_zero_names(d))
    names = [variable_name(m, r, l) for (m, r, l) in indices]
    free = [n for n in names if n not in defect_zero]
    saturating = [n for n in names if n in defect_zero]
    # Stable sorts, so within a level the generated order is preserved.
    return sorted(free, key=level.get, reverse=True) + sorted(saturating, key=level.get, reverse=True)


def coordinate_ring(base_field, d, order=None):
    """
    The polynomial ring on N_d, together with the weight of each variable.

    The variable for q^{mr}_l is named ``q_l_m_r``; the returned dict sends it
    to its index ``(l, m, r)``, from which its weight is ``z_m + z_r - z_l``.

    ``order`` overrides the variable sequence.  That matters enormously: the
    Groebner basis at ``d = 7`` costs 0.2 s with the defect-zero variables last
    and tens of seconds with the natural order, for the same ideal.
    """
    indices = weight_indices(d)
    names = order or [variable_name(m, r, l) for (m, r, l) in indices]
    ring = multivariate_ring(base_field, names, order="degrevlex")
    index_of = {ring(variable_name(m, r, l)): (l, m, r) for (m, r, l) in indices}
    return ring, index_of


def normal_weight(variable, index_of, z):
    """The T_d-weight z_m + z_r - z_l carried by a coordinate direction."""
    l, m, r = index_of[variable]
    return z[m - 1] + z[r - 1] - z[l - 1]


def multiweight(polynomial, index_of, d):
    """
    The T_d-multiweight of a multihomogeneous polynomial, as a tuple in z.

    Raises if the polynomial is not multihomogeneous: the ideal of the orbit
    closure is T_d-invariant, so an inhomogeneous relation can only mean an
    index error in whatever produced it.
    """
    weights = set()
    for monomial in polynomial.monomials():
        vector = [0] * d
        for variable in monomial.variables():
            power = monomial.degree(variable)
            l, m, r = index_of[variable]
            vector[m - 1] += power
            vector[r - 1] += power
            vector[l - 1] -= power
        weights.add(tuple(vector))
    if len(weights) != 1:
        raise RuntimeError(f"relation {polynomial} is not multihomogeneous; weights found: {weights}")
    return weights.pop()


def random_orbit_point(d, base_field):
    """
    A random B_d-translate of eps_ref, as a dict of coordinate values.

    Used to check that a candidate ideal really does vanish on the orbit -- in
    particular that saturation has not removed too much.
    """
    ring = PolynomialRing(base_field, "E", 2 * d)
    E, F = ring.gens()[:d], ring.gens()[d:]

    unipotent = matrix(base_field, d, d)
    for i in range(d):
        unipotent[i, i] = 1
    for j in range(2, d + 1):
        for i in range(1, j):
            unipotent[i - 1, j - 1] = base_field.random_element()
    inverse = unipotent.inverse()

    torus = {}
    for i in range(1, d + 1):
        value = base_field.random_element()
        while value == 0:
            value = base_field.random_element()
        torus[i] = value

    def act(expression):
        substitution = {}
        for index in range(d):
            column = range(index + 1)
            substitution[E[index]] = sum(unipotent[i, index] * E[i] for i in column)
            substitution[F[index]] = sum(unipotent[i, index] * F[i] for i in column)
        return ring(expression).subs(substitution)

    def epsilon(k):
        return sum((E[m - 1] * F[k - m - 1] for m in range(1, k)), ring(0))

    values = {}
    for l in range(1, d + 1):
        image = sum(
            (inverse[k - 1, l - 1] * act(epsilon(k)) for k in range(1, l + 1) if inverse[k - 1, l - 1] != 0),
            ring(0),
        )
        for m in range(1, l):
            for r in range(m, l):
                if m + r <= l:
                    coefficient = image.monomial_coefficient(E[m - 1] * F[r - 1])
                    values[f"q_{l}_{m}_{r}"] = coefficient * torus[m] * torus[r] / torus[l]
    return values


# --------------------------------------------------------------------------
# Chamber coordinates
# --------------------------------------------------------------------------


def chamber_names(d):
    """
    The chamber variables ``x_1, ..., x_{d-1}``, where ``x_j = z_j / z_{j+1}``.

    Generated rather than tabulated, so every order is named on the same
    footing.  The papers write ``a, b, c, e`` for ``d = 5`` and ``a, b, c, d, e``
    for ``d = 6``; those are the same variables in the same sequence, so
    ``x_1 = a``, ``x_2 = b``, and so on.
    """
    return [f"x{i}" for i in range(1, d)]


def chamber_monomial(m, l, xs, ring):
    """z_m / z_l in the chamber variables, i.e. x_m x_{m+1} ... x_{l-1}."""
    result = ring(1)
    for k in range(m, l):
        result *= xs[k - 1]
    return result


def chamber_image(polynomial, ring, d):
    """
    Substitute ``z_i = x_i x_{i+1} ... x_{d-1}``, working on exponents directly.

    Since ``x_k`` occurs in ``z_i`` exactly when ``i <= k``, the image of
    ``z^alpha`` is ``x^mu`` with ``mu_k = alpha_1 + ... + alpha_k``: the
    substitution is a partial-sum map on exponent vectors.

    Doing it this way rather than through ``ring(str(poly.subs(...)))`` is not
    an optimisation but a necessity -- at ``d = 7`` the substituted polynomial
    has 14586 terms and Sage's string parser exceeds its recursion limit.
    """
    terms = {}
    for exponents, coefficient in polynomial.dict().items():
        partial, running = [], 0
        for k in range(d - 1):
            running += int(exponents[k])
            partial.append(running)
        key = tuple(partial)
        terms[key] = terms.get(key, 0) + coefficient
    return ring({k: v for k, v in terms.items() if v != 0})


def to_python(polynomial, characteristic):
    """Sage polynomial -> plain dict of int exponent tuples to int coefficients."""
    half = characteristic // 2 if characteristic else 0
    result = {}
    for exponents, coefficient in polynomial.dict().items():
        value = int(coefficient)
        if characteristic and value > half:
            value -= characteristic
        result[tuple(int(e) for e in exponents)] = value
    return result


def chamber_algebra(d, multidegree_poly, weight_ring, base_field, characteristic):
    """
    Rewrite the residue formula in chamber coordinates and package the artifact.

    In the coordinates x_j = z_j / z_{j+1}, each Vandermonde factor (z_l - z_m)
    becomes z_l (1 - z_m/z_l) and each denominator factor (z_m + z_r - z_l)
    becomes -z_l (1 - z_m/z_l - z_r/z_l).  The leftover powers of z_l and the
    overall sign are bookkeeping and must cancel exactly, which the division
    below asserts.

    The chamber series is ``numerator / prod_r (1 - denominator_factors[r])``.
    """
    names = chamber_names(d)
    ring = multivariate_ring(base_field, names)
    xs = ring.gens()
    multidegree_chamber = chamber_image(multidegree_poly, ring, d)

    indices = weight_indices(d)
    vandermonde_factors = [1 - chamber_monomial(m, l, xs, ring) for l in range(1, d + 1) for m in range(1, l)]
    denominator_factors = [
        chamber_monomial(m, l, xs, ring) + chamber_monomial(r, l, xs, ring) for (m, r, l) in indices
    ]
    vandermonde = prod(vandermonde_factors, ring(1))

    per_level = {}
    for _, _, l in indices:
        per_level[l] = per_level.get(l, 0) + 1
    numerator_correction, denominator_correction = ring(1), ring(1)
    for l in range(2, d):
        exponent = (l - 1) - per_level.get(l, 0)
        if exponent == 0:
            continue
        monomial = chamber_monomial(l, d, xs, ring)
        if exponent > 0:
            numerator_correction *= monomial**exponent
        else:
            denominator_correction *= monomial ** (-exponent)

    sign = -1 if (len(vandermonde_factors) - len(indices)) % 2 else 1
    combined = ring(sign) * multidegree_chamber * numerator_correction

    normalized = combined // denominator_correction
    if normalized * denominator_correction != combined:
        raise RuntimeError(f"A_{d}: chamber correction failed to divide exactly")

    constant = int(normalized.constant_coefficient())
    if characteristic and constant > characteristic // 2:
        constant -= characteristic
    if constant == 0:
        raise RuntimeError(
            f"A_{d}: chamber numerator has no constant term, so the c_1^{d} coefficient is lost"
        )
    if constant < 0:
        normalized, multidegree_chamber, constant = (
            -normalized,
            -multidegree_chamber,
            -constant,
        )
    if constant != 1:
        normalized /= base_field(constant)
        multidegree_chamber /= base_field(constant)

    return {
        "order": d,
        "field": str(base_field),
        "characteristic": characteristic,
        "chamber_vars": tuple(names),
        "numerator": to_python(vandermonde * normalized, characteristic),
        "denominator_factors": [to_python(f, characteristic) for f in denominator_factors],
        "multidegree": to_python(multidegree_chamber, characteristic),
        "normalized_numerator": to_python(normalized, characteristic),
        "vandermonde": to_python(vandermonde, characteristic),
    }
