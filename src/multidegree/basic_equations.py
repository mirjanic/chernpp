"""
Multidegree from the explicit defining equations of the orbit closure.

For the Morin family this is the route of Bérczi--Szenes, Annals 175 (2012),
Prop. 7.3.  Writing u^l_{mr} for the coordinate dual to q^{mr}_l, for every
quadruple of positive integers (i, j, m, l) with i + j + m <= l <= d the three
sums

    T_i = sum_{s=j+m}^{l-i} u^s_{jm} u^l_{is}
    T_j = sum_{s=i+m}^{l-j} u^s_{im} u^l_{js}
    T_m = sum_{s=i+j}^{l-m} u^s_{ij} u^l_{ms}

agree on the orbit closure.  They are symmetric in i, j, m, so it suffices to
enumerate i <= j <= m and emit the two differences.

Their common zero locus contains the orbit closure but, from d = 6 on, also has
other components of the same dimension (loc. cit. Section 7.4).  Since eps_ref
has every defect-zero coordinate u^{m+r}_{mr} equal to 1, the orbit closure lies
in no defect-zero coordinate hyperplane; saturating by those coordinates removes
the spurious components.

The multidegree is then read off an initial ideal.  Passing to the initial ideal
of a Groebner basis is a flat degeneration, so it preserves the equivariant
Poincare dual, and for a monomial ideal that dual is the sum over
top-dimensional components of the multiplicity times the product of the normal
weights.
"""

from sage.all import *

from . import Multidegree, get_logger, morin
from .monomial import minimal_transversals, restrict_generators, standard_monomial_count

logger = get_logger(__name__)


def basic_equations(ring, d):
    """The relations of Proposition 7.3, deduplicated and with zeros dropped."""

    def u(l, a, b):
        m, r = (a, b) if a <= b else (b, a)
        return ring(0) if m + r > l else ring(f"q_{l}_{m}_{r}")

    relations = []
    for l in range(3, d + 1):
        for i in range(1, l):
            for j in range(i, l):
                for m in range(j, l):
                    if i + j + m > l:
                        continue
                    partial = [
                        sum(
                            (u(s, j, m) * u(l, i, s) for s in range(j + m, l - i + 1)),
                            ring(0),
                        ),
                        sum(
                            (u(s, i, m) * u(l, j, s) for s in range(i + m, l - j + 1)),
                            ring(0),
                        ),
                        sum(
                            (u(s, i, j) * u(l, m, s) for s in range(i + j, l - m + 1)),
                            ring(0),
                        ),
                    ]
                    for lhs, rhs in ((partial[0], partial[1]), (partial[1], partial[2])):
                        relation = lhs - rhs
                        if relation != 0 and relation not in relations:
                            relations.append(relation)
    return relations


#: Singular's `std` beats `slimgb` by one to two orders of magnitude on these
#: ideals, and produces a smaller basis.
ALGORITHM = "libsingular:std"

#: Recorded in each artifact, so a file says which route produced it.
NAME = "basic-equations"


def _strip_last_variable(polynomial, index, ring):
    """Divide out the top power of variable ``index``, by exponent surgery."""
    terms = polynomial.dict()
    if not terms:
        return polynomial
    power = min(exponents[index] for exponents in terms)
    if power == 0:
        return polynomial
    return ring(
        {
            tuple(e - (power if i == index else 0) for i, e in enumerate(exponents)): c
            for exponents, c in terms.items()
        }
    )


def reindex(polynomial, source_names, target_ring, target_names):
    """
    Rewrite a polynomial into a ring whose variables are a permutation of ``source_names``.

    Every saturation step permutes the variable order, and the obvious way to
    move a generator across is ``target_ring(str(polynomial))``.  That is both
    slow and fragile: at ``d = 7`` some orders produce generators whose printed
    form exceeds Python's recursion limit in Sage's parser, which is why several
    promising variable orders looked like errors rather than like timings.
    Permuting exponent vectors is exact, allocation-free and cannot overflow a
    stack.
    """
    lookup = {name: position for position, name in enumerate(source_names)}
    permutation = [lookup[name] for name in target_names]
    terms = {}
    for exponents, coefficient in polynomial.dict().items():
        terms[tuple(exponents[i] for i in permutation)] = coefficient
    return target_ring(terms)


def saturate_by_variables(generators, names, variables, base_field):
    """
    ``I : (prod variables)^infinity`` for a homogeneous ideal, one variable at a time.

    Degrevlex trick: with ``v`` the *last* variable, a Groebner basis of ``I``
    saturates by ``v`` on dividing each element by the largest power of ``v``
    that divides it.  That replaces an elimination -- which is what Sage's
    generic ``saturation`` does, and which does not terminate at ``d = 7`` --
    with a single basis computation.

    The order matters twice over.  The basis is cheap when the defect-zero
    variables sit last, and the trick then demands moving one of them to the
    very end.  Doing that **back-to-front** keeps each move a small perturbation
    of the cheap order; front-to-back does not, and at ``d = 7`` the first step
    alone fails to finish.  Same trick, same mathematics, seconds against hours.
    """
    for name in reversed(list(variables)):
        order = [n for n in names if n != name] + [name]
        ring = PolynomialRing(base_field, order, order="degrevlex")
        moved = [reindex(g, names, ring, order) for g in generators]
        basis = ring.ideal(moved).groebner_basis(algorithm=ALGORITHM)
        last = len(ring.gens()) - 1
        generators = [_strip_last_variable(g, last, ring) for g in basis]
        names = order
    return generators, names


def orbit_ideal(base_field, d):
    """
    The ideal of the orbit closure, with three self-checks.

    Returns ``(ideal, ring, index_of)``.  Raises rather than returning anything
    it cannot justify: every relation must be multihomogeneous for the torus
    weights, and both the raw and the saturated ideal must vanish on random
    points of the orbit.
    """
    cheap = morin.groebner_order(d)
    defect_zero = [n for n in cheap if n in set(morin.defect_zero_names(d))]

    ring, index_of = morin.coordinate_ring(base_field, d, order=cheap)
    relations = basic_equations(ring, d)
    for relation in relations:
        morin.multiweight(relation, index_of, d)
    logger.info("A_%d: %d basic equations, all multihomogeneous", d, len(relations))

    sample = morin.random_orbit_point(d, base_field)

    def vanishes(generators, names, label):
        point = [base_field(sample[name]) for name in names]
        for generator in generators:
            value = sum(
                (
                    coefficient * prod(p**e for p, e in zip(point, exponents))
                    for exponents, coefficient in generator.dict().items()
                ),
                base_field(0),
            )
            if value != 0:
                raise RuntimeError(f"A_{d}: {label} does not vanish on the orbit: {generator}")

    vanishes(relations, cheap, "a basic equation")

    logger.info("A_%d: saturating by %d defect-zero coordinates", d, len(defect_zero))
    generators, names = saturate_by_variables(relations, cheap, defect_zero, base_field)
    vanishes(generators, names, "a saturated generator")
    logger.info("A_%d: orbit ideal verified on random B_d-translates of eps_ref", d)

    final_ring, final_index = morin.coordinate_ring(base_field, d, order=names)
    ideal = final_ring.ideal([reindex(g, names, final_ring, names) for g in generators])
    return ideal, final_ring, final_index


def _multiplicity(exponents, indices, cache):
    """
    Length of the monomial ideal localised at the prime on ``indices``.

    Setting every other variable to 1 leaves an artinian monomial ideal whose
    length is the multiplicity of that component.  Restricted ideals repeat
    heavily across components -- 572 components share 51 of them at ``d = 7`` --
    so ``cache`` memoises by minimal generators.  The cache is passed in rather
    than held at module scope: it is only valid for one ideal.
    """
    # Keyed by the size too: an empty generating set does not determine it.
    key = (restrict_generators(exponents, indices), len(indices))
    if key not in cache:
        cache[key] = standard_monomial_count(key[0], len(indices))
    return cache[key]


def analyse(order, base_field):
    """
    Everything the saturation yields: the multidegree and the working it left behind.

    The twelve saturating Groebner bases are the dominant cost of the whole Sage
    stage, so a caller that wants the initial ideal or the component structure as
    well --- :mod:`multidegree.geometry` does --- must not pay for them twice.

    Returns ``(multidegree, context)`` where ``context`` carries ``ring``,
    ``index_of``, ``ideal``, ``initial``, ``components`` and ``multiplicities``.
    ``initial`` and the rest are ``None`` in the degenerate case ``d <= 3``.
    """
    ideal, ring, index_of = orbit_ideal(base_field, order)
    weight_ring = PolynomialRing(base_field, "z", order)
    z = weight_ring.gens()

    expected = morin.expected_multidegree_degree(order)
    if expected == 0:
        # d <= 3: the basic equations are vacuous, the orbit closure is the
        # whole of N_d, and the multidegree of the ambient space is 1.  This
        # is [N, Section 7.2]: "In these cases deg Q_d = 0, and thus Q_d = 1;
        # geometrically, this means that O_d = eps_ref, and thus O_d = N_d."
        if ideal.gens() and not ideal.is_zero():
            raise RuntimeError(
                f"A_{order}: expected the orbit to fill N_d, but the ideal is " f"nontrivial: {ideal.gens()}"
            )
        logger.info("A_%d: orbit fills the ambient space, Q_%d = 1", order, order)
        degenerate = Multidegree(
            polynomial=weight_ring(1),
            ring=weight_ring,
            family=morin.FAMILY,
            order=order,
            codim=0,
        )
        return degenerate, {
            "ring": ring,
            "index_of": index_of,
            "ideal": ideal,
            "initial": None,
            "components": None,
            "multiplicities": None,
        }

    logger.info("A_%d: Groebner basis and initial ideal", order)
    basis = ideal.groebner_basis(algorithm=ALGORITHM)
    initial = ring.ideal([g.lt() for g in basis])

    exponents = [next(iter(g.exponents())) for g in initial.gens()]
    supports = [frozenset(i for i, e in enumerate(m) if e) for m in exponents]
    components = minimal_transversals(supports, bound=expected)
    if not components:
        raise RuntimeError(
            f"A_{order}: the initial ideal has no component of codimension "
            f"{expected}. Saturation did not isolate the orbit component."
        )
    codimension = min(len(component) for component in components)
    if codimension != expected:
        raise RuntimeError(
            f"A_{order}: orbit closure has codimension {codimension}, expected "
            f"{expected}. Saturation did not isolate the orbit component."
        )
    # The search is bounded at `expected` and the minimum is `expected`, so every
    # surviving component already has that size; filtering says so out loud.
    top = [c for c in components if len(c) == codimension]
    logger.info("A_%d: %d components of codimension %d", order, len(top), codimension)

    variables = ring.gens()
    cache = {}
    multiplicities = []
    total = weight_ring(0)
    for component in top:
        indices = sorted(component)
        multiplicity = _multiplicity(exponents, indices, cache)
        multiplicities.append(multiplicity)
        normal = prod(morin.normal_weight(variables[i], index_of, z) for i in indices)
        total += multiplicity * normal

    logger.info("A_%d: deg Q_%d = %d as required", order, order, expected)
    result = Multidegree(
        polynomial=total,
        ring=weight_ring,
        family=morin.FAMILY,
        order=order,
        codim=expected,
    )
    return result, {
        "ring": ring,
        "index_of": index_of,
        "ideal": ideal,
        "initial": initial,
        "components": [sorted(c) for c in top],
        "multiplicities": multiplicities,
    }


def compute(order, base_field):
    """The multidegree ``Q_d`` of the Morin orbit closure, as a :class:`Multidegree`."""
    return analyse(order, base_field)[0]
