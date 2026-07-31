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

from .. import get_logger, morin
from .base import Multidegree, MultidegreeBackend

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


def orbit_ideal(base_field, d):
    """
    The ideal of the orbit closure, with three self-checks.

    Returns ``(ideal, ring, index_of)``.  Raises rather than returning anything
    it cannot justify: every relation must be multihomogeneous for the torus
    weights, and both the raw and the saturated ideal must vanish on random
    points of the orbit.
    """
    ring, index_of = morin.coordinate_ring(base_field, d)
    relations = basic_equations(ring, d)
    for relation in relations:
        morin.multiweight(relation, index_of, d)
    logger.info("A_%d: %d basic equations, all multihomogeneous", d, len(relations))

    ideal = ring.ideal(relations)

    sample = morin.random_orbit_point(d, base_field)
    point = {ring(name): base_field(value) for name, value in sample.items()}

    def vanishes(generators, label):
        for generator in generators:
            if generator.subs(point) != 0:
                raise RuntimeError(
                    f"A_{d}: {label} does not vanish on the orbit: {generator}"
                )

    vanishes(ideal.gens(), "a basic equation")

    defect_zero = [
        ring(f"q_{l}_{m}_{r}") for (m, r, l) in morin.weight_indices(d) if m + r == l
    ]
    logger.info("A_%d: saturating by %d defect-zero coordinates", d, len(defect_zero))
    for coordinate in defect_zero:
        ideal = ideal.saturation(coordinate)[0]

    vanishes(ideal.gens(), "a saturated generator")
    logger.info("A_%d: orbit ideal verified on random B_d-translates of eps_ref", d)
    return ideal, ring, index_of


class BasicEquationsBackend(MultidegreeBackend):
    name = "basic-equations"
    families = (morin.FAMILY,)
    description = (
        "Explicit quadratic relations of the orbit closure (Berczi-Szenes Prop. 7.3), "
        "saturated to isolate the orbit component; multidegree via an initial ideal."
    )

    def compute(self, family, order, base_field):
        if not self.supports(family):
            raise ValueError(f"{self.name} does not support family {family!r}")

        ideal, ring, index_of = orbit_ideal(base_field, order)
        weight_ring = PolynomialRing(base_field, "z", order)
        z = weight_ring.gens()

        logger.info("A_%d: Groebner basis and initial ideal", order)
        basis = ideal.groebner_basis(algorithm="libsingular:slimgb")
        initial = ring.ideal([g.lt() for g in basis])

        def support(prime):
            return [v for v in prime.gens() if v != 0]

        primes = initial.minimal_associated_primes()
        codimension = min(len(support(p)) for p in primes)

        expected = morin.expected_multidegree_degree(order)
        if codimension != expected:
            raise RuntimeError(
                f"A_{order}: orbit closure has codimension {codimension}, expected "
                f"{expected}. Saturation did not isolate the orbit component."
            )

        total = weight_ring(0)
        for prime in (p for p in primes if len(support(p)) == codimension):
            variables = support(prime)
            collapse = {v: ring(1) for v in ring.gens() if v not in variables}

            local_ring = PolynomialRing(base_field, [str(v) for v in variables])
            local = local_ring.ideal(
                [
                    local_ring(str(g.subs(collapse)))
                    for g in initial.gens()
                    if g.subs(collapse) != 0
                ]
            )
            # Localising at a monomial prime by setting the other variables to 1
            # leaves a zero-dimensional ideal whose length is the multiplicity.
            # Sage models a univariate ideal (the codimension-1 case)
            # differently, and there the length is the degree of the generator.
            multiplicity = (
                local.vector_space_dimension()
                if hasattr(local, "vector_space_dimension")
                else local.gen().degree()
            )
            normal = prod(morin.normal_weight(v, index_of, z) for v in variables)
            total += multiplicity * normal

        logger.info("A_%d: deg Q_%d = %d as required", order, order, expected)
        return Multidegree(
            polynomial=total,
            ring=weight_ring,
            family=family,
            order=order,
            codim=expected,
        )
