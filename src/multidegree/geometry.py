"""
Geometry of the orbit closure that comes free with the multidegree.

The Sage stage already builds the ideal of ``O_d``, a Gröbner basis, the initial
ideal and its component structure, and then throws almost all of it away: the
residue formula needs only ``Q_d``.  What is discarded is not junk.  This module
saves it as a second artifact per order, ``a{d}_geometry.npz``, which nothing in
:mod:`chernpp` reads --- it is recorded because it is geometry, and because two of
these quantities check each other.

What is kept
------------
**The factorisation of** ``Q_d``.  At ``d = 4`` the class is the single linear form
``2z_1 + z_2 - z_4``; at ``d = 5`` it factors as ``(2z_1 + z_2 - z_5) P_5``, and that
factorisation has been read as a hint about the geometry.  It is not a pattern:
``Q_6`` and ``Q_7`` are irreducible over ``Q``.  ``d = 5`` is the only order where the
class factors at all, which is worth knowing before building on it.

**The Hilbert numerator** of the orbit closure.  Passing to the initial ideal is a
flat degeneration, so it preserves the Hilbert function, and the numerator is
therefore an invariant of ``O_d`` itself --- its dimension, its degree, and the
ungraded shadow of the K-polynomial.  The multigraded refinement is the natural
next object, since that is what a K-theoretic Thom polynomial would need; it is
not computed here (see the report's open questions).

**The component structure** of the initial ideal: which coordinate subspaces
appear in the degeneration and with what multiplicity --- 2, 5, 44 and 572 of
them for ``d = 4, \ldots, 7``.  This is the combinatorial skeleton the multidegree
is summed over, and the data a Stanley--Reisner or toric analysis would start
from.  Unlike everything else recorded here it is *not* canonical: the initial
ideal depends on the term order, and reordering the variables to speed up the
saturation took ``d = 7`` from 621 components to 572.  The multidegree summed over
them did not move, which is the point --- it is an invariant of the variety, and
the degeneration is only a way of reaching it.

The check they give each other
------------------------------
Setting every ``z_i = 1`` in a multidegree collapses it to the ordinary degree of
the variety.  The Hilbert numerator computes that degree by a completely
different route --- a Hilbert series, not a sum over components --- so
``Q_d(1, ..., 1)`` and the Hilbert degree agreeing is a real cross-check on both.
They agree: 2, 6, 55 and 957 for ``d = 4, 5, 6, 7``, and so does a third
route, the sum of the component multiplicities.  :func:`build` refuses to
write an artifact where they do not.
"""

from sage.all import *

from . import basic_equations, get_logger, morin

logger = get_logger(__name__)

#: Bumped when the layout changes.
FORMAT_VERSION = 1


def hilbert_data(initial, ambient_dimension):
    """
    ``(numerator coefficients, dimension, codimension, degree)`` of the orbit closure.

    The Hilbert series is ``numerator(t) / (1 - t)^ambient``.  Cancelling the
    ``(1 - t)^codim`` that the codimension forces leaves a polynomial whose value
    at ``t = 1`` is the degree.
    """
    numerator = initial.hilbert_numerator()
    dimension = initial.dimension()
    codimension = ambient_dimension - dimension

    reduced, t = numerator, numerator.parent().gen()
    for _ in range(codimension):
        reduced, remainder = reduced.quo_rem(1 - t)
        if remainder != 0:
            raise RuntimeError(
                f"the Hilbert numerator is not divisible by (1 - t)^{codimension}, so the "
                f"codimension {codimension} read off the ideal is not the one the series sees"
            )
    return (
        [int(c) for c in numerator.list()],
        int(dimension),
        int(codimension),
        int(reduced(1)),
    )


def build(order, base_field=None):
    """
    Compute the geometry record for ``A_order``, cross-checking degree two ways.

    Returns a dict of plain Python and NumPy-ready values, the same shape
    :func:`chernpp.artifacts.save_geometry` writes.
    """
    if base_field is None:
        base_field = QQ

    result = basic_equations.compute(order, base_field)
    multidegree = result.polynomial
    weight_ring = result.ring

    ideal, ring, index_of = basic_equations.orbit_ideal(base_field, order)

    degenerate = result.codim == 0
    if degenerate:
        # d <= 3: the orbit fills N_d.  The Hilbert series of the zero ideal is
        # 1/(1 - t)^n, so the numerator is 1 and the degree is 1, matching
        # Q_d = 1 evaluated anywhere.  Singular will not take a Groebner basis in
        # zero variables, which d = 1 would ask of it.
        initial = ideal
        numerator, dimension, codimension, degree = [1], ring.ngens(), 0, 1
    else:
        basis = ideal.groebner_basis(algorithm=basic_equations.ALGORITHM)
        initial = ring.ideal([g.lt() for g in basis])
        numerator, dimension, codimension, degree = hilbert_data(initial, ring.ngens())

    at_one = multidegree.subs({v: 1 for v in weight_ring.gens()})
    if int(at_one) != degree:
        raise RuntimeError(
            f"A_{order}: Q_d(1,...,1) = {at_one} but the Hilbert series gives degree "
            f"{degree}. These are independent computations of the same integer; a "
            "disagreement means one of them is wrong."
        )
    if codimension != result.codim:
        raise RuntimeError(
            f"A_{order}: the ideal has codimension {codimension} but the multidegree "
            f"claims {result.codim}"
        )
    logger.info(
        "A_%d: dim %d, codim %d, degree %d (agreeing with Q_%d(1,...,1))",
        order,
        dimension,
        codimension,
        degree,
        order,
    )

    factors = []
    for factor, multiplicity in multidegree.factor():
        factors.append((factor, int(multiplicity)))
    total = sum(f.degree() * m for f, m in factors)
    if total != multidegree.degree():
        raise RuntimeError(f"A_{order}: factor degrees sum to {total}, not deg Q_d = {multidegree.degree()}")
    logger.info(
        "A_%d: Q_%d has %d irreducible factor(s) of degree(s) %s",
        order,
        order,
        sum(m for _, m in factors),
        sorted(f.degree() for f, m in factors for _ in range(m)),
    )

    component_records = []
    if not degenerate:
        exponents = [next(iter(g.exponents())) for g in initial.gens()]
        supports = [frozenset(i for i, e in enumerate(m) if e) for m in exponents]
        cache = {}
        component_records = [
            (sorted(c), basic_equations._multiplicity(exponents, sorted(c), base_field, cache))
            for c in basic_equations.minimal_transversals(supports, bound=codimension)
            if len(c) == codimension
        ]
        # Each component contributes multiplicity x (product of its normal
        # weights), and at z = 1 every weight is 1, so the multiplicities sum to
        # the degree.  A third route to the same integer.
        total = sum(m for _, m in component_records)
        if total != degree:
            raise RuntimeError(f"A_{order}: component multiplicities sum to {total}, not the degree {degree}")

    variables = [str(v) for v in ring.gens()]
    weights = []
    for variable in ring.gens():
        l, m, r = index_of[variable]
        vector = [0] * order
        vector[m - 1] += 1
        vector[r - 1] += 1
        vector[l - 1] -= 1
        weights.append(vector)

    return {
        "order": order,
        "field": str(base_field),
        "dimension": dimension,
        "codimension": codimension,
        "degree": degree,
        "ambient_dimension": ring.ngens(),
        "variables": variables,
        "weights": weights,
        "hilbert_numerator": numerator,
        "factors": [(morin.to_python(factor, 0), multiplicity) for factor, multiplicity in factors],
        "components": component_records,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("-d", "--order", type=int, default=6, help="highest order")
    parser.add_argument("--from-order", type=int, default=4, help="lowest order")
    arguments = parser.parse_args()

    from chernpp.artifacts import DATA_DIR, save_geometry

    for order in range(arguments.from_order, arguments.order + 1):
        logger.info("=" * 60)
        record = build(order)
        path = DATA_DIR / f"a{order}_geometry.npz"
        save_geometry(record, path)
        logger.info("A_%d: wrote %s", order, path)
