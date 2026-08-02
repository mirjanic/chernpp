"""
Corank-two singularities: the I_{a,b} family and its jet-space orbit geometry.

Where :mod:`multidegree.morin` handles the corank-one Morin singularities A_d,
this module handles the first corank-two family.  ``I_{a,b}`` (Mather's notation,
as used on Rimányi's registry) is the class of germs with local algebra

    Q(I_{a,b}) = C[[x, y]] / (xy, x^a + y^b),      2 <= a <= b,

of dimension ``a + b``.  For equidimensional germs ``(C^2, 0) -> (C^2, 0)`` a
normal form is ``f(x, y) = (xy, x^a + y^b)``, so ``I_{a,b}`` lives in the space of
``max(a, b)``-jets.  ``I_{2,2}`` is the generic corank-two germ: its locus is the
Thom--Boardman stratum ``Sigma^2``, of codimension 4, whose Thom polynomial is the
Giambelli--Thom--Porteous class ``s_{2,2} = c_2^2 - c_1 c_3``.

Note ``I_{2,2} = C[[x,y]]/(xy, x^2 + y^2) = C[[x,y]]/(x^2, y^2)``: over C the two
presentations differ by ``u = x + iy``, ``v = x - iy``, both giving the rank-two
symmetric form on ``m/m^2``.  It is *not* ``C[[x,y]]/(x^2, xy, y^2)``, which needs
three ideal generators and so is not the local algebra of any germ ``C^2 -> C^2``;
that algebra belongs to the Veronese germ ``(C^2,0) -> (C^3,0)``.

What this module does and does not give
---------------------------------------
It computes the ``B_2 x B_2``-orbit closure of a corank-two jet and its
equivariant multidegree.  It deliberately stops there, for two reasons.

**There is no corank-two residue formula here.**  The Bérczi--Szenes formula that
:mod:`chernpp` implements turns ``Q_d`` into ``Tp_{A_d}``, and it is specific to
corank one.  Nothing in this package converts a corank-two multidegree into a
Thom polynomial, so this module emits no artifact; producing one with empty
denominator factors would let the Morin residue machinery run on corank-two data
and return a plausible-looking wrong answer.

**The Borel orbit closure is not an invariant of the singularity**, and the usual
ways out do not apply.  :func:`representative_survey` exhibits the problem: four
germs of the one class ``I_{2,2}`` give three different orbit closures, of
codimension 1, 2 and 2, one of them determinantal rather than a coordinate
subspace.  Even the two standard presentations of the algebra disagree --
``(xy, x^2+y^2)`` gives codimension 1, ``(x^2, y^2)`` codimension 2.

The natural repair is to demand that the representative be *generic* against the
flags, and it fails, for a structural reason worth stating plainly.  On the
corank-two 2-jet space ``GL_2 x GL_2`` does have a dense orbit -- ``I_{2,2}`` is
the generic corank-two 2-jet, of codimension 0 -- but ``B_2 x B_2`` does not: its
generic orbit is 5-dimensional in a 6-dimensional space, so the action has
modality at least one and the generic jet lies on a one-parameter family of
orbits.  Its closure is then a hypersurface whose equation moves with the
representative.  There is no generic Borel orbit to single out.

What Bérczi--Szenes do is therefore not "take the class of the ``A_d`` locus".
``Q_d`` is the equivariant class of *one specific* Borel orbit closure, that of a
canonically defined reference jet, and the residue formula is what converts it
into a Thom polynomial.  A corank-two analogue needs both halves: a canonical
reference jet, and a residue formula proved against it.  Neither is in the
literature this package builds on, which is why the module stops where it does.

A third fact worth recording before that work starts: the published ``I_{a,b}``
Thom polynomials are *Schur*-positive but have mixed signs in the Chern monomial
basis -- ``Tp(I_{2,2}) = c_2^2 - c_1 c_3`` already does.  So this family cannot
test Rimányi's Chern-monomial positivity, which is a corank-one statement.  What
it can test is Schur positivity, the Pragacz--Weber conjecture.
"""

from . import get_logger

logger = get_logger(__name__)

# Sage is imported inside the functions that need it.  The index bookkeeping, the
# normal forms and the weight arithmetic are plain Python, so the module -- and
# the test tier that checks them -- stays importable in the ordinary virtualenv.

FAMILY = "corank2-i"


def jet_indices(order):
    """
    Coordinates on the space of corank-two ``order``-jets ``(C^2,0) -> (C^2,0)``.

    A corank-two germ has vanishing differential, so its jet is a pair of power
    series with no constant or linear term.  The ambient space is therefore
    ``Hom(C^2, sum_{j=2}^{order} Sym^j C^2)``, and an index ``(t, i, j)`` names
    the coefficient of ``x^i y^j`` in the component ``t``, with ``i + j = j``-th
    graded piece.
    """
    return [
        (t, i, degree - i) for degree in range(2, order + 1) for t in (0, 1) for i in range(degree, -1, -1)
    ]


def variable_name(t, i, j):
    return f"q_{'uv'[t]}_{i}_{j}"


def ambient_dimension(order):
    """``2 * sum_{j=2}^{order} (j + 1)``; 6 for 2-jets, 14 for 3-jets."""
    return 2 * sum(degree + 1 for degree in range(2, order + 1))


def coordinate_ring(base_field, order):
    """The polynomial ring on the jet space, with each variable's index."""
    from sage.all import PolynomialRing, prod

    indices = jet_indices(order)
    names = [variable_name(*index) for index in indices]
    ring = PolynomialRing(base_field, len(names), names, order="degrevlex")
    return ring, {ring(variable_name(*index)): index for index in indices}


def normal_weight(variable, index_of, z):
    """
    The ``T_2 x T_2`` weight of a coordinate direction.

    ``z = [s1, s2, t1, t2]``: the source torus acts on ``x, y`` with weights
    ``s1, s2`` and the target on the two components with weights ``t1, t2``, so
    the coefficient of ``x^i y^j`` in component ``t`` has weight
    ``i*s1 + j*s2 - t``.  Same convention as :func:`multidegree.morin.normal_weight`
    -- the weight of the coordinate *function*, which is what multiplies into a
    multidegree.
    """
    t, i, j = index_of[variable]
    return i * z[0] + j * z[1] - z[2 + t]


def normal_form(a, b):
    """
    ``I_{a,b}`` as a pair of polynomials: ``f = (xy, x^a + y^b)``.

    Returned as coefficient dictionaries ``{(i, j): coefficient}``, the format
    :func:`orbit_closure` consumes.  ``I_{2,2}`` also has the familiar
    presentation ``(x^2, y^2)``; see :func:`representative_survey` for why the
    choice between them is not cosmetic.
    """
    if not 2 <= a <= b:
        raise ValueError(f"I_{{a,b}} needs 2 <= a <= b, got a={a}, b={b}")
    return ({(1, 1): 1}, {(a, 0): 1, (0, b): 1})


def orbit_closure(jet, order, base_field=None):
    """
    The ideal of the closure of ``B_2 x B_2 . jet``, by elimination.

    ``B_2 x B_2`` acts by substitution in the source and a change of basis in the
    target.  Both Borels are taken lower-triangular for the standard flags:
    the source substitutes ``x -> A x``, ``y -> B x + C y``, and the target sends
    ``(f_u, f_v) -> (M f_u, L f_u + N f_v)``, with ``A, C, M, N`` invertible.

    The ideal is obtained by eliminating the group parameters rather than
    guessed and checked, so the result is the orbit closure by construction.

    Restricted to ``order = 2``.  On 2-jets only the *linear* parts of a source
    and target diffeomorphism act: for ``f`` with no constant or linear term,
    ``f(phi_1 + phi_2 + ...) = f(phi_1) + O(3)`` and ``psi(f) = psi_1(f) + O(4)``.
    From order 3 on the higher parts contribute, so the group is the jet group
    rather than ``GL_2 x GL_2`` and this parametrisation would compute the orbit
    of the wrong group.  :func:`jet_indices` and :func:`ambient_dimension` still
    describe those larger jet spaces, which is what the ``I_{a,b}`` with
    ``b >= 3`` will need.
    """
    if order != 2:
        raise NotImplementedError(
            f"orbit_closure is restricted to 2-jets, got order={order}. Order 3 and "
            "above need the action of the jet group, not of GL_2 x GL_2; see the "
            "docstring."
        )

    from sage.all import PolynomialRing, prod

    if base_field is None:
        from sage.all import QQ

        base_field = QQ
    indices = jet_indices(order)
    names = [variable_name(*index) for index in indices]
    parameters = ["A", "B", "C", "L", "M", "N", "w"]
    ring = PolynomialRing(base_field, names + parameters, order="degrevlex")
    q = list(ring.gens()[: len(names)])
    A, B, C, L, M, N, w = ring.gens()[len(names) :]

    source = PolynomialRing(ring, ["x", "y"])
    x, y = source.gens()

    def act(component):
        image = source(0)
        for (i, j), coefficient in component.items():
            image += coefficient * (A * x) ** i * (B * x + C * y) ** j
        return image

    f_u, f_v = act(jet[0]), act(jet[1])
    images = [M * f_u, L * f_u + N * f_v]

    relations = []
    for position, (t, i, j) in enumerate(indices):
        relations.append(q[position] - images[t].coefficient({x: i, y: j}))
    # The torus and target-change parameters must be invertible.
    relations.append(w * A * C * M * N - 1)

    ideal = ring.ideal(relations).elimination_ideal([A, B, C, L, M, N, w])
    target = PolynomialRing(base_field, len(names), names, order="degrevlex")
    return target.ideal([target(str(g)) for g in ideal.gens()]), target


def multidegree(jet, order, base_field=None):
    """
    The ``T_2 x T_2``-equivariant multidegree of the orbit closure of ``jet``.

    Returned as ``(polynomial, codimension)`` in the weight ring
    ``QQ[s1, s2, t1, t2]``.  As in :mod:`multidegree.basic_equations`,
    the class is read off an initial ideal: passing to the initial ideal of a
    Groebner basis is a flat degeneration, so it preserves the equivariant dual,
    and for a monomial ideal that dual is the sum over top-dimensional components
    of the multiplicity times the product of the normal weights.
    """
    from sage.all import PolynomialRing, prod

    if base_field is None:
        from sage.all import QQ

        base_field = QQ
    ideal, ring = orbit_closure(jet, order, base_field)
    _, index_of = coordinate_ring(base_field, order)
    index_of = {str(variable): index for variable, index in index_of.items()}

    weights = PolynomialRing(base_field, ["s1", "s2", "t1", "t2"])
    z = weights.gens()

    if ideal.is_zero() or not ideal.gens():
        return weights(1), 0  # the orbit is dense in the jet space

    initial = ring.ideal([g.lt() for g in ideal.groebner_basis()])
    primes = initial.minimal_associated_primes()
    support = lambda prime: [v for v in prime.gens() if v != 0]
    codimension = min(len(support(prime)) for prime in primes)

    total = weights(0)
    for prime in (p for p in primes if len(support(p)) == codimension):
        variables = support(prime)
        collapse = {v: ring(1) for v in ring.gens() if v not in variables}
        local_ring = PolynomialRing(base_field, [str(v) for v in variables])
        collapsed = [g.subs(collapse) for g in initial.gens()]
        if any(g.is_constant() and g != 0 for g in collapsed):
            # A generator supported away from the prime collapses to 1, not to 0,
            # making the localised ideal the unit ideal.  Its length is zero, and
            # keeping the generator would report a positive multiplicity instead.
            raise RuntimeError(
                f"a generator collapses to a unit on {[str(v) for v in variables]}, "
                "so that set is not a minimal prime of the initial ideal"
            )
        local = local_ring.ideal([local_ring(str(g)) for g in collapsed if g != 0])
        multiplicity = (
            local.vector_space_dimension()
            if hasattr(local, "vector_space_dimension")
            else local.gen().degree()
        )
        normal = prod((normal_weight(str(v), index_of, z) for v in variables), weights(1))
        total += multiplicity * normal
    return total, codimension


#: A 2-jet in general position, used to measure generic orbit dimensions.
_GENERIC_JET = ({(2, 0): 1, (1, 1): 3, (0, 2): 5}, {(2, 0): 7, (1, 1): 11, (0, 2): 2})

#: Four germs of the single class I_{2,2}, in the coefficient-dictionary format
#: :func:`orbit_closure` consumes.  They give three different Borel orbit
#: closures; see :func:`representative_survey`.
_REPRESENTATIVES = {
    "(xy, x^2 + y^2)": ({(1, 1): 1}, {(2, 0): 1, (0, 2): 1}),
    "(x^2, y^2)": ({(2, 0): 1}, {(0, 2): 1}),
    "(y^2, x^2)": ({(0, 2): 1}, {(2, 0): 1}),
    "(x^2 + xy, y^2)": ({(2, 0): 1, (1, 1): 1}, {(0, 2): 1}),
}


def representative_survey(base_field=None):
    """
    Four germs of the single class ``I_{2,2}``, and their four orbit closures.

    Evidence that the Borel orbit closure depends on the representative, not only
    on the singularity: three distinct closures appear, of codimensions 1, 2 and
    2, and one of them is a determinantal variety rather than a coordinate
    subspace.  Returned as ``{label: (generators, codimension)}``.
    """
    from sage.all import PolynomialRing, prod

    if base_field is None:
        from sage.all import QQ

        base_field = QQ
    survey = {}
    for label, jet in _REPRESENTATIVES.items():
        ideal, ring = orbit_closure(jet, 2, base_field)
        codimension = ring.ngens() - ideal.dimension()
        survey[label] = (sorted(str(g) for g in ideal.gens()), codimension)
    return survey


def generic_orbit_dimensions(base_field=None):
    """
    Dimensions of the generic ``B_2 x B_2`` and ``GL_2 x GL_2`` orbits on 2-jets.

    Computed as the rank of the Jacobian of the orbit map at a jet in general
    position.  The pair ``(5, 6)`` is what closes off the "just take a generic
    representative" repair: ``GL_2 x GL_2`` is transitive on generic 2-jets, so
    ``I_{2,2}`` has codimension 0 there, while ``B_2 x B_2`` is not, so its
    generic orbit sits in a one-parameter family and has no canonical member.
    """
    from sage.all import PolynomialRing, matrix

    if base_field is None:
        from sage.all import QQ

        base_field = QQ

    jet = _GENERIC_JET

    def rank(parameters, source_rows, target_combination):
        ring = PolynomialRing(base_field, parameters)
        gens = ring.gens()
        polynomials = PolynomialRing(ring.fraction_field(), ["x", "y"])
        x, y = polynomials.gens()

        def image(component):
            total = polynomials(0)
            for (i, j), coefficient in component.items():
                total += coefficient * source_rows(gens, x, y, i, j)
            return [total.coefficient({x: i, y: j}) for i, j in ((2, 0), (1, 1), (0, 2))]

        coordinates = target_combination(gens, image(jet[0]), image(jet[1]))
        jacobian = matrix(
            ring.fraction_field(),
            [[ring(c).derivative(p) for p in gens] for c in coordinates],
        )
        return jacobian.rank()

    borel = rank(
        ["A", "B", "C", "L", "M", "N"],
        lambda g, x, y, i, j: (g[0] * x) ** i * (g[1] * x + g[2] * y) ** j,
        lambda g, u, v: [g[4] * a for a in u] + [g[3] * a + g[5] * b for a, b in zip(u, v)],
    )
    general = rank(
        ["a11", "a12", "a21", "a22", "b11", "b12", "b21", "b22"],
        lambda g, x, y, i, j: (g[0] * x + g[1] * y) ** i * (g[2] * x + g[3] * y) ** j,
        lambda g, u, v: [g[4] * a + g[5] * b for a, b in zip(u, v)]
        + [g[6] * a + g[7] * b for a, b in zip(u, v)],
    )
    return {"borel": int(borel), "general_linear": int(general)}


def survey_document(base_field=None):
    """
    The corank-two findings, as a JSON-able document.

    Written by the Sage stage and consumed by the test tier, exactly as the
    ``.npz`` algebras are: nothing downstream re-derives geometry, so no test
    needs SageMath.
    """
    if base_field is None:
        from sage.all import QQ

        base_field = QQ

    def terms(polynomial):
        return sorted(
            ([int(e) for e in exponents], int(coefficient))
            for exponents, coefficient in polynomial.dict().items()
        )

    representatives = []
    for label, jet in _REPRESENTATIVES.items():
        ideal, ring = orbit_closure(jet, 2, base_field)
        polynomial, codimension = multidegree(jet, 2, base_field)
        representatives.append(
            {
                "germ": label,
                "jet": [{str(list(k)): int(v) for k, v in part.items()} for part in jet],
                "generators": sorted(str(g) for g in ideal.gens()),
                "codimension": codimension,
                "multidegree": terms(polynomial),
            }
        )
    return {
        "source": "multidegree.corank2, generated by the SageMath stage",
        "note": (
            "Borel orbit closures of corank-two 2-jets.  All four germs represent "
            "the single class I_{2,2}, yet they give three different closures: the "
            "closure is an invariant of the jet, not of the singularity."
        ),
        "weight_variables": ["s1", "s2", "t1", "t2"],
        "jet_space": {
            str(order): {
                "ambient_dimension": ambient_dimension(order),
                "indices": [list(index) for index in jet_indices(order)],
            }
            for order in (2, 3, 4)
        },
        "normal_forms": [
            {
                "a": a,
                "b": b,
                "local_algebra": f"C[[x,y]]/(xy, x^{a} + y^{b})",
                "dimension": a + b,
                "jet_order": b,
                "ambient_dimension": ambient_dimension(b),
            }
            for a, b in ((2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4))
        ],
        "representatives": representatives,
        "generic_orbit": {
            "ambient_dimension": ambient_dimension(2),
            **generic_orbit_dimensions(base_field),
            "note": (
                "GL_2 x GL_2 is transitive on generic corank-two 2-jets, so I_{2,2} "
                "has codimension 0 there.  B_2 x B_2 is not: its generic orbit is a "
                "hypersurface, so the action has modality at least one and no generic "
                "Borel orbit can be singled out."
            ),
        },
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("output", help="path to write the survey JSON to")
    arguments = parser.parse_args()
    document = survey_document()
    with open(arguments.output, "w") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
    logger.info("wrote %s", arguments.output)
