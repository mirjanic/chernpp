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

**The Borel orbit closure is not an invariant of the singularity.**  It depends on
which representative of the ``GL_2 x GL_2`` class is used, because the Borel sees
the flag.  :func:`representative_survey` exhibits this for ``I_{2,2}``: four
germs of the one class give three different orbit closures, of codimension 1, 2
and 2, one of them a determinantal variety rather than a coordinate subspace.
Even the two standard presentations of the algebra disagree -- ``(xy, x^2+y^2)``
gives codimension 1 and ``(x^2, y^2)`` codimension 2.  Bérczi--Szenes avoid this because their
``eps_ref`` is a specific jet, not merely a representative; pinning down the
corank-two analogue is a prerequisite for any formula built on this data.

A third fact worth recording before that work starts: the published ``I_{a,b}``
Thom polynomials are *Schur*-positive but have mixed signs in the Chern monomial
basis -- ``Tp(I_{2,2}) = c_2^2 - c_1 c_3`` already does.  So this family cannot
test Rimányi's Chern-monomial positivity, which is a corank-one statement.  What
it can test is Schur positivity, the Pragacz--Weber conjecture.
"""

from sage.all import *

from . import get_logger

logger = get_logger(__name__)

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
        (t, i, degree - i)
        for degree in range(2, order + 1)
        for t in (0, 1)
        for i in range(degree, -1, -1)
    ]


def variable_name(t, i, j):
    return f"q_{'uv'[t]}_{i}_{j}"


def ambient_dimension(order):
    """``2 * sum_{j=2}^{order} (j + 1)``; 6 for 2-jets, 14 for 3-jets."""
    return 2 * sum(degree + 1 for degree in range(2, order + 1))


def coordinate_ring(base_field, order):
    """The polynomial ring on the jet space, with each variable's index."""
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


def orbit_closure(jet, order, base_field=QQ):
    """
    The ideal of the closure of ``B_2 x B_2 . jet``, by elimination.

    ``B_2 x B_2`` acts by substitution in the source and a change of basis in the
    target.  Both Borels are taken lower-triangular for the standard flags:
    the source substitutes ``x -> A x``, ``y -> B x + C y``, and the target sends
    ``(f_u, f_v) -> (M f_u, L f_u + N f_v)``, with ``A, C, M, N`` invertible.

    The ideal is obtained by eliminating the group parameters rather than
    guessed and checked, so the result is the orbit closure by construction.
    """
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


def multidegree(jet, order, base_field=QQ):
    """
    The ``T_2 x T_2``-equivariant multidegree of the orbit closure of ``jet``.

    Returned as ``(polynomial, codimension)`` in the weight ring
    ``QQ[s1, s2, t1, t2]``.  As in :mod:`multidegree.backends.basic_equations`,
    the class is read off an initial ideal: passing to the initial ideal of a
    Groebner basis is a flat degeneration, so it preserves the equivariant dual,
    and for a monomial ideal that dual is the sum over top-dimensional components
    of the multiplicity times the product of the normal weights.
    """
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
        local = local_ring.ideal(
            [
                local_ring(str(g.subs(collapse)))
                for g in initial.gens()
                if g.subs(collapse) != 0
            ]
        )
        multiplicity = (
            local.vector_space_dimension()
            if hasattr(local, "vector_space_dimension")
            else local.gen().degree()
        )
        normal = prod((normal_weight(str(v), index_of, z) for v in variables), weights(1))
        total += multiplicity * normal
    return total, codimension


def representative_survey(base_field=QQ):
    """
    Four germs of the single class ``I_{2,2}``, and their four orbit closures.

    Evidence that the Borel orbit closure depends on the representative, not only
    on the singularity: three distinct closures appear, of codimensions 1, 2 and
    2, and one of them is a determinantal variety rather than a coordinate
    subspace.  Returned as ``{label: (generators, codimension)}``.
    """
    representatives = {
        "(xy, x^2 + y^2)": ({(1, 1): 1}, {(2, 0): 1, (0, 2): 1}),
        "(x^2, y^2)": ({(2, 0): 1}, {(0, 2): 1}),
        "(y^2, x^2)": ({(0, 2): 1}, {(2, 0): 1}),
        "(x^2 + xy, y^2)": ({(2, 0): 1, (1, 1): 1}, {(0, 2): 1}),
    }
    survey = {}
    for label, jet in representatives.items():
        ideal, ring = orbit_closure(jet, 2, base_field)
        codimension = ring.ngens() - ideal.dimension()
        survey[label] = (sorted(str(g) for g in ideal.gens()), codimension)
    return survey
