"""
STUB -- multidegrees by Bott--Samelson resolution and localisation.

Not implemented.  This file records the design so the work can be picked up.

The idea
--------
The orbit closure ``O_d`` is singular, which is what makes its ideal awkward.
Schubert calculus offers a standard way round: resolve it.  Build a smooth
variety ``X`` -- a Bott--Samelson tower, an iterated fibration of projective
lines -- with a proper birational map ``pi : X -> O_d``.  Since ``pi`` is
birational and ``X`` is smooth,

    [O_d]  =  pi_* [X],

and the pushforward of the fundamental class can be computed by
Atiyah--Bott--Berline--Vergne localisation: a sum over the isolated torus fixed
points of ``X`` of ``1 / e(T_p X)``, the equivariant Euler class of the tangent
space.  On a Bott--Samelson tower the fixed points are indexed by subwords of the
defining word and the tangent weights are explicit, so the whole computation
becomes a sum over a combinatorial tree -- no Gröbner basis anywhere.

Why it is attractive
--------------------
The current backend's cost is dominated by a Gröbner basis and a primary
decomposition in ``dim N_d`` variables (22 at ``d = 6``, 34 at ``d = 7``).
Localisation replaces that with a sum whose size is controlled by the length of
the word, so it is the most plausible route to ``d = 7``, where
``deg Q_7 = 13`` in a 34-variable ambient space puts the present method out of
reach.

The honest caveat
-----------------
Bérczi--Szenes, Remark 7.2, say the opposite of the usual folklore: there *are*
standard algorithms for the equivariant dual of a **toric** orbit -- they give one
-- but "no such algorithm is known for Borel orbits".  They suggest the
fibration of their Lemma 7.1, under which ``pr_0`` restricted to the orbit
fibres over the toric orbit ``T_d . eps_ref``, as the way one *might* reduce the
Borel case to the toric one.

So this is not an implementation task with a known recipe.  Constructing a
Bott--Samelson resolution of *this* orbit closure is open research, and the
first question to settle is whether the Lemma 7.1 fibration makes the reduction
work at all.

Work to be done
---------------
1. Settle the geometry first, on paper, at ``d = 4`` where ``Q_4 = 2z_1 + z_2 - z_4``
   is known: find an explicit resolution and check localisation reproduces it.
2. Only then attempt ``d = 5``, where ``Q_5 = (2z_1 + z_2 - z_5) P_5`` is known and
   the factorisation is itself a hint about the geometry.
3. Implement the fixed-point enumeration and the tangent weights; the sum is over
   subwords, so the natural data structure is a tree with weights on edges.
4. ``d = 6`` as validation against the existing backend, then ``d = 7`` as the
   actual payoff.
5. Localisation sums individually have denominators that must cancel; keep the
   arithmetic exact and assert the cancellation, rather than trusting it.

Interface note
--------------
Unlike :mod:`multidegree.backends.restriction`, this *does* produce a
multidegree, so it slots straight in as a
:class:`~multidegree.backends.base.MultidegreeBackend` with
``name = "bott-samelson"`` and is selected with ``--backend bott-samelson``.
The ``Multidegree`` result type already asserts that the returned polynomial has
the codimension the backend claims, which is exactly the check that would catch
a mis-assembled localisation sum.
"""

raise NotImplementedError(
    "Bott--Samelson localisation is not implemented; see the module docstring "
    "for the design, the open questions, and the work required"
)
