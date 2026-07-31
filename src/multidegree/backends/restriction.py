"""
STUB -- Thom polynomials from Rimányi's restriction equations.

Not implemented.  This file records the design so the work can be picked up.

The idea
--------
Rimányi's method computes the Thom polynomial directly, without ever forming an
orbit closure.  For a singularity ``eta``, the class ``Tp_eta`` is pinned down by
*restriction equations*: for every singularity ``zeta`` in the closure of the
``eta``-stratum, restricting ``Tp_eta`` to the maximal torus of the symmetry
group of ``zeta`` must give

  * ``0``                                    if ``eta`` is not adjacent to ``zeta``,
  * the equivariant Euler class of the normal bundle, at ``zeta = eta`` itself.

Write an ansatz with unknown coefficients in a basis of the right degree, impose
those evaluations, and solve.  The system is massively overdetermined, which is
what makes it self-checking: a consistent solution of an overdetermined linear
system is strong evidence the input strata and weights are right.

Why this is not simply another :class:`MultidegreeBackend`
---------------------------------------------------------
It computes ``Tp``, not ``Q_d``.  The rest of Chern++ consumes ``Q_d``: the
chamber series ``F_d``, its Laurent coefficients ``A_beta``, the certificates and
the tail analysis all live upstream of the residue.  ``Tp`` is a residue extracted
*from* that data and cannot be inverted back.

So this belongs in the pipeline as an **independent check**, not a replacement:
compute ``Tp_{A_d}`` here, and compare against
:func:`chernpp.chern.thom_polynomial`.  Agreement at ``d = 6`` would be a genuinely
independent confirmation of ``Q_6``, derived from different mathematics -- much
stronger than the eleven classical coefficients we currently check against.

Work to be done
---------------
1. Enumerate the strata adjacent to ``A_d``: the lesser Morin singularities
   ``A_k`` for ``k < d``, and the relevant multi-germs.  This is a finite,
   documented combinatorial list, but assembling it correctly is the bulk of the
   work and the part most likely to hide errors.
2. For each stratum, its maximal-torus weights and the Euler class of its normal
   bundle.
3. A basis for the ansatz: Chern monomials of the correct degree, or Schur
   polynomials, which Rimányi's own presentation favours.
4. Assemble and solve the (overdetermined) linear system over ``QQ``; report the
   residual rank so an inconsistent system is loud rather than silent.
5. Cross-check against :func:`chernpp.chern.thom_polynomial` for ``d = 4, 5, 6``.

Interface note
--------------
Because the output is a Thom polynomial rather than a multidegree, this should
*not* subclass :class:`~multidegree.backends.base.MultidegreeBackend`.  Give it
its own entry point, e.g. ``thom_polynomial(family, order, relative_dimension)``,
and a separate registry if a second such method ever appears.
"""

raise NotImplementedError(
    "restriction-equation Thom polynomials are not implemented; see the module "
    "docstring for the design and the work required"
)
