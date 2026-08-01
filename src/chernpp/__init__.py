"""
Chern++ -- Positive Chern Classes in Thom Polynomials.

Tools for the Thom polynomials of the Morin singularities A_d, in the form
given by the Bérczi--Szenes residue formula, and for the two positivity
conjectures attached to them:

* **Rimányi weak Chern positivity** -- the coefficients of Tp_{A_d} in the
  relative Chern classes are nonnegative.  This is the target.
* **Bérczi--Szenes strong Laurent positivity** -- the chamber expansion
  ``F_d = sum_beta A_beta x^beta`` is itself coefficientwise nonnegative.  It
  implies the first, holds for ``d = 4``, and fails for ``d >= 5``.

Layout.  :mod:`multidegree` (SageMath, run separately) computes the Borel orbit
multidegree ``Q_d`` and writes one artifact per order into ``chernpp/data``.
This package reads those artifacts and never re-derives them:

* :mod:`chernpp.polynomial`   -- exact sparse polynomial arithmetic
* :mod:`chernpp.artifacts`    -- the chamber algebra and its invariants
* :mod:`chernpp.chamber`      -- the chamber series and the structural reductions
* :mod:`chernpp.chern`        -- Chern coefficients via an XLA-compiled expansion
* :mod:`chernpp.certificates` -- denominator certificates and order obstructions
* :mod:`chernpp.families`     -- closed-form domination along infinite families
* :mod:`chernpp.lorentzian`   -- log-concavity / M-convexity tests
"""

from .artifacts import ChamberAlgebra, available_orders, load_algebra
from .certificates import (
    Certificate,
    minimum_order,
    projection_is_feasible,
    search_certificate,
)
from .chamber import (
    ballot_orderings,
    chamber_series,
    chern_coefficient,
    paired_defects,
    sorted_negatives,
    tau,
    unpaired_tail_defects,
)
from .polynomial import expand_rational, is_nonneg, negative_terms

__version__ = "1.0.0"

#: Names served from :mod:`chernpp.chern`, imported on first use rather than at
#: package import.  That module pulls in JAX, which the SageMath stage does not
#: have -- and the SageMath stage needs :mod:`chernpp.artifacts` to write the
#: artifacts in the first place.  Deferring keeps one definition of the storage
#: format instead of a reader here and a writer there.
_LAZY = {"chern_coefficients": "chern", "thom_polynomial": "chern"}


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LAZY))


__all__ = [
    "ChamberAlgebra",
    "Certificate",
    "available_orders",
    "ballot_orderings",
    "chamber_series",
    "chern_coefficient",
    "chern_coefficients",
    "expand_rational",
    "is_nonneg",
    "load_algebra",
    "minimum_order",
    "negative_terms",
    "paired_defects",
    "projection_is_feasible",
    "search_certificate",
    "sorted_negatives",
    "tau",
    "thom_polynomial",
    "unpaired_tail_defects",
]
