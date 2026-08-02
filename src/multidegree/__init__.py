"""
The SageMath stage of Chern++: computing equivariant multidegrees.

This package is imported only when regenerating the algebra artifacts, and is
the sole part of the project that requires SageMath.  Everything in
:mod:`chernpp` reads the artifacts it produces and needs nothing but NumPy,
SciPy, JAX and SymPy.

Structure
---------
* :mod:`multidegree.morin` -- the singularity *model*: which ambient space, what
  torus weights, which reference point, and how the residue formula is
  rewritten in chamber coordinates.  Everything Morin-specific lives here.
* :mod:`multidegree.basic_equations` -- the *algorithm*: the explicit relations of
  the orbit closure, saturated, with the multidegree read off an initial ideal.
* :mod:`multidegree.monomial` -- the combinatorics that step rests on, kept free
  of Sage so it can be tested in the ordinary virtualenv.
* :mod:`multidegree.corank2` -- exploratory corank-two geometry; produces no
  artifact, see its docstring for why.
* :mod:`multidegree.build` -- the command-line entry point.

Submodules are not imported here, because importing them pulls in ``sage.all``.
"""

import logging
import sys
from dataclasses import dataclass

__all__ = ["Multidegree", "build", "get_logger", "monomial", "morin"]


@dataclass(frozen=True)
class Multidegree:
    """
    An equivariant multidegree, with the one invariant worth enforcing.

    ``polynomial`` is homogeneous of degree ``codim`` in the torus weight
    variables of ``ring`` (``z_0, ..., z_{d-1}`` standing for ``z_1, ..., z_d``).
    A multidegree whose degree is not the codimension of the variety it came from
    is a bug in whatever produced it, and one that would otherwise propagate
    silently into an artifact, so it is refused here at the boundary.
    """

    polynomial: object
    ring: object
    family: str
    order: int
    codim: int

    def __post_init__(self):
        degree = self.polynomial.degree()
        if degree != self.codim:
            raise RuntimeError(
                f"{self.family} order {self.order}: multidegree has degree {degree}, "
                f"but the orbit closure has codimension {self.codim}"
            )


def get_logger(name: str) -> logging.Logger:
    """A stdout logger; deliberately free of any Sage import."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.propagate = False
    return logger
