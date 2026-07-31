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
* :mod:`multidegree.backends` -- the *algorithms* that compute a multidegree for
  a model.  Selectable by name, so a new algorithm (or a model beyond the
  Morin A_d family) can be added without touching the rest of the pipeline.
* :mod:`multidegree.build` -- the command-line entry point.

Submodules are not imported here, because importing them pulls in ``sage.all``.
"""

import logging
import sys

__all__ = ["backends", "build", "get_logger", "morin"]


def get_logger(name: str) -> logging.Logger:
    """A stdout logger; deliberately free of any Sage import."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
            )
        )
        logger.addHandler(handler)
        logger.propagate = False
    return logger
