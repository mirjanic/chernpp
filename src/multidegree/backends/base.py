"""
The contract every multidegree backend implements.

A backend answers one question: *given a singularity family and an order, what
is the equivariant multidegree of the associated orbit closure?*  How it gets
there -- explicit defining equations, a parametrisation and elimination,
restriction equations, a resolution and pushforward -- is entirely its own
business, and is exactly what varies between backends.

Everything downstream consumes only :class:`Multidegree`, so a new algorithm
plugs in without touching the chamber assembly or the artifact schema.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Multidegree:
    """
    The result of a backend computation.

    ``polynomial`` is the equivariant multidegree, homogeneous of degree
    ``codim``, in the torus weight variables of ``ring`` (``z_0, ..., z_{d-1}``
    standing for ``z_1, ..., z_d``).
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


class MultidegreeBackend(ABC):
    """
    Base class for multidegree algorithms.

    Subclasses set :attr:`name`, :attr:`families` and :attr:`description`, and
    implement :meth:`compute`.  Register them with
    :func:`multidegree.backends.register` to make them selectable by name from
    the command line.
    """

    #: Selector used by ``--backend``.
    name: str = ""
    #: Singularity families this backend can handle, e.g. ``("morin-a",)``.
    families: Tuple[str, ...] = ()
    #: One line shown by ``--list-backends``.
    description: str = ""

    def supports(self, family: str) -> bool:
        return family in self.families

    @abstractmethod
    def compute(self, family: str, order: int, base_field) -> Multidegree:
        """Compute the multidegree for ``family`` at the given ``order``."""

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<{type(self).__name__} name={self.name!r} families={self.families}>"
