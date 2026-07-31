"""
The chamber algebra: what the SageMath stage produces and everything else reads.

One artifact per Morin order, holding the Bérczi--Szenes chamber series

    F_d  =  numerator / prod_r (1 - denominator_factors[r])

as exponent dictionaries.  Storing exponents rather than expressions is what
keeps the rest of the package free of symbolic algebra: no parsing, no
re-expansion, and exact integer arithmetic throughout.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .polynomial import Poly, is_nonneg, total_degree

DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class ChamberAlgebra:
    """
    The chamber series of A_d, in the coordinates x_j = z_j / z_{j+1}.

    Only :attr:`numerator` and :attr:`denominator_factors` are needed to
    reconstruct ``F_d``; the remaining fields record how it was assembled and
    are what the artifact-level tests check.
    """

    order: int
    chamber_vars: Tuple[str, ...]

    #: N_d, the fully expanded numerator of the chamber series.
    numerator: Poly
    #: The f_r.  Each has nonnegative coefficients and zero constant term, so
    #: 1/(1 - f_r) = sum_j f_r^j is coefficientwise nonnegative.
    denominator_factors: List[Poly]

    #: Q_d pushed forward to chamber coordinates.
    multidegree: Poly
    #: Q_d after sign and correction-monomial normalisation; constant term 1.
    normalized_numerator: Poly
    #: prod_{m<l} (1 - z_m/z_l), the Vandermonde of the residue formula.
    vandermonde: Poly

    field: str = "Rational Field"
    characteristic: int = 0

    #: Provenance: which singularity family, and which multidegree algorithm.
    family: str = "morin-a"
    backend: str = "basic-equations"

    @property
    def nvars(self) -> int:
        return len(self.chamber_vars)

    def validate(self) -> None:
        """Re-check the invariants the miner promises.  Raises on violation."""
        if len(self.chamber_vars) != self.order - 1:
            raise ValueError(
                f"A_{self.order}: expected {self.order - 1} chamber variables, "
                f"got {len(self.chamber_vars)}"
            )
        if self.normalized_numerator.get((0,) * self.nvars) != 1:
            raise ValueError(f"A_{self.order}: normalized numerator has constant term != 1")
        for index, factor in enumerate(self.denominator_factors):
            if not is_nonneg(factor):
                raise ValueError(
                    f"A_{self.order}: denominator factor {index} has a negative coefficient"
                )
            if any(sum(e) == 0 for e in factor):
                raise ValueError(
                    f"A_{self.order}: denominator factor {index} has a constant term"
                )

    def summary(self) -> str:
        return (
            f"A_{self.order}: variables {', '.join(self.chamber_vars)}; "
            f"numerator {len(self.numerator)} terms, degree {total_degree(self.numerator)}; "
            f"{len(self.denominator_factors)} denominator factors"
        )


def artifact_path(order: int) -> Path:
    return DATA_DIR / f"a{order}_algebra.pkl"


def available_orders() -> List[int]:
    """Morin orders for which an artifact is present."""
    return sorted(
        int(path.name[1 : path.name.index("_")]) for path in DATA_DIR.glob("a*_algebra.pkl")
    )


def load_algebra(order: int, validate: bool = True) -> ChamberAlgebra:
    """Load the artifact for A_order, checking its invariants by default."""
    path = artifact_path(order)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Regenerate it with "
            f"'python -m multidegree.build -d {order}' under SageMath."
        )
    with open(path, "rb") as handle:
        raw: Dict[str, object] = pickle.load(handle)

    algebra = ChamberAlgebra(
        order=raw["order"],
        chamber_vars=tuple(raw["chamber_vars"]),
        numerator=raw["numerator"],
        denominator_factors=list(raw["denominator_factors"]),
        multidegree=raw["multidegree"],
        normalized_numerator=raw["normalized_numerator"],
        vandermonde=raw["vandermonde"],
        field=raw.get("field", "Rational Field"),
        characteristic=raw.get("characteristic", 0),
        family=raw.get("family", "morin-a"),
        backend=raw.get("backend", "basic-equations"),
    )
    if validate:
        algebra.validate()
    return algebra
