"""
The chamber algebra: what the SageMath stage produces and everything else reads.

One artifact per Morin order, holding the Bérczi--Szenes chamber series

    F_d  =  numerator / prod_r (1 - denominator_factors[r])

as exponent dictionaries.  Storing exponents rather than expressions is what
keeps the rest of the package free of symbolic algebra: no parsing, no
re-expansion, and exact integer arithmetic throughout.

Storage format
--------------
Compressed ``.npz``.  Each polynomial is two arrays -- an ``(n, nvars)`` block
of exponents and an ``(n,)`` vector of coefficients -- and the list of
denominator factors adds an offsets vector delimiting the blocks.

Coefficients are stored as ``int64``.  They are small -- the largest anywhere in
``A_7`` is 196803 -- and NumPy would otherwise fall back to an object array of
Python integers, which can only be read back with ``allow_pickle=True``.  That
would reintroduce, through the back door, exactly the property this format
exists to avoid, so :func:`pack_polynomial` refuses a coefficient that does not
fit rather than widening the dtype.

The choice matters for a repository meant to be shared: ``.npz`` is data, while
a pickle is a program, and unpickling a file executes whatever it contains.  It
is also readable from any language with a zip and NumPy-format reader, and it
compresses, where a pickle of the same dictionaries does not.  Parquet would do
the second and third of those but not the first-class fit: the payload is
ragged integer arrays, not a table, and it would add a dependency where NumPy
is already required.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .polynomial import Poly, is_nonneg, total_degree

DATA_DIR = Path(__file__).resolve().parent / "data"

#: Bumped when the layout changes, so a stale artifact is reported rather than
#: silently misread.  Version 2 stores coefficients as int64 instead of as an
#: object array, which is what lets the loader refuse pickles outright.
FORMAT_VERSION = 2

#: Coefficients must round-trip exactly through int64.
_INT64_RANGE = (-(2**63), 2**63 - 1)


def _as_int64(coefficients):
    """Coefficients as an int64 array, refusing anything that would not fit."""
    values = [int(c) for c in coefficients]
    for value in values:
        if not _INT64_RANGE[0] <= value <= _INT64_RANGE[1]:
            raise OverflowError(
                f"coefficient {value} does not fit in int64. Storing it would "
                "need an object array, which could only be read back with "
                "allow_pickle=True; widen the format instead."
            )
    return np.array(values, dtype=np.int64)


def pack_polynomial(poly: Poly, nvars: int):
    """``{exponent tuple: coefficient}`` -> ``(exponents, coefficients)`` arrays."""
    if not poly:
        return (
            np.zeros((0, nvars), dtype=np.int32),
            np.zeros((0,), dtype=np.int64),
        )
    items = sorted(poly.items())
    exponents = np.array([e for e, _ in items], dtype=np.int32)
    coefficients = _as_int64(c for _, c in items)
    return exponents, coefficients


def unpack_polynomial(exponents, coefficients) -> Poly:
    """Inverse of :func:`pack_polynomial`."""
    return {
        tuple(int(x) for x in exponent): int(coefficient)
        for exponent, coefficient in zip(exponents, coefficients)
    }


def pack_polynomial_list(polynomials: Sequence[Poly], nvars: int):
    """Concatenate several polynomials, with offsets delimiting them."""
    blocks = [pack_polynomial(p, nvars) for p in polynomials]
    offsets = np.cumsum([0] + [len(c) for _, c in blocks]).astype(np.int32)
    if blocks:
        exponents = np.concatenate([e for e, _ in blocks]) if blocks else None
        coefficients = np.concatenate([c for _, c in blocks])
    else:
        exponents = np.zeros((0, nvars), dtype=np.int32)
        coefficients = np.zeros((0,), dtype=np.int64)
    return exponents, coefficients, offsets


def unpack_polynomial_list(exponents, coefficients, offsets) -> List[Poly]:
    return [unpack_polynomial(exponents[a:b], coefficients[a:b]) for a, b in zip(offsets[:-1], offsets[1:])]


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
        """Re-check the invariants the Sage stage promises.  Raises on violation."""
        if len(self.chamber_vars) != self.order - 1:
            raise ValueError(
                f"A_{self.order}: expected {self.order - 1} chamber variables, "
                f"got {len(self.chamber_vars)}"
            )
        if self.characteristic != 0:
            # Everything downstream -- the chamber series, the Chern
            # coefficients, the certificates -- assumes exact integers. A build
            # run with `-p` writes a mod-p artifact into the same directory, and
            # nothing else would notice.
            raise ValueError(
                f"A_{self.order}: artifact is over characteristic {self.characteristic}, "
                "but the chamber-series machinery assumes characteristic 0"
            )
        if self.normalized_numerator.get((0,) * self.nvars) != 1:
            raise ValueError(f"A_{self.order}: normalized numerator has constant term != 1")
        for index, factor in enumerate(self.denominator_factors):
            if not is_nonneg(factor):
                raise ValueError(f"A_{self.order}: denominator factor {index} has a negative coefficient")
            if any(sum(e) == 0 for e in factor):
                raise ValueError(f"A_{self.order}: denominator factor {index} has a constant term")

    def summary(self) -> str:
        return (
            f"A_{self.order}: {self.nvars} chamber variables; "
            f"numerator {len(self.numerator)} terms, degree {total_degree(self.numerator)}; "
            f"{len(self.denominator_factors)} denominator factors"
        )


def artifact_path(order: int) -> Path:
    return DATA_DIR / f"a{order}_algebra.npz"


def available_orders() -> List[int]:
    """Morin orders for which an artifact is present."""
    return sorted(int(path.name[1 : path.name.index("_")]) for path in DATA_DIR.glob("a*_algebra.npz"))


def save_algebra(fields: Dict[str, object], path: Path) -> None:
    """
    Write one artifact.  ``fields`` uses the same keys as :class:`ChamberAlgebra`.

    Kept here rather than in the SageMath stage so that the reader and the
    writer of the format cannot drift apart.
    """
    nvars = len(fields["chamber_vars"])
    payload = {
        "format_version": np.array(FORMAT_VERSION),
        "order": np.array(int(fields["order"])),
        "characteristic": np.array(int(fields.get("characteristic", 0))),
        "chamber_vars": np.array(list(fields["chamber_vars"]), dtype=np.str_),
        "field": np.array(str(fields.get("field", "Rational Field"))),
        "family": np.array(str(fields.get("family", "morin-a"))),
        "backend": np.array(str(fields.get("backend", "basic-equations"))),
    }
    for name in ("numerator", "multidegree", "normalized_numerator", "vandermonde"):
        exponents, coefficients = pack_polynomial(fields[name], nvars)
        payload[f"{name}__exponents"] = exponents
        payload[f"{name}__coefficients"] = coefficients
    exponents, coefficients, offsets = pack_polynomial_list(fields["denominator_factors"], nvars)
    payload["denominator__exponents"] = exponents
    payload["denominator__coefficients"] = coefficients
    payload["denominator__offsets"] = offsets

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_algebra(order: int, validate: bool = True) -> ChamberAlgebra:
    """Load the artifact for A_order, checking its invariants by default."""
    path = artifact_path(order)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Regenerate it with "
            f"'python -m multidegree.build -d {order}' under SageMath."
        )
    # allow_pickle stays off: an artifact is data, and must never be able to
    # execute anything on being read.
    with np.load(path, allow_pickle=False) as raw:
        version = int(raw["format_version"])
        if version != FORMAT_VERSION:
            raise ValueError(
                f"{path} is format version {version}, this build expects " f"{FORMAT_VERSION}; regenerate it"
            )
        algebra = ChamberAlgebra(
            order=int(raw["order"]),
            chamber_vars=tuple(str(v) for v in raw["chamber_vars"]),
            numerator=unpack_polynomial(raw["numerator__exponents"], raw["numerator__coefficients"]),
            denominator_factors=unpack_polynomial_list(
                raw["denominator__exponents"],
                raw["denominator__coefficients"],
                raw["denominator__offsets"],
            ),
            multidegree=unpack_polynomial(raw["multidegree__exponents"], raw["multidegree__coefficients"]),
            normalized_numerator=unpack_polynomial(
                raw["normalized_numerator__exponents"],
                raw["normalized_numerator__coefficients"],
            ),
            vandermonde=unpack_polynomial(raw["vandermonde__exponents"], raw["vandermonde__coefficients"]),
            field=str(raw["field"]),
            characteristic=int(raw["characteristic"]),
            family=str(raw["family"]),
            backend=str(raw["backend"]),
        )
    if validate:
        algebra.validate()
    return algebra


@dataclass(frozen=True)
class OrbitGeometry:
    """
    Geometry of the orbit closure that the residue formula does not need.

    Written by :mod:`multidegree.geometry`; nothing in the pipeline reads it.
    ``degree`` is computed there two independent ways -- as ``Q_d(1, ..., 1)`` and
    from the Hilbert series -- and the artifact is only written when they agree.
    """

    order: int
    dimension: int
    codimension: int
    degree: int
    ambient_dimension: int
    variables: Tuple[str, ...]
    #: T_d weight of each coordinate, as a row per variable.
    weights: Tuple[Tuple[int, ...], ...]
    #: Coefficients of the Hilbert numerator, indexed by degree.
    hilbert_numerator: Tuple[int, ...]
    #: Irreducible factors of Q_d in z, with multiplicities.
    factors: Tuple[Tuple[Poly, int], ...]
    #: Components of the initial ideal: coordinate indices, and multiplicity.
    components: Tuple[Tuple[Tuple[int, ...], int], ...]

    @property
    def factors_are_trivial(self) -> bool:
        """True when Q_d is irreducible, i.e. has a single factor of multiplicity 1."""
        return len(self.factors) == 1 and self.factors[0][1] == 1

    def validate(self) -> None:
        if self.codimension != self.ambient_dimension - self.dimension:
            raise ValueError(f"A_{self.order}: dimension and codimension disagree")
        if len(self.weights) != self.ambient_dimension:
            raise ValueError(f"A_{self.order}: one weight per coordinate expected")
        if any(len(indices) != self.codimension for indices, _ in self.components):
            raise ValueError(f"A_{self.order}: a component has the wrong codimension")
        total = sum(total_degree(f) * m for f, m in self.factors)
        if self.factors and total != self.codimension:
            raise ValueError(
                f"A_{self.order}: factor degrees sum to {total}, not deg Q_d = {self.codimension}"
            )


def save_geometry(record: Dict[str, object], path: Path) -> None:
    """Write a :class:`OrbitGeometry` record, in the same pickle-free format."""
    order = int(record["order"])
    payload = {
        "format_version": np.int64(FORMAT_VERSION),
        "order": np.int64(order),
        "field": np.array(str(record["field"]), dtype=np.str_),
        "dimension": np.int64(record["dimension"]),
        "codimension": np.int64(record["codimension"]),
        "degree": np.int64(record["degree"]),
        "ambient_dimension": np.int64(record["ambient_dimension"]),
        "variables": np.array(list(record["variables"]), dtype=np.str_),
        "weights": np.array(record["weights"], dtype=np.int32).reshape(-1, order),
        "hilbert_numerator": _as_int64(record["hilbert_numerator"]),
    }

    polynomials = [factor for factor, _ in record["factors"]]
    exponents, coefficients, offsets = pack_polynomial_list(polynomials, order)
    payload["factor__exponents"] = exponents
    payload["factor__coefficients"] = coefficients
    payload["factor__offsets"] = offsets
    payload["factor__multiplicities"] = _as_int64(m for _, m in record["factors"])

    components = list(record["components"])
    payload["component__indices"] = np.array(
        [i for indices, _ in components for i in indices], dtype=np.int32
    )
    payload["component__offsets"] = np.cumsum([0] + [len(indices) for indices, _ in components]).astype(
        np.int32
    )
    payload["component__multiplicities"] = _as_int64(m for _, m in components)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_geometry(order: int, validate: bool = True) -> OrbitGeometry:
    """Load the geometry record for A_order, checking its invariants by default."""
    path = DATA_DIR / f"a{order}_geometry.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Regenerate it with "
            f"'python -m multidegree.geometry -d {order}' under SageMath."
        )
    with np.load(path, allow_pickle=False) as raw:
        version = int(raw["format_version"])
        if version != FORMAT_VERSION:
            raise ValueError(
                f"{path} is format version {version}, this build expects {FORMAT_VERSION}; " "regenerate it"
            )
        factors = unpack_polynomial_list(
            raw["factor__exponents"], raw["factor__coefficients"], raw["factor__offsets"]
        )
        offsets = raw["component__offsets"]
        indices = raw["component__indices"]
        geometry = OrbitGeometry(
            order=int(raw["order"]),
            dimension=int(raw["dimension"]),
            codimension=int(raw["codimension"]),
            degree=int(raw["degree"]),
            ambient_dimension=int(raw["ambient_dimension"]),
            variables=tuple(str(v) for v in raw["variables"]),
            weights=tuple(tuple(int(x) for x in row) for row in raw["weights"]),
            hilbert_numerator=tuple(int(c) for c in raw["hilbert_numerator"]),
            factors=tuple((factor, int(m)) for factor, m in zip(factors, raw["factor__multiplicities"])),
            components=tuple(
                (tuple(int(i) for i in indices[a:b]), int(m))
                for (a, b), m in zip(zip(offsets[:-1], offsets[1:]), raw["component__multiplicities"])
            ),
        )
    if validate:
        geometry.validate()
    return geometry


def available_geometry_orders() -> List[int]:
    """Orders with a geometry artifact present."""
    return sorted(int(p.stem.split("_")[0][1:]) for p in DATA_DIR.glob("a*_geometry.npz"))
