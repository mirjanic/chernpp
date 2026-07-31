"""
Compute equivariant multidegrees and export the algebras consumed by ``chernpp``.

Run under SageMath from the ``src`` directory::

    "$(dirname "$(command -v sage)")/python" -m multidegree.build -d 6

The algorithm is selectable, so a different route to the multidegree -- or a
singularity family beyond Morin A_d -- can be dropped in without touching the
export path::

    ... -m multidegree.build --list-backends
    ... -m multidegree.build -d 6 --backend basic-equations

See :mod:`multidegree.backends` for how to add one.
"""

import argparse
import pickle
from pathlib import Path

from sage.all import GF, QQ

from . import backends, get_logger, morin

logger = get_logger(__name__)

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "chernpp" / "data"
DEFAULT_BACKEND = "basic-equations"

#: How each family turns a multidegree into an exportable artifact.  A new
#: family supplies its own assembly function of the same signature.
ASSEMBLERS = {morin.FAMILY: morin.chamber_algebra}


def build(
    orders,
    characteristic=0,
    out_dir=DEFAULT_OUTPUT,
    backend_name=DEFAULT_BACKEND,
    family=morin.FAMILY,
):
    """Compute and export one artifact per order in ``orders``."""
    backend = backends.get(backend_name)
    if not backend.supports(family):
        supported = ", ".join(b.name for b in backends.for_family(family)) or "none"
        raise ValueError(
            f"backend {backend_name!r} does not support family {family!r}; "
            f"backends that do: {supported}"
        )
    if family not in ASSEMBLERS:
        raise ValueError(f"no artifact assembler registered for family {family!r}")
    assemble = ASSEMBLERS[family]

    base_field = QQ if characteristic == 0 else GF(characteristic)
    out_dir.mkdir(parents=True, exist_ok=True)

    for order in orders:
        logger.info("=" * 60)
        logger.info("%s order %d via backend %r", family, order, backend.name)
        result = backend.compute(family, order, base_field)
        artifact = assemble(
            order, result.polynomial, result.ring, base_field, characteristic
        )
        artifact["family"] = family
        artifact["backend"] = backend.name

        path = out_dir / f"a{order}_algebra.pkl"
        with open(path, "wb") as handle:
            pickle.dump(artifact, handle)
        logger.info("A_%d: wrote %s", order, path)


def _print_backends():
    print("Available multidegree backends:\n")
    for backend in backends.available():
        print(f"  {backend.name}")
        print(f"      families: {', '.join(backend.families)}")
        print(f"      {backend.description}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--order", type=int, default=6, help="highest order to build")
    parser.add_argument(
        "--from-order", type=int, default=4, help="lowest order to build (default 4)"
    )
    parser.add_argument(
        "-b",
        "--backend",
        default=DEFAULT_BACKEND,
        help=f"multidegree algorithm to use (default {DEFAULT_BACKEND})",
    )
    parser.add_argument(
        "-f",
        "--family",
        default=morin.FAMILY,
        help=f"singularity family (default {morin.FAMILY})",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="show the registered backends and exit",
    )
    parser.add_argument(
        "-p", "--prime", type=int, default=0, help="work over GF(p); 0 means QQ"
    )
    parser.add_argument("-o", "--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.list_backends:
        _print_backends()
        return

    build(
        orders=range(args.from_order, args.order + 1),
        characteristic=args.prime,
        out_dir=args.out,
        backend_name=args.backend,
        family=args.family,
    )


if __name__ == "__main__":
    main()
