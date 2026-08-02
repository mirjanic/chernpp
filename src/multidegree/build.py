"""
Compute equivariant multidegrees and export the algebras consumed by ``chernpp``.

Run under SageMath from the ``src`` directory::

    "$(dirname "$(command -v sage)")/python" -m multidegree.build -d 6

The model lives in :mod:`multidegree.morin`, the algorithm in
:mod:`multidegree.basic_equations`, and the export path here knows about neither
beyond the :class:`~multidegree.Multidegree` they agree on.
"""

import argparse
from pathlib import Path

from sage.all import GF, QQ

from . import get_logger, morin
from . import basic_equations

logger = get_logger(__name__)

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "chernpp" / "data"


def build(orders, characteristic=0, out_dir=DEFAULT_OUTPUT):
    """Compute and export one artifact per order in ``orders``."""
    base_field = QQ if characteristic == 0 else GF(characteristic)
    out_dir.mkdir(parents=True, exist_ok=True)

    for order in orders:
        logger.info("=" * 60)
        logger.info("%s order %d", morin.FAMILY, order)
        result = basic_equations.compute(order, base_field)
        artifact = morin.chamber_algebra(order, result.polynomial, result.ring, base_field, characteristic)
        artifact["family"] = result.family
        artifact["backend"] = basic_equations.NAME

        # The writer lives with the reader, in chernpp.artifacts, so the two
        # cannot drift.  That module imports only NumPy.
        from chernpp.artifacts import save_algebra

        path = out_dir / f"a{order}_algebra.npz"
        save_algebra(artifact, path)
        logger.info("A_%d: wrote %s", order, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--order", type=int, default=7, help="highest order to build")
    parser.add_argument("--from-order", type=int, default=1, help="lowest order to build (default 1)")
    parser.add_argument("-p", "--prime", type=int, default=0, help="work over GF(p); 0 means QQ")
    parser.add_argument("-o", "--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    build(
        orders=range(args.from_order, args.order + 1),
        characteristic=args.prime,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
