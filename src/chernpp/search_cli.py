import argparse
import logging
import sys
import time
import os

# Suppress fake JAX cuda allocation errors
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from chernpp.optimisation import gauge

logger = logging.getLogger("chernpp.cli")


def main():
    parser = argparse.ArgumentParser(
        description="Search for a gauge positivity fix for Morin Thom polynomials."
    )
    parser.add_argument(
        "-d",
        "--dimension",
        type=int,
        choices=[5, 6, 7],
        required=True,
        help="The dimension parameter d (5, 6, or 7)",
    )
    parser.add_argument(
        "--depth", type=int, default=30, help="Maximum polynomial degree to fit (default: 30)"
    )
    parser.add_argument(
        "--check-depth", type=int, default=None, help="If set, strictly verify the solution up to this depth"
    )
    parser.add_argument("--bound", type=float, default=20.0, help="Bound on coefficients (default: 20.0)")
    parser.add_argument(
        "--solver", choices=["continuous"], default="continuous", help="Solver backend to use"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        # Suppress overly verbose internal jax logs even in debug mode
        logging.getLogger("jax").setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)

    logger.info(
        f"Searching gauge for d={args.dimension}, depth={args.depth} with {args.solver.upper()} solver..."
    )
    start_time = time.time()

    try:
        res = gauge.solve_positive_gauge_continuous(
            order=args.dimension,
            fit_depth=args.depth,
            bound=args.bound,
        )
        end_time = time.time()

        logger.info(f"Search finished in {end_time - start_time:.2f}s")
        if res.found:
            logger.info("SUCCESS: Positive gauge found!")
            if res.note:
                logger.info(f"Details: {res.note}")
            if hasattr(res, "kernel_coefficients") and res.kernel_coefficients is not None:
                coefs_str = ", ".join(f"{c:.3f}" for c in res.kernel_coefficients)
                logger.info(f"All kernel coefficients: [{coefs_str}]")

            if args.check_depth and args.check_depth > args.depth:
                logger.info(f"Validating out-of-sample at depth {args.check_depth}...")
                v = gauge.validate_gauge(res.kernel, args.dimension, args.check_depth)
                if v.holds:
                    logger.info("Validation SUCCESS: The gauge holds perfectly out-of-sample.")
                else:
                    logger.warning(f"Validation FAILED: {v.negatives_gauged} negatives reappeared.")
        else:
            logger.warning("FAILED: Could not find a positive gauge.")
            if res.note:
                logger.warning(f"Reason: {res.note}")
    except Exception as e:
        logger.error(f"Error during search: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
