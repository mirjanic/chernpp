"""
Experiment runner for a single Morin singularity A_d.

    python -m chernpp.experiments --dim 6

Five independent sections -- the classical Thom polynomial, the negative
Laurent coefficients, the structural reductions, the Rimányi sweep over
relative dimension, and the denominator certificates.  Use ``--only`` to run a
subset.
"""

import argparse
import logging
import math
import sys

from .artifacts import load_algebra
from .certificates import minimum_order, search_certificate
from .chern import chern_coefficients, thom_polynomial
from .chamber import (
    chamber_series,
    monomial,
    paired_defects,
    sorted_negatives,
    unpaired_tail_defects,
)

logger = logging.getLogger("experiments")


def _setup_logging(logfile=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if logfile:
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers, force=True)


def section(title):
    logger.info("\n--- %s", title)


def classical_thom(dim, **_):
    """The classical Thom polynomial at relative dimension l = 0."""
    section(f"Classical Thom polynomial of A_{dim} at l = 0")
    logger.info("  Tp = %s", thom_polynomial(dim=dim, l_max=0))
    logger.info(
        "  (the coefficient of c_%d must be (%d-1)! = %d)",
        dim,
        dim,
        math.factorial(dim - 1),
    )


def laurent_negatives(dim, degree, **_):
    """Negative coefficients of the chamber series -- strong Laurent positivity."""
    section(f"Chamber series F_{dim}, truncated at total degree {degree}")
    alg = load_algebra(dim)
    F = chamber_series(dim, degree, algebra=alg)
    negs = sorted_negatives(F)
    logger.info("  chamber variables: %s", ", ".join(alg.chamber_vars))
    logger.info("  %d coefficients, %d negative", len(F), len(negs))
    if not negs:
        logger.info("  strong Laurent positivity holds in this range")
        return
    logger.info("  lowest-degree negatives:")
    for beta, c in negs[:6]:
        logger.info("    A_%s = %d   (total degree %d)", beta, c, sum(beta))


def reductions(dim, degree, **_):
    """The structural reductions from the A_5 handoff note, evaluated at this d."""
    section(f"Structural reductions for A_{dim} (degree <= {degree})")
    alg = load_algebra(dim)
    nvars = len(alg.chamber_vars)
    F = chamber_series(dim, degree, algebra=alg)
    B = chamber_series(dim, degree, extra_factors=[monomial(nvars, 0)], algebra=alg)

    tail = unpaired_tail_defects(F, degree)
    logger.info(
        "  unpaired tail   i > j  =>  A >= 0 :  %s",
        "holds" if not tail else f"FAILS, e.g. A_{tail[0][0]} = {tail[0][1]}",
    )
    pair = paired_defects(F, degree)
    logger.info(
        "  paired          A_b + A_tau(b) >= 0 :  %s",
        "holds" if not pair else f"FAILS, e.g. beta = {pair[0][0]}, sum = {pair[0][1]}",
    )
    pref = sorted_negatives(B)
    logger.info(
        "  prefix          F_%d/(1-a) >= 0 :  %s",
        dim,
        ("holds" if not pref else f"FAILS, {len(pref)} negative, e.g. A_{pref[0][0]} = {pref[0][1]}"),
    )
    if pref:
        logger.info(
            "    the prefix sum runs over r <= i, so negatives with i = 0 survive it: %s",
            [b for b, _ in sorted_negatives(F) if b[0] == 0][:4],
        )


def rimanyi(dim, max_l, **_):
    """Rimanyi weak Chern positivity: C(M) >= 0, swept over relative dimension."""
    section(f"Rimanyi weak Chern positivity for A_{dim}, l = 0..{max_l}")
    for l in range(max_l + 1):
        try:
            _, chern = chern_coefficients(dim=dim, l_max=l)
        except OverflowError as ex:
            logger.info("  l=%d: stopped -- %s", l, ex)
            break
        worst = min(chern.values())
        nneg = sum(1 for v in chern.values() if v < 0)
        logger.info(
            "  l=%2d: %6d Chern monomials, min coefficient %s  -> %s",
            l,
            len(chern),
            worst,
            "PASS" if nneg == 0 else f"FAIL ({nneg} negative)",
        )
        if nneg:
            break


def certificates(dim, cert_order, cert_degree, probe, **_):
    """Denominator certificates for coefficientwise positivity of F_d."""
    section(f"Denominator certificates for A_{dim}")
    alg = load_algebra(dim)
    nvars = len(alg.chamber_vars)
    N, fs = alg.numerator, alg.denominator_factors

    lo = minimum_order(N, fs, nvars, probe_degree=probe, max_order=cert_order)
    if lo is None:
        logger.info(
            "  no certificate of order <= %d exists at any degree "
            "(degree-<=%d projection of the LP is infeasible)",
            cert_order,
            probe,
        )
        return
    logger.info(
        "  order <= %d is impossible at any degree; least unobstructed order is %d",
        lo - 1,
        lo,
    )
    for order in range(lo, cert_order + 1):
        cert = search_certificate(
            N,
            fs,
            nvars,
            order=order,
            max_degree=cert_degree,
            varnames=alg.chamber_vars,
        )
        if cert:
            logger.info("  FOUND and exactly verified:\n%s", cert.summary())
            return
        logger.info("  order %d, deg <= %d: none", order, cert_degree)


SECTIONS = {
    "thom": classical_thom,
    "laurent": laurent_negatives,
    "reductions": reductions,
    "rimanyi": rimanyi,
    "certificates": certificates,
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-d", "--dim", type=int, default=6, help="Morin order d (artifacts ship for 1 through 7)")
    p.add_argument("--degree", type=int, default=12, help="chamber-series truncation degree")
    p.add_argument("--max-l", type=int, default=5, help="highest relative dimension to sweep")
    p.add_argument("--cert-order", type=int, default=4, help="highest certificate order to try")
    p.add_argument("--cert-degree", type=int, default=8, help="degree cap on the certificate parts")
    p.add_argument("--probe", type=int, default=2, help="depth of the order-obstruction probe")
    p.add_argument("--only", choices=sorted(SECTIONS), nargs="*", help="run only these sections")
    p.add_argument("--log", help="also append output to this file")
    args = p.parse_args()

    _setup_logging(args.log)
    logger.info("=" * 68)
    logger.info("A_%d Morin singularity -- computational experiments", args.dim)
    logger.info("=" * 68)

    for name in args.only or SECTIONS:
        SECTIONS[name](
            dim=args.dim,
            degree=args.degree,
            max_l=args.max_l,
            cert_order=args.cert_order,
            cert_degree=args.cert_degree,
            probe=args.probe,
        )
    logger.info("")


if __name__ == "__main__":
    main()
