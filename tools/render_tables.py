"""
Render the report's data tables from the artifacts.

The counts in these tables move whenever the pipeline changes -- a variable
reordering alters the component count, a rebuild alters the term counts -- and a
table transcribed by hand into the LaTeX drifts silently.  This writes them, and
`papers/chernpp_report.tex` inputs the result.

    python tools/render_tables.py papers/tables
"""

import argparse
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chernpp.artifacts import load_algebra, load_geometry  # noqa: E402
from chernpp.chamber import chamber_series  # noqa: E402
from chernpp.chern import chern_coefficients  # noqa: E402
from chernpp.tables import ALL_ORDERS, latex_table, polynomial_stats  # noqa: E402

#: Truncation degree used for the cancellation table, per order.
CANCELLATION_DEPTH = {4: 12, 5: 12, 6: 10, 7: 10}


def algebra_rows(orders):
    for order in orders:
        algebra = load_algebra(order)
        multidegree = polynomial_stats(algebra.multidegree)
        numerator = polynomial_stats(algebra.numerator)
        share = (
            f"{numerator['negative']} \\ ({100 * numerator['negative'] / numerator['terms']:.0f}\\%)"
            if numerator["terms"]
            else "---"
        )
        yield [
            f"$A_{order}$",
            len(algebra.denominator_factors),
            len(algebra.denominator_factors) - comb(order, 2),
            multidegree["terms"],
            multidegree["max_abs"],
            len(algebra.vandermonde),
            numerator["terms"],
            numerator["max_abs"],
            share,
        ]


def geometry_rows(orders):
    for order in orders:
        geometry = load_geometry(order)
        multiplicities = [m for _, m in geometry.components]
        degrees = sorted(sum(e) for f, m in geometry.factors for _ in range(m) for e in [max(f, key=sum)])
        degrees = sorted(
            sum(exponents)
            for factor, m in geometry.factors
            for _ in range(m)
            for exponents in [max(factor, key=sum)]
        )
        shape = (
            "$1$"
            if not degrees
            else ("irreducible" if len(degrees) == 1 else " $\\cdot$ ".join(f"${e}$" for e in degrees))
        )
        yield [
            f"$A_{order}$",
            geometry.ambient_dimension,
            geometry.dimension,
            geometry.codimension,
            geometry.degree,
            len(geometry.components) or "---",
            f"${min(multiplicities)}$--${max(multiplicities)}$" if multiplicities else "---",
            len(geometry.hilbert_numerator) - 1,
            shape,
        ]


def cancellation_rows(orders):
    for order in orders:
        cap = CANCELLATION_DEPTH[order]
        series = chamber_series(order, cap)
        negatives = [v for v in series.values() if v < 0]
        positive = sum(v for v in series.values() if v > 0)
        _, chern = chern_coefficients(order, l_max=0)
        yield [
            f"$A_{order}$",
            cap,
            len(series),
            len(negatives),
            f"${100 * len(negatives) / len(series):.1f}\\%$",
            f"${min(negatives)}$" if negatives else "---",
            -sum(negatives),
            f"${100 * -sum(negatives) / positive:.2f}\\%$",
            min(chern.values()),
        ]


TABLES = {
    "algebra": (
        algebra_rows,
        ALL_ORDERS,
        [
            "",
            "$\\dim\\Nhat_d$",
            "$\\deg\\Qd_d$",
            "$\\Qd_d$ terms",
            "$\\max|\\Qd_d|$",
            "Vand.\\ terms",
            "$N_d$ terms",
            "$\\max|N_d|$",
            "$N_d$ negative",
        ],
        "The mined chamber algebras. $\\deg\\Qd_d = \\dim\\Nhat_d - \\binom{d}{2}$ is the "
        "codimension of the orbit closure; there are $d-1$ chamber variables throughout.",
        "tab:algebra",
        True,
    ),
    "geometry": (
        geometry_rows,
        ALL_ORDERS,
        [
            "",
            "$\\dim\\Nhat_d$",
            "$\\dim\\mathcal{O}_d$",
            "$\\codim$",
            "$\\deg\\mathcal{O}_d$",
            "components",
            "mult",
            "deg Hilb",
            "$\\Qd_d$",
        ],
        "The orbit closure $\\mathcal{O}_d \\subset \\Nhat_d$. ``mult'' is the range of component "
        "multiplicities, ``deg Hilb'' the degree of the Hilbert numerator. For $d \\le 3$ the orbit "
        "fills $\\Nhat_d$ and there is no degeneration.",
        "tab:geom",
        True,
    ),
    "cancellation": (
        cancellation_rows,
        (4, 5, 6, 7),
        [
            "",
            "$\\deg \\le$",
            "$A_\\beta$",
            "negative",
            "\\%",
            "least",
            "neg.\\ mass",
            "of pos.\\",
            "$\\min C(M)$",
        ],
        "Negative Laurent coefficients against the positive mass they sit in, on the truncation "
        "shown, and the Chern coefficients at $\\ell = 0$ they cancel into.",
        "tab:cancel",
        False,
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("out", type=Path, help="directory to write the .tex fragments into")
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)

    for name, (rows, orders, headers, caption, label, small) in TABLES.items():
        body = list(rows(orders))
        rendered = latex_table(body, headers, caption, label, small=small)
        path = arguments.out / f"{name}.tex"
        path.write_text(rendered + "\n")
        print(f"  {path}  ({len(body)} rows)")


if __name__ == "__main__":
    main()
