"""
Small text-table renderer, and statistics for the objects the suite computes.

The interesting objects here -- the multidegree ``Q_d``, the chamber numerator,
the Laurent expansion -- are far too large to print, but their *shape* is
informative: how many terms, how sparse, how large the coefficients grow, how
much cancellation the positivity statement is asking for.  These summaries are
what actually convey the difficulty of each order.
"""

from typing import Dict, Iterable, List, Optional, Sequence

from .polynomial import Poly, negative_terms, total_degree


def table(
    rows: Sequence[Sequence[object]],
    headers: Sequence[str] = (),
    align: Optional[str] = None,
) -> str:
    """
    Render rows as a fixed-width table.

    ``align`` is one character per column, ``"l"`` or ``"r"``; numeric columns
    default to right alignment so magnitudes line up.
    """
    body = [[("" if cell is None else str(cell)) for cell in row] for row in rows]
    ncols = max([len(r) for r in body] + [len(headers)]) if (body or headers) else 0
    body = [r + [""] * (ncols - len(r)) for r in body]
    head = list(headers) + [""] * (ncols - len(headers)) if headers else []

    if align is None:
        align = "".join("r" if all(_looks_numeric(r[c]) for r in body if r[c]) else "l" for c in range(ncols))
    align = (align + "l" * ncols)[:ncols]

    widths = [max([len(r[c]) for r in body] + [len(head[c]) if head else 0]) for c in range(ncols)]

    def render(cells):
        return "  ".join(
            cell.rjust(widths[c]) if align[c] == "r" else cell.ljust(widths[c])
            for c, cell in enumerate(cells)
        ).rstrip()

    lines = []
    if head:
        lines.append(render(head))
        lines.append("  ".join("-" * w for w in widths))
    lines.extend(render(r) for r in body)
    return "\n".join(lines)


def _looks_numeric(text: str) -> bool:
    return bool(text) and text.replace("-", "").replace(".", "").replace(",", "").replace("/", "").isdigit()


def polynomial_stats(p: Poly) -> Dict[str, object]:
    """Shape summary of a polynomial: size, degree, sign split, coefficient range."""
    if not p:
        return {
            "terms": 0,
            "degree": 0,
            "negative": 0,
            "min_coeff": 0,
            "max_coeff": 0,
            "max_abs": 0,
        }
    coefficients = list(p.values())
    return {
        "terms": len(p),
        "degree": total_degree(p),
        "negative": len(negative_terms(p)),
        "min_coeff": min(coefficients),
        "max_coeff": max(coefficients),
        "max_abs": max(abs(c) for c in coefficients),
    }


def algebra_report(orders: Iterable[int] = (4, 5, 6)) -> str:
    """
    A table of the mined chamber algebras: how big each object actually is.

    ``deg Q_d`` is the codimension of the orbit closure, forced by homogeneity
    of the residue formula.  The numerator's negative-term count is a direct
    measure of how much cancellation the positivity statements ask for.
    """
    from math import comb

    from .artifacts import load_algebra

    rows: List[List[object]] = []
    for order in orders:
        algebra = load_algebra(order)
        multidegree = polynomial_stats(algebra.multidegree)
        numerator = polynomial_stats(algebra.numerator)
        rows.append(
            [
                f"A_{order}",
                algebra.nvars,
                len(algebra.denominator_factors),
                len(algebra.denominator_factors) - comb(order, 2),
                multidegree["terms"],
                numerator["terms"],
                numerator["degree"],
                numerator["negative"],
                f"{100 * numerator['negative'] / numerator['terms']:.0f}%",
                numerator["max_abs"],
            ]
        )
    return table(
        rows,
        headers=[
            "",
            "vars",
            "dim N_d",
            "deg Q_d",
            "Q_d terms",
            "N_d terms",
            "deg N_d",
            "N_d neg",
            "neg %",
            "max |coeff|",
        ],
    )


def series_report(orders: Iterable[int] = (4, 5, 6), max_deg: int = 10) -> str:
    """
    A table of the chamber series: how the negative Laurent coefficients behave.

    Strong Laurent positivity is the claim that the ``negative`` column is zero;
    it holds at ``d = 4`` and fails from ``d = 5``.
    """
    from .chamber import chamber_series, sorted_negatives

    rows: List[List[object]] = []
    for order in orders:
        series = chamber_series(order, max_deg)
        negatives = sorted_negatives(series)
        rows.append(
            [
                f"A_{order}",
                len(series),
                len(negatives),
                negatives[0][0] if negatives else "--",
                sum(negatives[0][0]) if negatives else "--",
                min((c for _, c in negatives), default="--"),
                len([b for b, _ in negatives if b[0] == 0]),
            ]
        )
    return table(
        rows,
        headers=[
            "",
            f"terms (deg<={max_deg})",
            "negative",
            "first negative",
            "its degree",
            "most negative",
            "with i=0",
        ],
    )
