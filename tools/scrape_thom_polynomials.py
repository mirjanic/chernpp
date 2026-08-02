#!/usr/bin/env python3
"""Scrape Rimányi's Thom polynomial registry into normalised JSON.

The registry at https://tpp.web.unc.edu/ publishes, for each relative
dimension ``l``, the Thom polynomials of the contact singularities that occur
there, written in the monomial basis of the relative Chern classes::

    2*c[1]^2*c[4] +3*c[1]*c[2]*c[3] +c[2]^3 +10*c[1]*c[5]

This tool fetches those pages, parses every entry, and emits a term list per
singularity, each term being ``{"chern_indices": [...], "coefficient": int}``
where ``chern_indices`` is the sorted multiset of indices ``i`` of the ``c_i``
appearing in the monomial.  So ``2*c[1]^2*c[4]`` becomes
``{"chern_indices": [1, 1, 4], "coefficient": 2}``.

Everything is checked before it is written.  The weighted degree of every
monomial must equal the codimension the page states for the singularity, every
coefficient must be a non-zero integer, and the codimension of an ``A_d`` must
be ``d * (l + 1)``.  An entry that fails is rejected with a reason and counted
in the per-page summary; it never reaches the output.

Standard library only, so this runs under any Python 3.9+ interpreter.

Usage::

    python tools/scrape_thom_polynomials.py OUTPUT.json
    python tools/scrape_thom_polynomials.py OUTPUT.json --relative-dimension 0 1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

SOURCE = "https://tpp.web.unc.edu/"

#: The l = 1 page does not follow the naming scheme of the other five.
PAGE_SLUGS = {
    0: "thom-polynomials-relative-dimension-l0",
    1: "thom-polynomials-relative-dimension-1",
    2: "thom-polynomials-relative-dimension-l2",
    3: "thom-polynomials-relative-dimension-l3",
    4: "thom-polynomials-relative-dimension-l4",
    5: "thom-polynomials-relative-dimension-l5",
}

USER_AGENT = (
    "chernplusplus-thom-polynomial-scraper/1.0 " "(research use; caches locally, one request per page)"
)

#: Seconds between requests.  The registry is a small academic site.
REQUEST_DELAY = 2.0

DEFAULT_METADATA = {
    "source": SOURCE,
    "method": "Rimanyi restriction equations",
    "note": (
        "Reference Thom polynomials, computed independently of the "
        "Berczi-Szenes residue formula implemented in this repository. Used "
        "to constrain our own computation, never to replace it."
    ),
    "basis": (
        "Relative Chern classes. A term is the multiset of indices i of the "
        "c_i appearing, as a sorted list, so [1,1,1,5] means c_1^3 c_5."
    ),
}


# --------------------------------------------------------------------------
# fetching


def page_url(relative_dimension: int) -> str:
    return f"{SOURCE}{PAGE_SLUGS[relative_dimension]}/"


def fetch(url: str, cache_dir: Path | None = None, delay: bool = True) -> str:
    """Return the HTML of ``url``, hitting the network at most once per run."""
    cached = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / (re.sub(r"[^A-Za-z0-9]+", "_", url).strip("_") + ".html")
        if cached.exists():
            return cached.read_text(encoding="utf-8")
    if delay:
        time.sleep(REQUEST_DELAY)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        raw = response.read()
    text = raw.decode("utf-8", "replace")
    if cached is not None:
        cached.write_text(text, encoding="utf-8")
    return text


# --------------------------------------------------------------------------
# HTML parsing
#
# Each page is a sequence of ``<h4>codimension N</h4>`` headers, and under each
# header an accordion of panels.  A panel's toggle button holds the name of the
# singularity and the collapsed body holds the two expressions::
#
#   <h4>codimension 3 </h4>
#   ...<button id="button-details-3-0" ...>A3</button>...
#   <div id="details-3-0" ...>
#     <div class="panel-body ...">
#       <li>Thom polynomial in monomial basis: </li>
#       <p>c[1]^3 +3*c[1]*c[2] +2*c[3] </p>
#       <li>Thom polynomial in Schur basis: </li>
#       <p>6*s[[3]] +5*s[[2,1]] +s[[1,1,1]]
#     </div>
#   </div>
#
# The markup is tag soup (unclosed <p>, <li> outside any list), so we do not
# try to build a tree; we walk the token stream and emit a flat event list.

# Some headers carry a remark after the number, e.g. "codimension 9 — beyond
# Mather's nice dimensions", so only the prefix is anchored.
CODIMENSION_HEADER = re.compile(r"\s*codimension\s+(\d+)\b", re.IGNORECASE)
MONOMIAL_BASIS = re.compile(
    r"Thom\s+polynomial\s+in\s+monomial\s+basis\s*:\s*(.*?)"
    r"(?:Thom\s+polynomial\s+in\s+\w+\s+basis\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)


class RegistryParser(HTMLParser):
    """Emit ``('codimension', int)``, ``('name', str)`` and ``('body', str)``."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.events: list[tuple[str, object]] = []
        self._mode: str | None = None
        self._buffer: list[str] = []
        self._depth = 0

    # -- helpers

    def _text(self) -> str:
        return "".join(self._buffer)

    def _open(self, mode: str) -> None:
        self._mode = mode
        self._buffer = []

    def _emit(self, kind: str, value: object) -> None:
        self.events.append((kind, value))

    def _reset(self) -> None:
        self._mode = None
        self._buffer = []

    # -- HTMLParser hooks

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key: (value or "") for key, value in attrs}
        if tag == "div":
            if self._mode == "body":
                self._depth += 1
            elif attributes.get("id", "").startswith("details-"):
                self._open("body")
                self._depth = 1
            return
        if self._mode == "body":
            return
        if tag == "h4":
            # The codimension headers carry no class; panel titles do.
            if "panel-title" in attributes.get("class", ""):
                self._mode = None
            else:
                self._open("h4")
        elif tag == "button" and attributes.get("id", "").startswith("button-details"):
            self._open("button")

    def handle_endtag(self, tag: str) -> None:
        if tag == "div" and self._mode == "body":
            self._depth -= 1
            if self._depth <= 0:
                self._emit("body", self._text())
                self._reset()
            return
        if self._mode == "body":
            return
        if tag == "h4" and self._mode == "h4":
            match = CODIMENSION_HEADER.match(self._text())
            if match is not None:  # otherwise an ordinary heading
                self._emit("codimension", int(match.group(1)))
            self._reset()
        elif tag == "button" and self._mode == "button":
            self._emit("name", self._text())
            self._reset()

    def handle_data(self, data: str) -> None:
        if self._mode is not None:
            self._buffer.append(data)


def parse_entries(html: str) -> list[dict]:
    """Return ``{"name", "codimension", "expression"}`` in document order."""
    parser = RegistryParser()
    parser.feed(html)
    parser.close()

    entries: list[dict] = []
    codimension: int | None = None
    name: str | None = None
    for kind, value in parser.events:
        if kind == "codimension":
            codimension = value
        elif kind == "name":
            name = " ".join(str(value).split())
        elif kind == "body":
            if not name:
                continue
            match = MONOMIAL_BASIS.search(str(value))
            entries.append(
                {
                    "name": name,
                    "codimension": codimension,
                    "expression": " ".join(match.group(1).split()) if match else None,
                }
            )
            name = None
    return entries


# --------------------------------------------------------------------------
# names

NAME_PATTERN = re.compile(r"([A-Z]+)_?\{?(\d+)(?:,\s*(\d+))?\}?\Z")

#: Families whose label carries a single index, e.g. A_7.  Everything else in
#: the registry (I, III, B, C, D, E) is a two-index family, except that the
#: "square" members B_4, B_6, C_3, C_6 are written with one index only.
SINGLE_INDEX_FAMILIES = {"A"}

#: I_{a,b} and III_{a,b} always carry two single-digit indices, so "I22" splits
#: as (2, 2).  For B/C/D/E the first digit is the family number and the rest is
#: the position in Rimányi's list, so "C510" splits as (5, 10).
DIGIT_PER_INDEX_FAMILIES = {"I", "II", "III", "IV"}


def canonical_name(raw: str) -> tuple[str, int | None]:
    """Return ``(canonical name, order)``; ``order`` is set only for ``A_d``.

    The registry writes labels without punctuation ("A7", "I22", "C510"); the
    notation is documented at /singularities-in-mathers-nice-dimensions/.  We
    restore the subscripts: ``A7 -> A_7``, ``I22 -> I_2,2``, ``C510 -> C_5,10``.
    Labels that are not of that shape (explicit normal forms such as
    ``(x^2+y^3,y^4)``) are kept verbatim.
    """
    label = " ".join(raw.split())
    match = NAME_PATTERN.fullmatch(label)
    if match is None:
        return label, None
    family, first, second = match.groups()
    if second is not None:
        return f"{family}_{int(first)},{int(second)}", None
    if family in SINGLE_INDEX_FAMILIES:
        return f"{family}_{int(first)}", int(first)
    if family in DIGIT_PER_INDEX_FAMILIES:
        if len(first) != 2:
            return label, None
        return f"{family}_{first[0]},{first[1]}", None
    if len(first) == 1:
        return f"{family}_{first}", None
    return f"{family}_{first[0]},{int(first[1:])}", None


# --------------------------------------------------------------------------
# polynomials

FACTOR = re.compile(r"c\[(\d+)\](?:\^(\d+))?")
MONOMIAL = re.compile(
    r"(?:(\d+)\*)?"  # optional integer coefficient
    r"(c\[\d+\](?:\^\d+)?"  # first Chern factor
    r"(?:\*c\[\d+\](?:\^\d+)?)*)"  # further Chern factors
    r"\Z"
)
INTEGER = re.compile(r"\d+\Z")


class ParseError(ValueError):
    """The expression is not a plain integer polynomial in the c[i]."""


def parse_polynomial(expression: str) -> list[dict]:
    """Turn ``2*c[1]^2*c[4] +c[2]^3`` into a list of normalised terms."""
    text = expression.replace("−", "-").replace("–", "-")
    text = re.sub(r"\s+", "", text)
    if not text:
        raise ParseError("empty expression")

    # Split on the top-level + and - signs.  Monomials contain no signs of
    # their own: exponents and Chern indices are positive integers.
    chunks: list[tuple[int, str]] = []
    sign = 1
    start = 0
    if text[0] in "+-":
        sign = -1 if text[0] == "-" else 1
        start = 1
    for index in range(start, len(text)):
        if text[index] in "+-":
            chunks.append((sign, text[start:index]))
            sign = -1 if text[index] == "-" else 1
            start = index + 1
    chunks.append((sign, text[start:]))

    terms: list[dict] = []
    for chunk_sign, chunk in chunks:
        if not chunk:
            raise ParseError(f"empty term in {expression!r}")
        if INTEGER.fullmatch(chunk):
            # A bare constant: the only Chern-degree-zero monomial.
            terms.append({"chern_indices": [], "coefficient": chunk_sign * int(chunk)})
            continue
        match = MONOMIAL.fullmatch(chunk)
        if match is None:
            raise ParseError(f"unrecognised term {chunk!r} in {expression!r}")
        coefficient = chunk_sign * int(match.group(1) or 1)
        indices: list[int] = []
        for index, exponent in FACTOR.findall(match.group(2)):
            indices.extend([int(index)] * int(exponent or 1))
        terms.append({"chern_indices": sorted(indices), "coefficient": coefficient})
    return terms


# --------------------------------------------------------------------------
# verification


def verify(table: dict) -> None:
    """Raise :class:`ParseError` unless the table is internally consistent."""
    codimension = table["codimension"]
    if not isinstance(codimension, int) or codimension < 0:
        raise ParseError(f"bad codimension {codimension!r}")
    if not table["terms"]:
        raise ParseError("no terms")
    seen: set[tuple[int, ...]] = set()
    for term in table["terms"]:
        indices = term["chern_indices"]
        coefficient = term["coefficient"]
        if not isinstance(coefficient, int) or coefficient == 0:
            raise ParseError(f"bad coefficient {coefficient!r}")
        if any(i < 1 for i in indices):
            raise ParseError(f"non-positive Chern index in {indices!r}")
        if list(indices) != sorted(indices):
            raise ParseError(f"unsorted indices {indices!r}")
        if sum(indices) != codimension:
            raise ParseError(
                f"weighted degree {sum(indices)} of {indices!r} " f"is not the codimension {codimension}"
            )
        key = tuple(indices)
        if key in seen:
            raise ParseError(f"repeated monomial {indices!r}")
        seen.add(key)
    order = table.get("order")
    if order is not None:
        expected = order * (table["relative_dimension"] + 1)
        if codimension != expected:
            raise ParseError(
                f"A_{order} at l={table['relative_dimension']} should have "
                f"codimension {expected}, page says {codimension}"
            )


# --------------------------------------------------------------------------
# scraping


def scrape_page(relative_dimension: int, cache_dir: Path | None, delay: bool) -> tuple[list[dict], list[str]]:
    """Return ``(tables, rejections)`` for one relative dimension."""
    html = fetch(page_url(relative_dimension), cache_dir=cache_dir, delay=delay)
    tables: list[dict] = []
    rejections: list[str] = []
    seen: set[str] = set()
    for entry in parse_entries(html):
        name, order = canonical_name(entry["name"])
        where = f"l={relative_dimension} {name}"
        if name in seen:
            rejections.append(f"{where}: duplicate label on the page")
            continue
        seen.add(name)
        if entry["codimension"] is None:
            rejections.append(f"{where}: no codimension header above the entry")
            continue
        if not entry["expression"]:
            rejections.append(f"{where}: no monomial-basis expression")
            continue
        try:
            terms = parse_polynomial(entry["expression"])
        except ParseError as error:
            rejections.append(f"{where}: {error}")
            continue
        table = {
            "singularity": name,
            "order": order,
            "relative_dimension": relative_dimension,
            "codimension": entry["codimension"],
            "terms": terms,
        }
        try:
            verify(table)
        except ParseError as error:
            rejections.append(f"{where}: {error}")
            continue
        tables.append(table)
    return tables, rejections


def table_key(table: dict) -> tuple:
    """Document order: by codimension, then A_d first, then by name."""
    return (
        table["relative_dimension"],
        table["codimension"],
        0 if table["order"] is not None else 1,
        table["order"] or 0,
        table["singularity"],
    )


def term_map(table: dict) -> dict[tuple[int, ...], int]:
    return {tuple(t["chern_indices"]): t["coefficient"] for t in table["terms"]}


def merge(
    existing: dict | None, tables: list[dict], relative_dimensions: list[int]
) -> tuple[dict, list[str]]:
    """Fold ``tables`` into the previous document, keeping its metadata.

    Tables for relative dimensions we did not scrape are carried over intact.
    A previously recorded table that we did scrape must agree with the new one
    term for term; the check is reported, never silently resolved.
    """
    problems: list[str] = []
    metadata = dict(DEFAULT_METADATA)
    kept: list[dict] = []
    if existing:
        for key in ("source", "method", "note", "basis"):
            if key in existing:
                metadata[key] = existing[key]
        scraped = set(relative_dimensions)
        previous = {}
        for table in existing.get("tables", []):
            identity = (table["relative_dimension"], table["singularity"])
            if table["relative_dimension"] in scraped:
                previous[identity] = table
            else:
                kept.append(table)
        found = {(t["relative_dimension"], t["singularity"]) for t in tables}
        for identity, table in previous.items():
            if identity not in found:
                problems.append(f"l={identity[0]} {identity[1]}: present before, not found now; kept")
                kept.append(table)
                continue
            index = next(
                i for i, t in enumerate(tables) if (t["relative_dimension"], t["singularity"]) == identity
            )
            new = tables[index]
            if term_map(table) != term_map(new):
                problems.append(f"l={identity[0]} {identity[1]}: terms disagree with the previous file")
            elif table["codimension"] != new["codimension"]:
                problems.append(
                    f"l={identity[0]} {identity[1]}: codimension disagrees with the previous file"
                )
            else:
                # Same polynomial.  Reuse the old record verbatim so that a
                # rerun leaves already-published tables byte for byte alone,
                # whatever order the page happens to list the monomials in.
                tables[index] = table

    # The pages of the merged document: the ones we just read, plus the ones
    # whose tables we carried over untouched.
    slugs = {PAGE_SLUGS[l] for l in relative_dimensions}
    slugs.update(PAGE_SLUGS[t["relative_dimension"]] for t in kept)
    document = {
        "source": metadata["source"],
        "pages": [slug for l, slug in sorted(PAGE_SLUGS.items()) if slug in slugs],
        "method": metadata["method"],
        "note": metadata["note"],
        "basis": metadata["basis"],
        "tables": sorted(kept + tables, key=table_key),
    }
    return document, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path, help="JSON file to write")
    parser.add_argument(
        "--relative-dimension",
        "-l",
        type=int,
        nargs="+",
        default=sorted(PAGE_SLUGS),
        metavar="N",
        help="relative dimensions to scrape (default: all six)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="reuse downloaded HTML from this directory instead of refetching",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="ignore any existing output file instead of extending it",
    )
    parser.add_argument("--no-delay", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)

    dimensions = sorted(set(arguments.relative_dimension))
    unknown = [l for l in dimensions if l not in PAGE_SLUGS]
    if unknown:
        parser.error(f"no page for relative dimension(s) {unknown}")

    tables: list[dict] = []
    rejected_total = 0
    for relative_dimension in dimensions:
        page, rejections = scrape_page(relative_dimension, arguments.cache_dir, not arguments.no_delay)
        tables.extend(page)
        rejected_total += len(rejections)
        terms = sum(len(t["terms"]) for t in page)
        print(
            f"{PAGE_SLUGS[relative_dimension]}: {len(page)} entries, "
            f"{terms} terms, {len(rejections)} rejected"
        )
        for reason in rejections:
            print(f"    rejected: {reason}")

    if not tables:
        print("no entries parsed at all; refusing to write", file=sys.stderr)
        return 1

    existing = None
    if arguments.output.exists() and not arguments.no_merge:
        existing = json.loads(arguments.output.read_text(encoding="utf-8"))

    document, problems = merge(existing, tables, dimensions)
    for problem in problems:
        print(f"    regression: {problem}", file=sys.stderr)
    if any("disagree" in problem for problem in problems):
        print("previously published values changed; refusing to write", file=sys.stderr)
        return 1

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    total_terms = sum(len(t["terms"]) for t in document["tables"])
    print(
        f"wrote {arguments.output}: {len(document['tables'])} tables, "
        f"{total_terms} terms, {rejected_total} entries rejected, "
        f"{arguments.output.stat().st_size} bytes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
