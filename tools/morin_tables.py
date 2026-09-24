"""
Exact Thom polynomials of A_d from one level box, written in the registry schema.

C(M) does not depend on the relative dimension, so the level box with
``min M >= -L`` gives Tp_{A_d} at every l <= L - 1 at once.  Usage::

    python tools/morin_tables.py 7 6 results/morin_a7.json
"""

import json
import sys
import time
from pathlib import Path

from chernpp import boxes

PUBLISHED = Path(__file__).resolve().parents[1] / "tests" / "data" / "published_thom_polynomials.json"


def tables_from(d, L, packets):
    out = []
    for l in range(L):
        terms = []
        for m, c in packets.items():
            if min(m) < -(l + 1) or c == 0:
                continue
            idx = sorted(l + 1 + a for a in m if l + 1 + a > 0)
            terms.append({"chern_indices": idx, "coefficient": c})
        terms.sort(key=lambda t: (t["chern_indices"][::-1]), reverse=True)
        out.append(
            {
                "singularity": f"A_{d}",
                "order": d,
                "relative_dimension": l,
                "codimension": d * (l + 1),
                "terms": terms,
            }
        )
    return out


def check_published(tables):
    published = json.loads(PUBLISHED.read_text())["tables"]
    key = lambda t: (t["singularity"], t["relative_dimension"])
    ref = {key(t): {tuple(x["chern_indices"]): x["coefficient"] for x in t["terms"]} for t in published}
    checked = 0
    for t in tables:
        if key(t) in ref:
            ours = {tuple(x["chern_indices"]): x["coefficient"] for x in t["terms"]}
            if ours != ref[key(t)]:
                raise AssertionError(f"{key(t)} disagrees with the published table")
            checked += 1
    return checked


def main(d, L, path):
    t0 = time.time()
    box = boxes.level_box(d, L)
    packets = boxes.chern_table_exact(box)
    tables = tables_from(d, L, packets)
    checked = check_published(tables)
    negative = [m for m, c in packets.items() if c < 0]
    if negative:
        raise AssertionError(f"negative Chern coefficients: {negative[:5]}")
    summary = {
        "d": d,
        "level": L,
        "packets": len(packets),
        "min_chern": min(packets.values()),
        "zero": sum(1 for c in packets.values() if c == 0),
        "published_tables_matched": checked,
        "seconds": round(time.time() - t0, 1),
        "tables": tables,
        "packet_values": [[list(m), c] for m, c in sorted(packets.items())],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(summary))
    print({k: v for k, v in summary.items() if k not in ("tables", "packet_values")})


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
