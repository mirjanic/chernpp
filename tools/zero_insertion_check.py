"""
Check zero-insertion positivity: for every zero-free base M0 in the exact tables,
the Newton coefficients in z of C(M0 + 0^z) and of C - b are nonnegative.

    python tools/zero_insertion_check.py results/zero_insertion.json
"""

import json
import sys
from pathlib import Path

from chernpp import boxes, sectors

RESULTS = Path(__file__).resolve().parent.parent / "results"


def tables():
    out = {d: boxes.chern_table_exact(boxes.level_box(d, L)) for d, L in ((2, 14), (3, 14), (4, 12), (5, 12))}
    for d in (6, 7):
        data = json.loads((RESULTS / f"morin_a{d}.json").read_text())
        out[d] = {tuple(m): c for m, c in data["packet_values"]}
    for path in sorted(RESULTS.glob("deep_*.json")):
        data = json.loads(path.read_text())
        out.setdefault(data["d"], {}).update({tuple(m): c for m, c in data["packet_values"]})
    return out


def main(path):
    by_d = tables()
    bases = {tuple(a for a in m if a != 0) for t in by_d.values() for m in t} - {()}
    checked, negative, certified = 0, [], []
    for base in sorted(bases):
        series = sectors.zero_insertion_series(by_d, base)
        if len(series) < 2:
            continue
        checked += 1
        nc = sectors.newton_coefficients([c for c, _ in series])
        ne = sectors.newton_coefficients([c - b for c, b in series])
        for label, coeffs in (("C", nc), ("C-b", ne)):
            for k, v in enumerate(coeffs):
                if v < 0:
                    negative.append({"base": list(base), "series": label, "k": k, "value": v})
        nz = [k for k, v in enumerate(nc) if v]
        if nz and nz[-1] < len(nc) - 1:
            certified.append({"base": list(base), "degree": nz[-1], "newton": nc[: nz[-1] + 1]})
    out = {"bases_checked": checked, "negative": negative, "certified_degrees": certified}
    Path(path).write_text(json.dumps(out, indent=1))
    print(json.dumps({"bases_checked": checked, "negative": len(negative), "certified": certified}))


if __name__ == "__main__":
    main(sys.argv[1])
