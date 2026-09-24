"""
Check the ballot conjecture C(M) >= b(M) (equality exactly on max M <= 1) on every
exact Chern table: d <= 5 from fresh level boxes, d = 6, 7 from ``results/``.

    python tools/ballot_check.py results/ballot_conjecture.json
"""

import json
import sys
from fractions import Fraction
from pathlib import Path

from chernpp import boxes, sectors

RESULTS = Path(__file__).resolve().parent.parent / "results"
FRESH = {2: 10, 3: 10, 4: 8, 5: 8}


def tables():
    for d, L in FRESH.items():
        yield d, f"level box L={L}", boxes.chern_table_exact(boxes.level_box(d, L))
    for d in (6, 7):
        data = json.loads((RESULTS / f"morin_a{d}.json").read_text())
        yield d, f"level box L={data['level']}", {tuple(m): c for m, c in data["packet_values"]}


def main(path):
    out = []
    for d, box, table in tables():
        violations, plane, equal = sectors.ballot_violations(table)
        off = [(Fraction(c, sectors.ballot_count(m)), m) for m, c in table.items() if not sectors.is_plane(m)]
        ratio, arg = min(off) if off else (None, None)
        row = {
            "d": d,
            "box": box,
            "packets": len(table),
            "violations": [list(v[0]) for v in violations],
            "plane_mismatches": [list(v[0]) for v in plane],
            "nonplane_equalities": [list(v[0]) for v in equal],
            "min_nonplane_ratio": str(ratio),
            "min_nonplane_ratio_at": list(arg) if arg else None,
        }
        out.append(row)
        print(json.dumps({k: v for k, v in row.items()}), flush=True)
    Path(path).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
