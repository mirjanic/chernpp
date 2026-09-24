"""
Exact Chern table on one large box, stored as packet values for later analysis.

    python tools/deep_tables.py charge 7 7      -> results/deep_a7_charge7.json
    python tools/deep_tables.py level 6 13      -> results/deep_a6_level13.json

The ballot and zero-insertion checkers pick up every ``results/deep_*.json``.
"""

import json
import sys
import time
from pathlib import Path

from chernpp import boxes

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main(kind, d, k):
    box = boxes.charge_box(d, k) if kind == "charge" else boxes.level_box(d, k)
    t0 = time.time()
    table = boxes.chern_table_exact(box)
    out = {
        "d": d,
        "box": box.label,
        "cells": box.cells,
        "packets": len(table),
        "min_chern": min(table.values()),
        "seconds": round(time.time() - t0, 1),
        "packet_values": [[list(m), c] for m, c in sorted(table.items())],
    }
    path = RESULTS / f"deep_a{d}_{kind}{k}.json"
    path.write_text(json.dumps(out))
    print({k: v for k, v in out.items() if k != "packet_values"}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
