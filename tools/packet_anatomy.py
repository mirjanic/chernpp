"""
Per-packet anatomy on charge boxes: every packet with max M <= p is complete.

For each packet: C(M), positive and negative mass, the largest and the
dominant-ordering term.  Usage::

    python tools/packet_anatomy.py results/anatomy_a7.json 7 2 3 4 5
"""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from chernpp import boxes


def main(path, d, charges):
    out = {}
    for p in charges:
        t0 = time.time()
        box = boxes.charge_box(d, p)
        grid = boxes.exact_box(box)
        rows = [asdict(boxes.packet_stats(grid, m)) for m in boxes.complete_packets(box) if max(m) == p]
        neg = [r for r in rows if r["negative"]]
        summary = {
            "packets": len(rows),
            "with_negatives": len(neg),
            "min_chern": min(r["chern"] for r in rows),
            "min_A": min(r["smallest"] for r in rows),
            "max_negative_over_largest": max((r["negative_mass"] / r["largest"] for r in neg), default=0.0),
            "worst": max(neg, key=lambda r: r["negative_mass"] / r["largest"])["multiset"] if neg else None,
            "seconds": round(time.time() - t0, 1),
        }
        out[p] = {"summary": summary, "packets": rows}
        print(p, summary, flush=True)
        Path(path).write_text(json.dumps({"d": d, "layers": out}))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), [int(a) for a in sys.argv[3:]])
