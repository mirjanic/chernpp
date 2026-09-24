"""Compute the 0-Hecke positivity landscape P_d on level boxes; append to a JSON file."""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from chernpp import boxes, hecke


def main(path, runs):
    path = Path(path)
    out = json.loads(path.read_text()) if path.exists() else []
    for d, L in runs:
        t0 = time.time()
        row = asdict(hecke.landscape(d, boxes.level_box(d, L)))
        row["level"] = L
        row["seconds"] = round(time.time() - t0, 1)
        out.append(row)
        path.write_text(json.dumps(out, indent=1))
        print(json.dumps({k: v for k, v in row.items() if k != "witnesses"}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], [tuple(map(int, a.split(","))) for a in sys.argv[2:]])
