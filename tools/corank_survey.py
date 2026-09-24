"""
Compute rho(Tp) = min{r : Tp in C_r} for every non-Morin table in the registry.

Resumable: tables already in the output file are skipped.  Usage::

    python tools/corank_survey.py results/corank_survey.json [max_rank] [seconds] [k/n]
    python tools/corank_survey.py --merge results/corank_survey.json

A table that exceeds the per-table time limit is recorded as a timeout -- never
as an answer.  Shard ``k/n`` takes every n-th table and writes
``<path>.shard<k>``; ``--merge`` folds the shards back in.
"""

import json
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

from chernpp.corank import load_registry, survey_table


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _key(row):
    return row["singularity"], row["relative_dimension"]


def main(path, max_rank=6, seconds=0, shard=(0, 1)):
    base = Path(path)
    decided = json.loads(base.read_text()) if base.exists() else []
    seen = {_key(r) for r in decided if "memberships" in r}
    out_path = base if shard[1] == 1 else Path(f"{base}.shard{shard[0]}")
    done = decided if shard[1] == 1 else []
    tables = sorted(
        (t for t in load_registry() if not t.singularity.startswith("A_")),
        key=lambda t: (t.codimension, t.singularity),
    )
    pending = [t for t in tables if (t.singularity, t.relative_dimension) not in seen]
    signal.signal(signal.SIGALRM, _alarm)
    for t in pending[shard[0] :: shard[1]]:
        t0 = time.time()
        signal.alarm(seconds)
        try:
            row = asdict(survey_table(t, ranks=range(1, max_rank + 1), max_generators=300000))
        except _Timeout:
            row = {
                "singularity": t.singularity,
                "relative_dimension": t.relative_dimension,
                "timeout": seconds,
            }
        except Exception as e:  # recorded, not hidden: the row says why
            row = {
                "singularity": t.singularity,
                "relative_dimension": t.relative_dimension,
                "error": repr(e)[:300],
            }
        signal.alarm(0)
        row["codimension"] = t.codimension
        row["seconds"] = round(time.time() - t0, 2)
        done = [r for r in done if _key(r) != _key(row)] + [row]
        out_path.write_text(json.dumps(done, indent=1))
        print(json.dumps(row), flush=True)


def merge(path):
    base = Path(path)
    rows = {_key(r): r for r in (json.loads(base.read_text()) if base.exists() else [])}
    for shard in sorted(base.parent.glob(base.name + ".shard*")):
        for r in json.loads(shard.read_text()):
            if "memberships" in r or _key(r) not in rows:
                rows[_key(r)] = r
        shard.unlink()
    base.write_text(json.dumps(list(rows.values()), indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "--merge":
        merge(sys.argv[2])
    else:
        k, n = map(int, sys.argv[4].split("/")) if len(sys.argv) > 4 else (0, 1)
        main(
            sys.argv[1],
            int(sys.argv[2]) if len(sys.argv) > 2 else 6,
            int(sys.argv[3]) if len(sys.argv) > 3 else 0,
            (k, n),
        )
