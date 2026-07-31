"""
Execute a notebook in a real kernel and store the outputs in place.

    python tools/execute_notebook.py src/examples.ipynb src

``examples.ipynb`` is committed *with* its outputs so it reads as a document,
which means the outputs have to be regenerated whenever the package changes.
This does that without pulling in nbconvert: ``jupyter_client`` and
``ipykernel`` are enough, and both are already needed for notebook work.

Exits nonzero if any cell errors, so it can be used as a check.
"""

import json
import queue
import sys
from pathlib import Path

from jupyter_client.manager import start_new_kernel

TIMEOUT = 600


def execute(path: Path, cwd: str) -> int:
    notebook = json.loads(path.read_text())
    manager, client = start_new_kernel(kernel_name="python3", cwd=cwd)
    failures = 0
    try:
        index = 0
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            index += 1
            message_id = client.execute("".join(cell["source"]))
            outputs, count = [], None
            while True:
                try:
                    message = client.get_iopub_msg(timeout=TIMEOUT)
                except queue.Empty:
                    print(f"cell {index}: TIMEOUT")
                    failures += 1
                    break
                if message["parent_header"].get("msg_id") != message_id:
                    continue
                kind, content = message["msg_type"], message["content"]
                if kind == "status" and content["execution_state"] == "idle":
                    break
                if kind == "stream":
                    outputs.append(
                        {
                            "output_type": "stream",
                            "name": content["name"],
                            "text": content["text"].splitlines(keepends=True),
                        }
                    )
                elif kind in ("execute_result", "display_data"):
                    output = {
                        "output_type": kind,
                        "data": content["data"],
                        "metadata": content.get("metadata", {}),
                    }
                    if kind == "execute_result":
                        output["execution_count"] = content.get("execution_count")
                    outputs.append(output)
                elif kind == "error":
                    outputs.append(
                        {
                            "output_type": "error",
                            "ename": content["ename"],
                            "evalue": content["evalue"],
                            "traceback": content["traceback"],
                        }
                    )
                    failures += 1
                    print(f"cell {index}: ERROR {content['ename']}: {content['evalue']}")
                elif kind == "execute_input":
                    count = content.get("execution_count")
            cell["outputs"] = outputs
            cell["execution_count"] = count
            print(f"cell {index}: ok ({len(outputs)} outputs)", flush=True)
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)

    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
    print("failures:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "src/examples.ipynb")
    sys.exit(execute(target, sys.argv[2] if len(sys.argv) > 2 else str(target.parent)))
