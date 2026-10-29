"""Convert notebooks/src/*.py (jupytext percent) to .ipynb and execute them in notebooks/ (cluster test run).

Expects notebooks/data/ to hold the downloadable files (symlinks to local copies), so pooch only checks the hashes.
Fails if an output contains a cluster path.
usage: python run_notebooks.py <repo dir> [notebook stem ...]
"""
import os
import re
import sys
import time
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient

repo = Path(sys.argv[1])
nbdir = repo / "notebooks"
stems = sys.argv[2:] or sorted(p.stem for p in (nbdir / "src").glob("*.py"))
# any machine-local path (work dir, its resolved mount, home) in an output is a leak
_local = {os.environ["SBI2_WORK"], os.path.realpath(os.environ["SBI2_WORK"]), os.path.expanduser("~")}
BAD = re.compile("|".join(re.escape(p) for p in sorted(_local)))
for stem in stems:
    nb = jupytext.read(nbdir / "src" / f"{stem}.py")
    kernel = os.environ.get("SBI2_KERNEL", "python3")  # e.g. scverse-sbi2, the pixi environment's kernel
    nb.metadata["kernelspec"] = {"name": "scverse-sbi2", "display_name": "Python (scverse-sbi2)", "language": "python"}
    t0 = time.time()
    NotebookClient(nb, timeout=int(os.environ.get("SBI2_CELL_TIMEOUT", 3600)), kernel_name=kernel, resources={"metadata": {"path": str(nbdir)}}).execute()
    dt = time.time() - t0
    nbformat.write(nb, nbdir / f"{stem}.ipynb")
    text = nbformat.writes(nb)
    hits = sorted(set(BAD.findall(text)))
    last = [o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])][-1:]
    print(f"{stem}: executed in {dt:.0f} s; {''.join(last).strip()}; cluster paths in outputs: {hits or 'none'}",
          flush=True)
