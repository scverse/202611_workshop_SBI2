"""cp_measure per-cell features on the 224 px crops (SLURM array; one part file per task, skipped if present).

Objects: nucleus and cell masks (mask > 0 -> label 1), all 5 channels, images scaled to [0, 1].
Costes needs the original bit depth: scale_max=255. One wide row per cell, columns '<object>_<feature>'.

sbatch --array=0-19 ... scripts/cpm_features.py --ntasks 20
"""

import argparse
import os
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import zarr
from cp_measure.featurizer import featurize, make_featurizer_config

W = Path(os.environ["SBI2_WORK"])
CHANNELS = ["DNA", "AGP", "ER", "Mito", "RNA"]
OBJECTS = ["nucleus", "cell"]

ap = argparse.ArgumentParser()
ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
ap.add_argument("--ntasks", type=int, default=20)
ap.add_argument("--work", default=str(W / "work"), help="reads crops224.zarr, writes cpm/part_*.parquet")
args = ap.parse_args()

work = Path(args.work)
out = work / f"cpm/part_{args.task:03d}.parquet"
if out.exists():
    print(f"{out} exists, skipping")
    raise SystemExit
out.parent.mkdir(parents=True, exist_ok=True)

g = zarr.open_group(work / "crops224.zarr", mode="r")
n = g["images"].shape[0]
lo, hi = np.array_split(np.arange(n), args.ntasks)[args.task][[0, -1]]
hi += 1
imgs = g["images"][lo:hi].astype(np.float32) / 255.0
masks = (g["masks"][lo:hi] > 0).astype(np.int32)
row_keys = ad.io.read_elem(g["obs"]).row_key.to_numpy()[lo:hi]

config = make_featurizer_config(CHANNELS, objects=OBJECTS, correlation_costes_params={"scale_max": 255})
t0 = time.time()
rows = []
for i in range(hi - lo):
    if not masks[i].any():  # featurize raises on a batch without any labelled object
        rows.append({})
        continue
    data, cols, r = featurize(imgs[i:i + 1], masks[i:i + 1], config)
    rows.append({f"{obj}_{c}": v for (_, obj, _), row in zip(r, data) for c, v in zip(cols, row)})
    if i % 100 == 0:
        print(f"{i}/{hi - lo} {time.time() - t0:.0f}s", flush=True)
df = pd.DataFrame(rows, index=pd.RangeIndex(lo, hi, name="cell_idx"))
# a cell whose nucleus mask is empty yields no nucleus row; keep the full column set
full = [f"{o}_{c}" for o in OBJECTS for c in cols]
df = df.reindex(columns=full).astype(np.float32)
df.insert(0, "row_key", row_keys)
df.to_parquet(out)
print(f"done {hi - lo} cells in {time.time() - t0:.0f}s ({(time.time() - t0) / (hi - lo):.2f} s/cell)")
