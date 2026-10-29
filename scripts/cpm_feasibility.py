"""cp_measure feasibility test: per-cell features on 200 crops (nucleus + cell objects, 5 channels).

Each crop is featurized on its own (one label per object mask: mask > 0 -> 1), images scaled to [0, 1].
Reports feature count, time per cell (default numpy backend and the numba accelerator), failures.
sbatch: scripts/run.sh 8 32G 120 scripts/cpm_feasibility.py
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"  # time a single core, the unit for an array job

import time
import traceback
import warnings
from pathlib import Path

import cp_measure
import numpy as np
import pandas as pd
import zarr
from cp_measure.featurizer import featurize, make_featurizer_config

W = Path(os.environ["SBI2_WORK"])
N = 200

g = zarr.open_group(W / "work/crops224.zarr", mode="r")
n_total = g["images"].shape[0]
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(n_total, N, replace=False))
imgs = g["images"].oindex[idx].astype(np.float32) / 255.0
masks = (g["masks"].oindex[idx] > 0).astype(np.int32)

empty = (masks.reshape(N, 2, -1).sum(-1) == 0)
print(f"empty masks among {N}: nucleus {empty[:, 0].sum()}, cell {empty[:, 1].sum()}")
config = make_featurizer_config(["DNA", "AGP", "ER", "Mito", "RNA"], objects=["nucleus", "cell"],
                                # images are 8-bit scaled to [0, 1]; Costes needs the original bit depth
                                correlation_costes_params={"scale_max": 255})

for acc in (None, "numba"):
    cp_measure.set_accelerator(acc)
    # warm-up (numba JIT), not timed
    featurize(imgs[:1], masks[:1], config)
    times, rows, fails, warns = [], [], [], {}
    for i in range(N):
        t = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                data, cols, r = featurize(imgs[i:i + 1], masks[i:i + 1], config)
            for x in w:
                key = f"{x.category.__name__}: {str(x.message)[:120]}"
                warns[key] = warns.get(key, 0) + 1
            # one row per object -> one wide row per cell
            wide = {f"{obj}_{c}": v for (img_id, obj, lab), row in zip(r, data) for c, v in zip(cols, row)}
            rows.append(wide)
        except Exception as e:  # noqa: BLE001
            fails.append((int(idx[i]), repr(e)[:200]))
            if len(fails) == 1:
                traceback.print_exc()
        times.append(time.perf_counter() - t)
    df = pd.DataFrame(rows)
    times = np.array(times)
    print(f"\n== accelerator={acc}")
    print(f"features per cell: {df.shape[1]} (per object: {len(cols)}); cells ok {len(df)}/{N}; failures {len(fails)}")
    print(f"time/cell: mean {times.mean():.3f}s median {np.median(times):.3f}s p95 {np.percentile(times, 95):.3f}s")
    print(f"est. single-core time for {n_total} cells: {times.mean() * n_total / 3600:.2f} h")
    nan_cols = df.columns[df.isna().any()]
    print(f"columns with any NaN: {len(nan_cols)}; all-NaN: {int(df.isna().all().sum())}; "
          f"constant: {int((df.nunique() <= 1).sum())}")
    print("NaN columns sample:", list(nan_cols[:10]))
    for k, v in sorted(warns.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  warning x{v}: {k}")
    for f in fails[:5]:
        print("  fail:", f)
    groups = pd.Series([c.split("_", 2)[1] for c in df.columns]).value_counts()
    print("feature groups (prefix after object):", groups.to_dict())
    df.index = idx[: len(df)] if not fails else None
    df.to_parquet(W / f"work/cpm_test_{acc}.parquet")

if (W / "work/cpm_test_None.parquet").exists() and (W / "work/cpm_test_numba.parquet").exists():
    a, b = pd.read_parquet(W / "work/cpm_test_None.parquet"), pd.read_parquet(W / "work/cpm_test_numba.parquet")
    common = a.columns.intersection(b.columns)
    diff = np.nanmax(np.abs(a[common].to_numpy() - b[common].to_numpy()) / (np.abs(a[common].to_numpy()) + 1e-6), axis=0)
    print(f"\nnumpy vs numba: {len(common)} common columns, max rel diff > 1e-3 in {(diff > 1e-3).sum()} columns")
