"""Write (or replace) one obsm embedding in a tier's h5ad in place, without rewriting the crops.

    python add_obsm.py --tier full --key X_ae --npy work/full/X_ae.npy [--row-keys-from data/full_selected.parquet]
The .npy rows must follow the h5ad's obs order; its row count is checked.
"""

import os
import argparse
from pathlib import Path

import anndata as ad
import h5py
import numpy as np

W = Path(os.environ["SBI2_WORK"])
ap = argparse.ArgumentParser()
ap.add_argument("--tier", choices=["small", "full"], required=True)
ap.add_argument("--key", required=True)
ap.add_argument("--npy", required=True)
ap.add_argument("--slice-from-full", action="store_true",
                help="the .npy follows workshop_full.h5ad; take the small tier's rows (highlight != 'other')")
args = ap.parse_args()

x = np.load(W / args.npy).astype(np.float32)
if args.slice_from_full:
    with h5py.File(W / "workshop_full.h5ad", "r") as f:
        full = ad.io.read_elem(f["obs"])
    with h5py.File(W / f"workshop_{args.tier}.h5ad", "r") as f:
        names = ad.io.read_elem(f["obs"]).index
    assert len(x) == len(full)
    mask = (full["highlight"] != "other").to_numpy()
    assert (full.index[mask] == names).all(), "small tier is not the highlight slice of full"
    x = x[mask]
with h5py.File(W / f"workshop_{args.tier}.h5ad", "a") as f:
    n = len(ad.io.read_elem(f["obs"]))
    assert x.shape[0] == n, f"{args.npy} has {x.shape[0]} rows, h5ad has {n}"
    if args.key in f["obsm"]:
        del f["obsm"][args.key]
    ad.io.write_elem(f["obsm"], args.key, x)
print(f"obsm[{args.key!r}] {x.shape} -> workshop_{args.tier}.h5ad")
