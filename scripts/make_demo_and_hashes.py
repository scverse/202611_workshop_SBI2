"""Write openphenom_demo_crops224.h5 (256 small-tier cells at 224 px, the resolution X_openphenom was computed at)
and print sha256 for every downloadable file (for notebooks/data.py)."""
import os
import hashlib
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import zarr

W = Path(os.environ["SBI2_WORK"])
with h5py.File(W / "workshop_small.h5ad", "r") as f:
    obs = ad.io.read_elem(f["obs"])
crops = zarr.open_group(W / "work/full/crops224.zarr", mode="r")
full_keys = ad.io.read_elem(crops["obs"]).row_key.to_numpy()
pos = {k: i for i, k in enumerate(full_keys)}
rng = np.random.default_rng(0)
take = np.sort(rng.choice(np.flatnonzero(~obs.qc_fail.to_numpy()), 256, replace=False))
keys = obs.index.to_numpy()[take]
rows = np.array([pos[k] for k in keys])
order = np.argsort(rows)
imgs = np.empty((256, 5, 224, 224), np.uint8)
imgs[order] = crops["images"].oindex[np.sort(rows)]
with h5py.File(W / "openphenom_demo_crops224.h5", "w") as f:
    f.create_dataset("images", data=imgs, chunks=(16, 5, 224, 224), compression="gzip", compression_opts=4)
    f.create_dataset("row_key", data=keys.astype("S"))
    f.attrs["channels"] = ["DNA", "AGP", "ER", "Mito", "RNA"]
    f.attrs["description"] = "256 workshop_small cells, 224 px crops (input of X_openphenom), rows in small-tier order"

for name in ["workshop_small.h5ad", "workshop_full.h5ad", "workshop_full_collection.zarr.zip", "autoencoder.pt",
             "autoencoder_full.pt", "openphenom_demo_crops224.h5"]:
    h = hashlib.sha256()
    with open(W / name, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    print(f'    "{name}": "sha256:{h.hexdigest()}",  # {(W / name).stat().st_size / 1e6:.0f} MB')
