"""Small tier = the full tier restricted to the 40 curated compounds + DMSO (obs["highlight"] != "other").

Same normalization (whole plate, computed on the full tier), features, var, embeddings and obs columns, so
workshop_small.h5ad is a row subset of workshop_full.h5ad. Crops are rewritten with the same HDF5 chunking and
compression as the full tier.

sbatch: scripts/run.sh 4 64G 60 scripts/derive_small.py
"""

import os
from pathlib import Path

import anndata as ad
import h5py
import numpy as np

W = Path(os.environ["SBI2_WORK"])
FULL, SMALL = W / "workshop_full.h5ad", W / "workshop_small.h5ad"
IMG = ("single_cell_images", "masks")

with h5py.File(FULL, "r") as f:
    obs = ad.io.read_elem(f["obs"])
    keep = np.flatnonzero((obs["highlight"] != "other").to_numpy())
    parts = {k: ad.io.read_elem(f[k]) for k in ("X", "var", "uns")}
    layers = {k: f["layers"][k][keep] for k in f["layers"]}
    obsm = {k: f["obsm"][k][keep] for k in f["obsm"] if k not in IMG}
    imgs = {k: (f["obsm"][k][keep], f["obsm"][k].chunks, f["obsm"][k].compression, f["obsm"][k].compression_opts)
            for k in IMG}
small = ad.AnnData(X=parts["X"][keep], obs=obs.iloc[keep], var=parts["var"], uns={**parts["uns"], "tier": "small"},
                   layers=layers, obsm=obsm)
for c in small.obs.select_dtypes("category"):
    small.obs[c] = small.obs[c].cat.remove_unused_categories()
small.write_h5ad(SMALL, compression="gzip")
with h5py.File(SMALL, "a") as f:
    for k, (arr, chunks, comp, opts) in imgs.items():
        ad.io.write_elem(f["obsm"], k, arr, dataset_kwargs={"chunks": chunks, "compression": comp,
                                                            "compression_opts": opts})
print(f"small tier: {len(keep)} cells, {small.obs.compound.nunique()} compounds (incl. DMSO)")
print(small.obs.groupby("Metadata_Source", observed=True).agg(cells=("compound", "size"),
                                                              dmso=("is_control", "sum")))
print(ad.read_h5ad(SMALL, backed="r"))
