"""Scale demo: annbatch pre-shuffled collection of all 645,390 run-05 fine-tuned single-cell embeddings (x 128).

obs: compound (JUMP-Target-2 name where known, else InChIKey), source, plate, well, plate_type, is_control.
Uses annbatch 0.3.1's DatasetCollection.add_adatas with a custom load_adata (X + the obs subset only).
Writes scale_ft_embeddings.zarr and checks it loads with annbatch.Loader.use_collection.

sbatch: scripts/run.sh 8 64G 60 scripts/make_scale_collection.py
"""

import os
import shutil
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import zarr
from annbatch import DatasetCollection, Loader

W = Path(os.environ["SBI2_WORK"])
SRC = Path(os.environ["SBI2_SOURCE"]) / Path("results/embeddings/05_JUMP_OpenPhenom_ECFP4_C_singlecell.h5ad")
OUT = W / "scale_ft_embeddings.zarr"
DMSO = "IAZDPXIOMUYVGZ-UHFFFAOYSA-N"

ann = pd.read_csv(W / "ref/JUMP-Target-2_compound_metadata.tsv", sep="\t").drop_duplicates("InChIKey")
names = dict(zip(ann.InChIKey, ann.pert_iname))


def load(path) -> ad.AnnData:
    a = ad.read_h5ad(path)
    ik = a.obs["Metadata_InChIKey"].astype(str)
    obs = pd.DataFrame({
        "compound": ik.map(names).fillna(ik).where(ik != DMSO, "DMSO"),
        "source": a.obs["source"].astype(str), "plate": a.obs["plate"].astype(str),
        "well": a.obs["well"].astype(str), "plate_type": a.obs["Metadata_PlateType"].astype(str),
        "is_control": (ik == DMSO).to_numpy(),
    }, index=a.obs_names)
    for c in ("compound", "source", "plate", "well", "plate_type"):
        obs[c] = obs[c].astype("category")
    return ad.AnnData(X=np.asarray(a.X, dtype=np.float32), obs=obs)


if OUT.exists():
    shutil.rmtree(OUT)
t0 = time.time()
coll = DatasetCollection(OUT)
coll.add_adatas([SRC], load_adata=load, dataset_size="100MB", n_obs_per_chunk=64, shard_size="64MB",
                rng=np.random.default_rng(0))
print(f"collection written in {time.time() - t0:.0f}s")
print("datasets:", [k for k in zarr.open_group(OUT, mode="r").group_keys()])

with ad.settings.override(remove_unused_categories=False):
    loader = Loader(batch_size=4096, chunk_size=64, preload_nchunks=64, shuffle=True, to="torch",
                    preload_to_gpu=False).use_collection(coll)
t0 = time.time()
n = 0
for i, batch in enumerate(loader):
    n += batch["X"].shape[0]
    if i == 0:
        print("first batch:", tuple(batch["X"].shape), list(batch["obs"].columns),
              "sources in batch:", batch["obs"]["source"].nunique())
print(f"one epoch over {n} cells in {time.time() - t0:.1f}s")
