"""Build a tier's crops collection grouped by compound, for class-balanced sampling with annbatch.

annbatch's WeightedClassSampler reads whole chunks of one class, so each class must sit in contiguous runs of at
least chunk_size rows. add_adatas(groupby="compound") shuffles and then sorts each dataset by compound, which gives
exactly that; the sampler then mixes classes by shuffling the rows of a preload window.

    python poc_build_grouped.py --tier small     -> workshop_small_collection_bycompound.zarr (prints build time)
"""

import argparse
import os
import shutil
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import zarr
from annbatch import DatasetCollection

W = Path(os.environ["SBI2_WORK"])
ap = argparse.ArgumentParser()
ap.add_argument("--tier", required=True)
args = ap.parse_args()
H5, OUT = W / f"workshop_{args.tier}.h5ad", W / f"workshop_{args.tier}_collection_bycompound.zarr"
_open = []


def load(path) -> ad.AnnData:
    f = h5py.File(path, "r")
    _open.append(f)
    obs = ad.io.read_elem(f["obs"])[["compound", "highlight", "Metadata_Source", "Metadata_Plate", "Metadata_Well"]]
    return ad.AnnData(obs=obs, obsm={"single_cell_images": ad.experimental.read_elem_lazy(
        f["obsm"]["single_cell_images"], chunks=(16, -1, -1, -1))})


if OUT.exists():
    shutil.rmtree(OUT)
t0 = time.time()
DatasetCollection(zarr.open(OUT, mode="w")).add_adatas(
    adata_paths=[H5], load_adata=load, groupby="compound", shuffle=True, n_obs_per_chunk=16,
    rng=np.random.default_rng(0))
print(f"grouped collection {OUT.name} built in {time.time() - t0:.0f} s")
for f in _open:
    f.close()
