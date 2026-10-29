"""Checks on a tier's h5ad and (with --collection) the pre-shuffled annbatch DatasetCollection built from it.

(1) ad.experimental.read_elem_lazy(f["obsm"]["single_cell_images"], chunks=(16,-1,-1,-1)) works on the h5ad
(2) DatasetCollection.add_adatas(load_adata=<X + obsm images, lazy>, shuffle=True, n_obs_per_chunk=16) writes
    datasets holding both X and obsm/single_cell_images
(3) a Loader over the collection's X yields features; a Loader over obsm/single_cell_images yields image batches
Then zips the collection directory (stored, no recompression: the chunks are already compressed).

sbatch: scripts/run.sh 8 64G 180 scripts/make_collection.py --tier full --collection
"""

import os
import argparse
import shutil
import subprocess
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import zarr
from annbatch import DatasetCollection, Loader

W = Path(os.environ["SBI2_WORK"])
CHUNKS = (16, -1, -1, -1)
OBS_COLS = ["compound", "highlight", "Metadata_Source", "Metadata_Plate", "Metadata_Well", "is_control", "qc_fail"]

ap = argparse.ArgumentParser()
ap.add_argument("--tier", choices=["small", "full"], default="full")
ap.add_argument("--collection", action="store_true")
args = ap.parse_args()
H5 = W / f"workshop_{args.tier}.h5ad"
COLL = W / f"workshop_{args.tier}_collection.zarr"

# (1)
with h5py.File(H5, "r") as f:
    lazy = ad.experimental.read_elem_lazy(f["obsm"]["single_cell_images"], chunks=CHUNKS)
    print("(1) read_elem_lazy:", type(lazy).__name__, lazy.shape, lazy.dtype, "chunks", lazy.chunksize,
          "| h5 chunks", f["obsm"]["single_cell_images"].chunks, f["obsm"]["single_cell_images"].compression)
    first = np.asarray(lazy[:16])
    assert first.shape == (16, 5, 96, 96) and first.dtype == np.uint8 and first.any()
if not args.collection:
    raise SystemExit

_open = []


def load(path) -> ad.AnnData:
    f = h5py.File(path, "r")
    _open.append(f)  # keep the file open while the collection is written
    return ad.AnnData(
        X=ad.experimental.read_elem_lazy(f["X"], chunks=(16, -1)),
        obs=ad.io.read_elem(f["obs"])[OBS_COLS],
        var=ad.io.read_elem(f["var"])[["selected"]],
        obsm={"single_cell_images": ad.experimental.read_elem_lazy(f["obsm"]["single_cell_images"], chunks=CHUNKS)},
    )


if COLL.exists():
    shutil.rmtree(COLL)
t0 = time.time()
coll = DatasetCollection(zarr.open(COLL, mode="w"))
coll.add_adatas(adata_paths=[H5], load_adata=load, shuffle=True, n_obs_per_chunk=16, rng=np.random.default_rng(0))
print(f"collection written in {time.time() - t0:.0f}s")
# (2)
groups = list(coll)
for g in groups:
    assert "X" in g and "single_cell_images" in g["obsm"], f"{g.path} lacks X or obsm/single_cell_images"
print("(2) datasets:", [(g.path, g["X"].shape, g["obsm"]["single_cell_images"].shape,
                         g["obsm"]["single_cell_images"].chunks) for g in groups])

# (3)
obs = [ad.io.read_elem(g["obs"]) for g in groups]
for key, bs in (("X", 1024), ("obsm/single_cell_images", 64)):
    loader = Loader(batch_size=bs, chunk_size=16, preload_nchunks=64, shuffle=True, to="torch", preload_to_gpu=False)
    loader.add_datasets([g[key] for g in groups], obs=obs)
    t0, n = time.time(), 0
    for i, batch in enumerate(loader):
        n += batch["X"].shape[0]
        if i == 0:
            print(f"(3) {key}: first batch {tuple(batch['X'].shape)} {batch['X'].dtype}, "
                  f"{batch['obs']['compound'].nunique()} compounds in batch")
        if time.time() - t0 > 120:
            break
    print(f"    {n} rows in {time.time() - t0:.1f}s ({n / (time.time() - t0):.0f} rows/s)")

for f in _open:
    f.close()
z = COLL.with_suffix(".zarr.zip")
z.unlink(missing_ok=True)
subprocess.run(["zip", "-q", "-r", "-0", z.name, COLL.name], cwd=W, check=True)
print(f"zipped: {z} {z.stat().st_size / 1e9:.2f} GB")
