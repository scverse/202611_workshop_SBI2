"""Pick the HDF5 compression for obsm["single_cell_images"]: size and random chunk-read speed, gzip-4 vs lzf vs none.

Writes small-tier crops as (n,5,96,96) uint8 chunked (16,5,96,96), then reads 300 random 16-cell chunks through
ad.experimental.read_elem_lazy (the path annbatch uses) and through plain h5py.
"""

import os
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import zarr

W = Path(os.environ["SBI2_WORK"])
arr = zarr.open_group(W / "work/crops96.zarr", mode="r")["images"][:]
n = len(arr)
rng = np.random.default_rng(0)
starts = rng.choice(n // 16, 300, replace=False) * 16
tmp = Path(os.environ.get("TMPDIR", "/tmp"))  # node-local disk, so network-filesystem caching does not skew the reads
for name, kw in (("gzip4", {"compression": "gzip", "compression_opts": 4}), ("lzf", {"compression": "lzf"}),
                 ("none", {})):
    p = tmp / f"bench_{name}.h5"
    t0 = time.time()
    with h5py.File(p, "w") as f:
        ad.io.write_elem(f, "imgs", arr, dataset_kwargs={"chunks": (16, 5, 96, 96), **kw})
    tw = time.time() - t0
    with h5py.File(p, "r") as f:
        t0 = time.time()
        for s in starts:
            f["imgs"][s:s + 16]
        th = (time.time() - t0) / len(starts) * 1e3
        lazy = ad.experimental.read_elem_lazy(f["imgs"], chunks=(16, -1, -1, -1))
        t0 = time.time()
        for s in starts[:100]:
            np.asarray(lazy[s:s + 16])
        tl = (time.time() - t0) / 100 * 1e3
    print(f"{name:6s} size {p.stat().st_size / 1e6:7.1f} MB  write {tw:5.1f}s  "
          f"h5py chunk read {th:5.2f} ms  read_elem_lazy chunk read {tl:5.2f} ms")
    p.unlink()
