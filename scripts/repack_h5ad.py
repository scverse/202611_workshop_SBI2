"""Repack the tier h5ads (HDF5 keeps the space of rewritten groups; copying every element reclaims it).
Datasets keep their chunking and compression. Verifies obs/X/obsm shapes after the copy."""
import os
from pathlib import Path

import anndata as ad
import h5py

W = Path(os.environ["SBI2_WORK"])
for tier in ("small", "full"):
    src = W / f"workshop_{tier}.h5ad"
    tmp = src.with_suffix(".repack.h5ad")
    before = src.stat().st_size
    with h5py.File(src, "r") as fi, h5py.File(tmp, "w") as fo:
        for k, v in fi.attrs.items():
            fo.attrs[k] = v
        for k in fi:
            fi.copy(fi[k], fo, name=k)
    with h5py.File(src, "r") as a, h5py.File(tmp, "r") as b:
        for k in ("X", "obsm/single_cell_images", "obsm/masks", "obsm/X_openphenom"):
            assert a[k].shape == b[k].shape and a[k].chunks == b[k].chunks and a[k].compression == b[k].compression
        assert ad.io.read_elem(a["obs"]).equals(ad.io.read_elem(b["obs"]))
    tmp.replace(src)
    print(f"{src.name}: {before} -> {src.stat().st_size} bytes")
