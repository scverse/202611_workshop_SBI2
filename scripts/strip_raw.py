"""Drop layers["raw"] from a tier's h5ad (HDF5 does not shrink on delete, so copy everything else to a new file).

    python strip_raw.py --tier full
"""

import os
import argparse
from pathlib import Path

import h5py

W = Path(os.environ["SBI2_WORK"])
ap = argparse.ArgumentParser()
ap.add_argument("--tier", default="full")
args = ap.parse_args()
src = W / f"workshop_{args.tier}.h5ad"
tmp = src.with_suffix(".noraw.h5ad")
before = src.stat().st_size
with h5py.File(src, "r") as fi, h5py.File(tmp, "w") as fo:
    for k, v in fi.attrs.items():
        fo.attrs[k] = v
    for k in fi:
        if k != "layers":
            fi.copy(fi[k], fo, name=k)
    lay = fo.create_group("layers")
    for k, v in fi["layers"].attrs.items():
        lay.attrs[k] = v
    for k in fi["layers"]:
        if k != "raw":
            fi.copy(fi["layers"][k], lay, name=k)
tmp.replace(src)
print(f"{src.name}: {before / 1e9:.2f} GB -> {src.stat().st_size / 1e9:.2f} GB without layers['raw']")
