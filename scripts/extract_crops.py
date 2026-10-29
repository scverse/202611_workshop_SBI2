"""Extract the selected cells' crops from the jump-c WebDataset shards into zarr.

Reads only the needed shards; scans their tar headers for the wanted keys, then seeks to each
member (json row_key checked against selected.parquet).

Writes (zarr format 2, so both zarr-python 2 and 3 can read it):
  <work>/crops224.zarr  images (N,5,224,224) uint8, masks (N,2,224,224) uint8 {0,255}, obs   [for cp_measure]
  <work>/crops96.zarr   images (N,5,96,96) uint8 (cv2 INTER_AREA), masks (N,2,96,96) uint8 (INTER_NEAREST), obs
Rows follow data/selected.parquet; obs index = row_key.

sbatch: scripts/run.sh 4 32G 120 scripts/extract_crops.py
"""

import os
import argparse
import json
import tarfile
import time
from pathlib import Path

import anndata as ad
import cv2
import numpy as np
import pandas as pd
import zarr

SRC = Path(os.environ["SBI2_SOURCE"]) / Path("hf_release/jump-c/data")
W = Path(os.environ["SBI2_WORK"])
CELL, MASK, S = (5, 224, 224), (2, 224, 224), 96
CHANNELS, MASK_CHANNELS = ["DNA", "AGP", "ER", "Mito", "RNA"], ["nucleus", "cell"]
CHUNK = 256


def index_shard(path: Path, keys: set[str]) -> dict[str, tuple[int, int]]:
    """Map '<key>.<ext>' -> (data offset, size) for the wanted keys; reads tar headers only."""
    idx = {}
    with tarfile.open(path, mode="r:") as tf:
        for m in tf:
            if m.name.split(".", 1)[0] in keys:
                idx[m.name] = (m.offset_data, m.size)
    return idx


def read_sample(fh, idx: dict, key: str) -> tuple[np.ndarray, np.ndarray, dict]:
    out = {}
    for ext in ("cell.bin", "mask.bin", "json"):
        off, size = idx[f"{key}.{ext}"]
        fh.seek(off)
        out[ext] = fh.read(size)
    cell = np.frombuffer(out["cell.bin"], np.uint8).reshape(CELL)
    mask = np.frombuffer(out["mask.bin"], np.uint8).reshape(MASK)
    return cell, mask, json.loads(out["json"])


def resize(stack: np.ndarray, interp: int) -> np.ndarray:
    return np.stack([cv2.resize(c, (S, S), interpolation=interp) for c in stack])


def make_store(path: Path, n: int, size: int, obs: pd.DataFrame):
    g = zarr.open_group(path, mode="w", zarr_format=2)
    imgs = g.create_array("images", shape=(n, 5, size, size), chunks=(CHUNK, 5, size, size), dtype="uint8")
    msks = g.create_array("masks", shape=(n, 2, size, size), chunks=(CHUNK, 2, size, size), dtype="uint8")
    imgs.attrs["channels"], msks.attrs["channels"] = CHANNELS, MASK_CHANNELS
    ad.io.write_elem(g, "obs", obs)
    return g, imgs, msks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", default=str(W / "data/selected.parquet"))
    ap.add_argument("--work", default=str(W / "work"), help="writes crops224.zarr and crops96.zarr here")
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    sel = pd.read_parquet(args.selected)
    n = len(sel)
    obs = sel.set_index("row_key", drop=False)
    obs.index.name = None
    (W / "work").mkdir(exist_ok=True)
    g224, i224, m224 = make_store(work / "crops224.zarr", n, 224, obs)
    g96, i96, m96 = make_store(work / "crops96.zarr", n, S, obs)
    g96.attrs["description"] = ("SBI2 workshop crops: JUMP-CP TARGET2 cells (CP-BG-Bench jump-c view), 224 px crops "
                                "downsampled to 96 px; rows aligned with workshop_cells.h5ad obs.")

    t0 = time.time()
    buf = {k: [] for k in ("i224", "m224", "i96", "m96")}
    start = 0
    for shard, part in sel.groupby("shard_index", sort=True):
        path = SRC / f"shard-{shard:05d}.tar"
        idx = index_shard(path, set(part.key_in_shard))
        print(f"shard {shard}: indexed {len(idx)} members, {time.time() - t0:.0f}s", flush=True)
        with open(path, "rb") as fh:
            for row in part.itertuples():
                cell, mask, meta = read_sample(fh, idx, row.key_in_shard)
                if meta["row_key"] != row.row_key:
                    raise ValueError(f"row_key mismatch {meta['row_key']} != {row.row_key}")
                buf["i224"].append(cell)
                buf["m224"].append(mask)
                buf["i96"].append(resize(cell, cv2.INTER_AREA))
                buf["m96"].append(resize(mask, cv2.INTER_NEAREST))
                if len(buf["i224"]) == CHUNK:
                    start = flush(buf, start, i224, m224, i96, m96)
        print(f"shard {shard} done: {start + len(buf['i224'])}/{n} cells, {time.time() - t0:.0f}s", flush=True)
    flush(buf, start, i224, m224, i96, m96)
    # sel is sorted by shard/key, so groupby order == row order; double-check alignment
    assert list(ad.io.read_elem(g96["obs"]).row_key) == list(sel.row_key)
    print(f"done in {time.time() - t0:.0f}s")


def flush(buf, start, i224, m224, i96, m96):
    k = len(buf["i224"])
    if k == 0:
        return start
    sl = slice(start, start + k)
    i224[sl], m224[sl] = np.stack(buf["i224"]), np.stack(buf["m224"])
    i96[sl], m96[sl] = np.stack(buf["i96"]), np.stack(buf["m96"])
    for v in buf.values():
        v.clear()
    return start + k


if __name__ == "__main__":
    main()
