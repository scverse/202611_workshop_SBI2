"""Off-the-shelf OpenPhenom (HF recursionpharma/OpenPhenom) embeddings for the workshop cells (GPU job).

Input as OpenPhenom expects (model card + its MAEModel.predict): uint8 (N, C, 256, 256); the model
itself scales /255 and instance-normalises each image channel. Our 224 px crops are resized to 256 px
(bilinear, antialias; the same transform CP-BG-Bench used for run 05). Channels are passed in Recursion
order [DNA, ER, AGP, RNA, Mito]; the channel-agnostic ViT shares patch and positional embeddings across
channels and mean-pools all tokens, so the 384-d embedding does not depend on channel order (checked below).

Writes <work>/X_openphenom.npy (N, 384) float32, rows aligned with the crops' obs (work/full: data/full_selected.parquet).
Cross-checks against the CP-BG-Bench zero-shot single-cell embeddings when present.

sbatch -p gpu_p --gres=gpu:1 ... .venv-gpu/bin/python scripts/embed_openphenom.py
"""

import os
import argparse
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2.functional as TF
import zarr

W = Path(os.environ["SBI2_WORK"])
SNAP = next((W / ".hf/hub/models--recursionpharma--OpenPhenom/snapshots").iterdir())
ZEROSHOT = Path(os.environ["SBI2_SOURCE"]) / Path("results/baselines/openphenom_zeroshot/"
                "OpenPhenom_zeroshot_JUMP_singlecell.h5ad")
SRC_ORDER = ["DNA", "AGP", "ER", "Mito", "RNA"]
RXRX_ORDER = ["DNA", "ER", "AGP", "RNA", "Mito"]
PERM = [SRC_ORDER.index(c) for c in RXRX_ORDER]
BS = 128

ap = argparse.ArgumentParser()
ap.add_argument("--work", default=str(W / "work"), help="reads <work>/crops224.zarr, writes <work>/X_openphenom.npy")
args = ap.parse_args()
WORK = Path(args.work)

sys.path.insert(0, str(SNAP.parent))
OpenPhenom = __import__(SNAP.name + ".huggingface_mae", fromlist=["MAEModel"])
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
# MAEModel.from_pretrained calls torch.load on "model.safetensors", which is really a torch zip pickle;
# torch 2.15 routes the .safetensors suffix to the safetensors reader and fails. Load from a file handle instead.
model = OpenPhenom.MAEModel(OpenPhenom.MAEConfig.from_pretrained(str(SNAP)))
with open(SNAP / "model.safetensors", "rb") as fh:
    model.load_state_dict(torch.load(fh, map_location="cpu", weights_only=True)["state_dict"])
model = model.eval().to(dev)
model.return_channelwise_embeddings = False

g = zarr.open_group(WORK / "crops224.zarr", mode="r")
obs = ad.io.read_elem(g["obs"])
n = len(obs)


@torch.inference_mode()
def embed(x: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(x).to(dev)
    t = TF.resize(t, [256, 256], antialias=True)  # uint8 in, uint8 out (bilinear)
    return model.predict(t).float().cpu().numpy()


t0 = time.time()
out = np.zeros((n, 384), np.float32)
for s in range(0, n, BS):
    out[s:s + BS] = embed(g["images"][s:s + BS][:, PERM])
print(f"embedded {n} cells in {time.time() - t0:.0f}s; finite: {np.isfinite(out).all()}")
np.save(WORK / "X_openphenom.npy", out)

# channel-order invariance check on 64 cells
x = g["images"][:64]
a, b = embed(x[:, PERM]), embed(x)
print("max |emb(rxrx order) - emb(source order)|:", float(np.abs(a - b).max()))


def cos_rows(u, v):
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return (u * v).sum(1)


if ZEROSHOT.exists():
    z = ad.read_h5ad(ZEROSHOT, backed="r")
    print("zero-shot h5ad:", z.shape, list(z.obsm.keys()), list(z.uns.keys())[:10])
    keys = z.obs["row_key"].to_numpy() if "row_key" in z.obs else z.obs_names.to_numpy()
    pos = pd.Series(np.arange(len(keys)), index=keys).reindex(obs.row_key.to_numpy())
    ok = pos.notna().to_numpy()
    print(f"zero-shot coverage {ok.mean():.3f}")
    idx = pos[ok].astype(int).to_numpy()
    order = np.argsort(idx)
    ref = np.empty((len(idx), z.shape[1]), np.float32)
    ref[order] = np.asarray(z.X[np.sort(idx)])
    if ref.shape[1] == 384:
        c = cos_rows(out[ok], ref)
        print(f"cosine(ours, bench zero-shot) per cell: median {np.median(c):.4f}, 5th pct {np.percentile(c, 5):.4f}")
    else:
        print("zero-shot dim", ref.shape[1], "- not directly comparable")
