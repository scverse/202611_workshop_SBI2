"""Cross-check our off-the-shelf OpenPhenom embeddings against CP-BG-Bench's zero-shot single-cell embeddings.

The bench stored channelwise embeddings (5 x 384 = 1920). The non-channelwise embedding is the mean over all
tokens = the mean of the 5 per-channel means, so mean(reshape(5, 384)) must match ours (up to channel order,
which does not matter for the mean).
usage: python check_openphenom_vs_bench.py <crops224.zarr dir> <X_openphenom.npy>
"""
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import zarr

Z = os.environ["SBI2_SOURCE"] + "/results/baselines/openphenom_zeroshot/OpenPhenom_zeroshot_JUMP_singlecell.h5ad"
keys = ad.io.read_elem(zarr.open_group(sys.argv[1], mode="r")["obs"]).row_key.to_numpy()
ours = np.load(sys.argv[2])
z = ad.read_h5ad(Z, backed="r")
zk = z.obs["row_key"].to_numpy() if "row_key" in z.obs else z.obs_names.to_numpy()
pos = pd.Series(np.arange(len(zk)), index=zk).reindex(keys).to_numpy()
rng = np.random.default_rng(0)
take = np.sort(rng.choice(len(keys), min(2000, len(keys)), replace=False))
idx = pos[take].astype(int)
o = np.argsort(idx)
ref = np.empty((len(idx), z.shape[1]), np.float32)
ref[o] = np.asarray(z.X[np.sort(idx)])
ref = ref.reshape(len(idx), 5, 384).mean(1)
u, v = ours[take], ref
cos = (u * v).sum(1) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1)
print(f"cosine(ours, mean of bench channelwise) on {len(take)} cells: median {np.median(cos):.4f}, "
      f"min {cos.min():.4f}; max abs diff {np.abs(u - v).max():.4g}")
