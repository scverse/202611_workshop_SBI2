"""Sanity check: mantispy tl.map activity (vs DMSO wells, per source) for each representation of the workshop cells.

Per representation: cells without qc_fail -> embeddings mantispy mad_robustize vs DMSO per source (cp_measure X is
used as stored, whole-plate normalized), clipped to |z| <= 50, NaN -> 0 -> well means -> tl.map(mode="activity") run separately per source.
Reports the fraction of compounds active (q < 0.05), per source and per activity class.

sbatch: scripts/run.sh 4 32G 60 scripts/sanity_activity.py
"""

import os
import argparse
from pathlib import Path

import h5py

import anndata as ad
import mantispy as mp
import numpy as np
import pandas as pd

W = Path(os.environ["SBI2_WORK"])
KEYS = ["Metadata_Source", "Metadata_Plate", "Metadata_Well", "Metadata_Perturbation"]

ap = argparse.ArgumentParser()
ap.add_argument("--tier", choices=["small", "full"], default="small")
args = ap.parse_args()
with h5py.File(W / f"workshop_{args.tier}.h5ad", "r") as f:  # skip the crops in obsm
    emb = [k for k in ("X_openphenom", "X_openphenom_ft", "X_ae") if k in f["obsm"]]
    a = ad.AnnData(**{k: ad.io.read_elem(f[k]) for k in ("X", "obs", "var")},
                   obsm={k: ad.io.read_elem(f["obsm"][k]) for k in emb})
a = a[~a.obs.qc_fail.to_numpy()].copy()
obs = a.obs[KEYS + ["activity_class", "Metadata_Control_Type"]].astype(str)

sel = a.var.selected.to_numpy()
reps = {"cp_measure": np.clip(np.nan_to_num(a.X[:, sel]), -50, 50)}
for k in ("X_openphenom", "X_openphenom_ft", "X_ae"):
    if k in a.obsm:
        e = ad.AnnData(X=np.asarray(a.obsm[k], dtype=np.float32), obs=a.obs[["Metadata_Source", "Metadata_Control_Type"]].copy())
        mp.pp.normalize(e, method="mad_robustize", by="Metadata_Source", reference="negcon")
        reps[k.removeprefix("X_")] = np.clip(np.nan_to_num(e.X), -50, 50)

rows = []
for name, x in reps.items():
    df = pd.DataFrame(x, index=obs.index)
    wells = df.groupby([obs[c] for c in KEYS], observed=True).mean()
    wobs = wells.index.to_frame(index=False)
    w = ad.AnnData(X=wells.to_numpy(np.float32), obs=wobs.set_index(pd.Index(map(str, range(len(wobs))))))
    w.obs["is_dmso"] = (w.obs.Metadata_Perturbation == "DMSO").to_numpy()
    cls = obs.drop_duplicates("Metadata_Perturbation").set_index("Metadata_Perturbation").activity_class
    for src in sorted(w.obs.Metadata_Source.unique()):
        s = w[w.obs.Metadata_Source == src].copy()
        if s.obs.is_dmso.sum() < 2:
            print(f"{name} {src}: <2 DMSO wells, skipped")
            continue
        mp.tl.map(s, mode="activity", label="Metadata_Perturbation", batch="Metadata_Source", reference="is_dmso",
                  null_size=20000, key_added="map", n_jobs=4)
        t = s.uns["mantispy"]["map"][["Metadata_Perturbation", "score", "qvalue"]].copy()
        t["source"], t["rep"] = src, name
        t["activity_class"] = t.Metadata_Perturbation.map(cls).to_numpy()
        t["n_dmso_wells"] = int(s.obs.is_dmso.sum())
        rows.append(t)
res = pd.concat(rows, ignore_index=True)
res["active"] = res.qvalue < 0.05
res.to_parquet(W / f"work/sanity_activity_{args.tier}.parquet")

pd.set_option("display.width", 200)
print("DMSO wells per source:", res.drop_duplicates("source").set_index("source").n_dmso_wells.to_dict())
print("\nfraction of (compound, source) pairs active, q < 0.05")
tab = res.pivot_table(index="rep", columns="activity_class", values="active", aggfunc="mean")
tab["all"] = res.groupby("rep").active.mean()
print(tab.round(2).loc[list(reps)])
print("\nmean mAP")
print(res.pivot_table(index="rep", columns="activity_class", values="score", aggfunc="mean").round(2).loc[list(reps)])
print("\nfraction active per source")
print(res.pivot_table(index="rep", columns="source", values="active", aggfunc="mean").round(2).loc[list(reps)])
print("\ncompounds active in >= 2 of 4 sources")
print((res.groupby(["rep", "Metadata_Perturbation"]).active.sum() >= 2).groupby("rep").mean().round(2).loc[list(reps)])
