"""Select the workshop cell subset (TARGET2 plates, 4 sources, DMSO + active/moderate/inactive compounds).

Activity = mantispy tl.map(mode="activity") mAP of each compound's wells vs DMSO wells,
computed on the run-05 fine-tuned aggregated (well-level) profiles, separately per source
(batch="source": TARGET2 plates hold too few DMSO wells for a per-plate reference).

Writes:
  data/selected.parquet          one row per selected cell (metadata + compound annotation)
  data/compound_activity.parquet per (source, compound) mAP / q-value table for all TARGET2 compounds
"""

import os
import argparse
from pathlib import Path

import anndata as ad
import mantispy as mp
import numpy as np
import pandas as pd

P = Path(os.environ["SBI2_SOURCE"])
W = Path(os.environ["SBI2_WORK"])
DMSO = "IAZDPXIOMUYVGZ-UHFFFAOYSA-N"

ap = argparse.ArgumentParser()
ap.add_argument("--sources", default="source_4,source_13,source_5,source_11")
ap.add_argument("--n-active", type=int, default=25)
ap.add_argument("--n-inactive", type=int, default=5)
ap.add_argument("--n-moderate", type=int, default=10)
ap.add_argument("--null-size", type=int, default=20000)
args = ap.parse_args()
sources = args.sources.split(",")

meta = pd.read_parquet(P / "hf_release/jump-c/metadata.parquet")
t2 = meta[meta.Metadata_PlateType == "TARGET2"]

agg = ad.read_h5ad(P / "results/embeddings/05_JUMP_OpenPhenom_ECFP4_C_aggregated.h5ad")
agg = agg[agg.obs.Metadata_PlateType == "TARGET2"].copy()
agg.obs["is_control"] = agg.obs.Metadata_InChIKey == DMSO
# mantispy passes only Metadata_* obs columns to copairs, so the batch key needs that prefix
agg.obs["Metadata_Source"] = agg.obs.source.astype(str)
print(agg.obs.groupby("source", observed=True).agg(wells=("well", "size"), dmso_wells=("is_control", "sum")))

act_path = W / "data/compound_activity.parquet"
if act_path.exists():  # cached: the permutation null takes a while
    act = pd.read_parquet(act_path)
else:
    rows = []
    for src in sorted(agg.obs.source.unique()):
        a = agg[agg.obs.source == src].copy()
        if a.obs.is_control.sum() < 2:
            print(f"skip {src}: <2 DMSO wells")
            continue
        mp.tl.map(a, mode="activity", label="Metadata_InChIKey", batch="Metadata_Source", reference="is_control",
                  null_size=args.null_size, key_added="map")
        tab = a.uns["mantispy"]["map"].copy()
        tab["source"] = src
        tab["n_dmso_wells"] = int(a.obs.is_control.sum())
        rows.append(tab)
    act = pd.concat(rows, ignore_index=True)
    act.to_parquet(act_path)
key = "Metadata_InChIKey"
print(act.groupby("source").agg(n=("score", "size"), sig=("is_significant", "sum"), med=("score", "median"),
                                qmin=("qvalue", "min")))

# compound annotation (JUMP-Target-2, joined by InChIKey)
ann = pd.read_csv(W / "ref/JUMP-Target-2_compound_metadata.tsv", sep="\t")
ann = ann.drop_duplicates("InChIKey").rename(columns={"InChIKey": key, "pert_iname": "compound", "target_list": "target"})

sub = act[act.source.isin(sources)]
mapcol, qcol = "score", "qvalue"
per = sub.groupby(key).agg(map_mean=(mapcol, "mean"), map_min=(mapcol, "min"), q_max=(qcol, "max"),
                           q_min=(qcol, "min"), n_sig=(qcol, lambda q: int((q < 0.05).sum())),
                           n_src=("source", "nunique")).reset_index()
per = per[(per[key] != DMSO) & (per.n_src == len(sources))].merge(ann[[key, "compound", "target"]], on=key, how="left")
# active: significant (q < 0.05) in all but at most one source, ranked by mean mAP; inactive: never significant,
# lowest mean mAP
active = per[per.n_sig >= len(sources) - 1].sort_values("map_mean", ascending=False).head(args.n_active)
inactive = per[per.n_sig == 0].sort_values("map_mean").head(args.n_inactive)
print("compounds by number of sources with q<0.05:", per.n_sig.value_counts().sort_index().to_dict())
# moderate: significant in 1-2 sources with mean mAP 0.6-0.8, spread evenly over that range; this is where
# representations should differ (strong compounds score ~1.0 in every representation)
mod_pool = per[per.n_sig.between(1, 2) & per.map_mean.between(0.6, 0.8)].sort_values("map_mean").reset_index(drop=True)
print(f"moderate pool: {len(mod_pool)} compounds")
moderate = mod_pool.iloc[np.unique(np.linspace(0, len(mod_pool) - 1, args.n_moderate).round().astype(int))]
chosen = pd.concat([active.assign(activity_class="active"), moderate.assign(activity_class="moderate"),
                    inactive.assign(activity_class="inactive")])

cells = t2[t2.source.isin(sources)]
sel = cells[cells[key].isin(chosen[key]) | (cells[key] == DMSO)].copy()
sel = sel.merge(chosen[[key, "compound", "target", "map_mean", "activity_class"]], on=key, how="left")
sel.loc[sel[key] == DMSO, ["compound", "activity_class"]] = ["DMSO", "control"]
sel["is_control"] = sel[key] == DMSO
sel = sel.sort_values(["shard_index", "key_in_shard"]).reset_index(drop=True)
sel.to_parquet(W / "data/selected.parquet")

pd.set_option("display.width", 250, "display.max_rows", 100, "display.max_colwidth", 40)
print(sel.groupby("source").agg(cells=("row_key", "size"), dmso=("is_control", "sum"), plates=("plate", "nunique")))
print(sel.groupby(["activity_class", "compound"]).agg(cells=("row_key", "size"), map=("map_mean", "first"),
                                                       target=("target", "first")).sort_values(["activity_class", "map"]))
print("total cells", len(sel), "shards", sel.shard_index.nunique())
