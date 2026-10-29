"""Full tier: every TARGET2 cell (all compounds + DMSO) from the 5 most similar sources.

Sources: the Yokogawa spinning-disk confocals with NA 0.75 objectives in JUMP's microscope_config.csv
(CV8000: source_5, source_6, source_10; CV7000: source_7, source_13), ~0.65 um pixels, laser excitation.
Columns match data/selected.parquet (small tier) so build_tier.py treats both alike; activity columns come
import os
from data/compound_activity.parquet (run-05 activity mAP vs DMSO, per source) where available.

Writes data/full_selected.parquet.
"""

from pathlib import Path

import numpy as np
import pandas as pd

P = Path(os.environ["SBI2_SOURCE"])
W = Path(os.environ["SBI2_WORK"])
DMSO = "IAZDPXIOMUYVGZ-UHFFFAOYSA-N"
SOURCES = ["source_5", "source_6", "source_10", "source_7", "source_13"]
key = "Metadata_InChIKey"

meta = pd.read_parquet(P / "hf_release/jump-c/metadata.parquet")
sel = meta[(meta.Metadata_PlateType == "TARGET2") & meta.source.isin(SOURCES)].copy()

ann = pd.read_csv(W / "ref/JUMP-Target-2_compound_metadata.tsv", sep="\t").drop_duplicates("InChIKey")
ann = ann.rename(columns={"InChIKey": key, "pert_iname": "compound", "target_list": "target"})
sel = sel.merge(ann[[key, "compound", "target"]], on=key, how="left")
missing = sel.compound.isna() & (sel[key] != DMSO)
print(f"cells without a JUMP-Target-2 name: {missing.sum()} ({sel.loc[missing, key].nunique()} InChIKeys)")
sel["compound"] = sel.compound.fillna(sel[key])
sel.loc[sel[key] == DMSO, "compound"] = "DMSO"
sel["is_control"] = sel[key] == DMSO

act = pd.read_parquet(W / "data/compound_activity.parquet")
act = act[act.source.isin(SOURCES)].groupby(key).score.mean()
sel["map_mean"] = sel[key].map(act)
small = pd.read_parquet(W / "data/selected.parquet").drop_duplicates(key).set_index(key).activity_class
sel["activity_class"] = sel[key].map(small).fillna("other")
sel.loc[sel.is_control, "activity_class"] = "control"

sel = sel.sort_values(["shard_index", "key_in_shard"]).reset_index(drop=True)
sel.to_parquet(W / "data/full_selected.parquet")
print(sel.groupby("source").agg(cells=("row_key", "size"), dmso=("is_control", "sum"), plates=("plate", "nunique"),
                                compounds=(key, "nunique")))
print("total", len(sel), "cells,", sel[key].nunique(), "perturbations,", sel.shard_index.nunique(), "shards")
print(sel.activity_class.value_counts().to_dict())
