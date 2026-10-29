"""Re-check the curated 40 compounds' activity (run-05 fine-tuned well profiles, tl.map activity vs DMSO per source)
on the full-tier sources. Same rules as select_subset.py: active = q<0.05 in all but at most one scored source,
inactive = never significant, else moderate. source_10 has 1 DMSO well and is not scored.
Writes data/curated_recheck.parquet."""
import os
from pathlib import Path

import pandas as pd

W = Path(os.environ["SBI2_WORK"])
NEW = ["source_5", "source_6", "source_10", "source_7", "source_13"]
key = "Metadata_InChIKey"
act = pd.read_parquet(W / "data/compound_activity.parquet")
cur = pd.read_parquet(W / "data/selected.parquet").drop_duplicates(key)
cur = cur[~cur.is_control][[key, "compound", "activity_class"]]
a = act[act.source.isin(NEW)]
print("scored new sources:", sorted(a.source.unique()))
per = a.groupby(key).agg(map_new=("score", "mean"), n_sig=("qvalue", lambda q: int((q < 0.05).sum())),
                         n_src=("source", "nunique")).reset_index()
r = cur.merge(per, on=key, how="left")
r["class_new"] = "moderate"
r.loc[r.n_sig >= r.n_src - 1, "class_new"] = "active"
r.loc[r.n_sig == 0, "class_new"] = "inactive"
r.to_parquet(W / "data/curated_recheck.parquet")
print(pd.crosstab(r.activity_class, r.class_new, margins=True))
pd.set_option("display.width", 200, "display.max_rows", 60)
print(r.sort_values(["activity_class", "map_new"]).to_string(index=False, columns=["compound", "activity_class",
      "class_new", "map_new", "n_sig", "n_src"], float_format="%.2f"))
