#!/bin/bash
# CPU POC grid: lr sweep on small (4 threads, laptop-like), then plain vs balanced on small (4 threads)
# and on full (16 threads), 1 epoch each.
cd ${SBI2_WORK:?}
export PYTHONWARNINGS=default
OUT=poc
mkdir -p $OUT
PY=".venv/bin/python -u scripts/poc_annbatch_ooc.py"
for lr in 0.003 0.006 0.01; do
  $PY --tier small --collection workshop_small_collection.zarr --sampler shuffle --max-lr $lr --threads 4 --out $OUT
done
best=$(.venv/bin/python - <<PY
import pandas as pd, glob
df = pd.concat(pd.read_csv(p) for p in glob.glob("$OUT/small_shuffle_lr*_losses.csv"))
v = df[(df.metric == "val_balanced_acc") & (df.epoch == 1)].set_index("max_lr").value
print(v.idxmax())
PY
)
echo "best max_lr on small: $best"
$PY --tier small --collection workshop_small_collection_bycompound.zarr --sampler balanced --max-lr $best --threads 4 --out $OUT
$PY --tier full --collection workshop_full_collection.zarr --sampler shuffle --max-lr $best --threads 16 --out $OUT
$PY --tier full --collection workshop_full_collection_bycompound.zarr --sampler balanced --max-lr $best --threads 16 --out $OUT
.venv/bin/python scripts/poc_plot.py $OUT $best
