#!/bin/bash
# GPU job: conv AE on crops streamed by annbatch; both tiers, plain shuffle vs class-balanced sampling.
cd ${SBI2_WORK:?}
export PYTHONWARNINGS=default
mkdir -p poc
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader -l 5 > poc/gpu_util.csv &
SMI=$!
run() {  # tier epochs sampler collection
  s=$(date +%s)
  .venv-gpu/bin/python -u scripts/poc_annbatch_ooc.py --tier $1 --epochs $2 --sampler $3 --collection $4 --out poc
  echo "$1 $3 wall $(( $(date +%s) - s )) s"
}
run small 5 shuffle workshop_small_collection.zarr
run small 5 balanced workshop_small_collection_bycompound.zarr
run full 3 shuffle workshop_full_collection.zarr
run full 3 balanced workshop_full_collection_bycompound.zarr
kill $SMI
.venv-gpu/bin/python scripts/poc_plot.py poc
