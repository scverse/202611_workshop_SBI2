#!/bin/bash
# Submit the post-cp_measure chain for both tiers. usage: scripts/run_tiers.sh <cp_measure array job id>
set -euo pipefail
cd ${SBI2_WORK:?set SBI2_WORK to the workdir}
X="-p cpu_p --qos=cpu_normal -x cpusrv77,cpusrv114,cpusrv122"
E="export PYTHONWARNINGS=default;"
PY=".venv/bin/python -u"
sub() { sbatch --parsable $X "$@"; }
j1=$(sub -d afterok:$1 -c 8 --mem=128G -t 180 -J sbi2_bfull -o logs/build_full_%j.log --wrap "$E $PY scripts/build_tier.py --tier full")
j2=$(sub -d afterok:$j1 -c 4 --mem=96G -t 90 -J sbi2_dsmall -o logs/derive_small_%j.log --wrap "$E $PY scripts/derive_small.py")
j3=$(sub -d afterok:$j2 -c 2 --mem=8G -t 90 -J sbi2_strip -o logs/strip_raw_%j.log --wrap "$E if [ \$(stat -c %s workshop_full.h5ad) -gt 3000000000 ]; then $PY scripts/strip_raw.py --tier full; else echo keep raw; fi")
j4=$(sub -d afterok:$j2 -c 8 --mem=16G -t 180 -J sbi2_ae -o logs/ae_small_%j.log --wrap "$E OMP_NUM_THREADS=8 $PY scripts/train_autoencoder.py --adata workshop_small.h5ad --epochs 5000 --patience 60 --out autoencoder.pt --latent-out work/X_ae_small.npy")
j5=$(sub -d afterok:$j4:$j3 -c 8 --mem=32G -t 60 -J sbi2_aefull -o logs/ae_full_latent_%j.log --wrap "$E OMP_NUM_THREADS=8 $PY scripts/train_autoencoder.py --adata workshop_full.h5ad --init autoencoder.pt --epochs 0 --latent-out work/full/X_ae.npy && $PY scripts/add_obsm.py --tier small --key X_ae --npy work/X_ae_small.npy && $PY scripts/add_obsm.py --tier full --key X_ae --npy work/full/X_ae.npy")
j6=$(sub -d afterok:$j5 -c 4 --mem=32G -t 120 -J sbi2_sanity -o logs/sanity_small_%j.log --wrap "$E $PY scripts/sanity_activity.py --tier small")
j7=$(sub -d afterok:$j5 -c 8 --mem=96G -t 240 -J sbi2_sanity -o logs/sanity_full_%j.log --wrap "$E $PY scripts/sanity_activity.py --tier full")
j8=$(sub -d afterok:$j5 -c 8 --mem=64G -t 240 -J sbi2_coll -o logs/collection_full_%j.log --wrap "$E $PY scripts/make_collection.py --tier small && $PY scripts/make_collection.py --tier full --collection")
echo "build_full=$j1 derive_small=$j2 strip=$j3 ae=$j4 ae_full=$j5 sanity_small=$j6 sanity_full=$j7 collection=$j8"
