#!/bin/bash
# usage: scripts/run.sh <cpus> <mem> <time> script.py [args]  -> sbatch --wait, prints log
cd ${SBI2_WORK:?set SBI2_WORK to the workdir}
c=$1; m=$2; t=$3; shift 3
name=$(basename $1 .py)
sbatch --wait -p cpu_p --qos=cpu_normal -c $c --mem=$m -t $t -x cpusrv77,cpusrv114,cpusrv122 -J sbi2_$name -o logs/${name}_%j.log --wrap "export PYTHONWARNINGS=default; .venv/bin/python -u $*" >/dev/null
cat $(ls -t logs/${name}_*.log | head -1)
