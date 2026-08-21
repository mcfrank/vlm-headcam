#!/bin/bash
# Experiment 1: aligned vs size-matched-random, on ccn2. Run after emb/ cache is built.
# Usage (on ccn2): bash run_exp1.sh <GPU>
set -e
GPU=${1:-2}
PY=/data2/mcfrank/ladder/condaenv/bin/python
cd /data2/mcfrank/vlm-headcam/src
COMMON="--epochs 20 --batch 256 --lr 3e-4 --min-freq 5"

for ARM in aligned random; do
  env CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 $PY -B train.py \
    --manifest /data2/mcfrank/vlm-headcam/manifests/smoke_${ARM}.parquet \
    --out /data2/mcfrank/vlm-headcam/runs/exp1_${ARM} $COMMON
done
echo ALL_DONE
