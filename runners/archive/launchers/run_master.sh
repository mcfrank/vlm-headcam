#!/bin/bash
# Overnight chainer: wait for Batch 1 -> self-smoke the new E-step modes -> run Batch 2.
# Self-validating so it won't waste the night on a broken mode. Markers let the client
# resume/analyze independently.
set -u
W=/data2/mcfrank/vlm-headcam
PY=/data2/mcfrank/ladder/condaenv/bin/python
EV=$W/manifests/eval_frames.parquet

until [ -f $W/runs/BATCH1_DONE ]; do sleep 30; done
echo "[master] batch1 done; smoke-testing distinct+lang E-step"
rm -rf $W/runs/_smoke_b2
env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 $PY -B $W/src/train_region_mil.py \
  --manifest $W/manifests/boot_across_20000.parquet --region-cache $W/emb_reg --eval-frames $EV \
  --out $W/runs/_smoke_b2 --mode boot --warmup 1 --rounds 1 --epochs-per-round 1 \
  --score-mode distinct --lang-prior > $W/smoke_b2.log 2>&1
if [ $? -eq 0 ]; then
  echo "[master] smoke ok; launching batch 2"
  bash $W/run_batch2.sh
else
  echo "[master] BATCH2 SMOKE FAILED — see smoke_b2.log" ; touch $W/runs/BATCH2_SMOKE_FAILED
fi
touch $W/runs/MASTER_DONE
