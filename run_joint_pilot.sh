#!/bin/bash
# Joint-training PILOT: last-4-blocks DINOv3-B fine-tune at 300k pairs, 3 seeds, vs the
# matched frozen runs (B26_rand_300000_s0..2, mean 62.1). Waits for bundle 2 to finish and
# for the 224px frame cache. metrics.json family: B26J_300000. ~5h/run, one GPU each.
set -u
cd "$(dirname "$0")"
export HF_HOME=/ccn2/u/khaiaw/.cache/huggingface HF_HUB_OFFLINE=1
PY=/ccn2/u/khaiaw/miniconda3/envs/ccwm/bin/python

until grep -q "B26_DONE" logs/b26_driver.log 2>/dev/null; do
  echo "$(date +%H:%M) waiting for bundle 2"; sleep 600; done
until grep -q "CACHE224_DONE" logs/cache224.log 2>/dev/null; do
  echo "$(date +%H:%M) waiting for the 224px cache"; sleep 600; done

for s in 0 1 2; do
  [ -f runs/B26J_300000_s$s/metrics.json ] && continue
  CUDA_VISIBLE_DEVICES=$s $PY -B src/train_joint.py \
    --manifest manifests/bv26_rand_300000_s$s.parquet \
    --seed $s --out runs/B26J_300000_s$s > logs/b26j_300000_s$s.log 2>&1 &
done
wait
echo "B26J_PILOT_DONE $(date)"
for s in 0 1 2; do grep -h "DONE\|probe" logs/b26j_300000_s$s.log | tail -2; done
