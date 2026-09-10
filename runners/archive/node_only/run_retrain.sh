#!/bin/bash
cd /data2/mcfrank/vlm-headcam
export CUDA_VISIBLE_DEVICES=7
PY=/data2/mcfrank/ladder/condaenv/bin/python
run () {  # arm N seed
  $PY -B src/train.py --manifest manifests/${1}_top${2}.parquet --emb-dir emb_full \
      --out runs/${3} --epochs 20 --seed ${4} 2>&1 | tail -1
  echo "  ^ ${3} done"
}
echo "=== N=22000, 3 seeds ==="
for s in 0 1 2; do
  run gemini 22000 GEM_top22k_s${s} $s
  run clip   22000 CLIP_top22k_s${s} $s
done
echo "=== N=15000, seed 0 ==="
run gemini 15000 GEM_top15k_s0 0
run clip   15000 CLIP_top15k_s0 0
echo "ALL_RETRAIN_DONE"
