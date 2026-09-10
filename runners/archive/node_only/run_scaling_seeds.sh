#!/bin/bash
cd /data2/mcfrank/vlm-headcam
export PYTHONPATH=src
PY=/data2/mcfrank/ladder/condaenv/bin/python
MANS="scale_rand_10000 scale_rand_30000 scale_rand_100000 scale_rand_300000 scale_align_10000 scale_align_30000 scale_align_85000 scale_div_1c scale_div_3c scale_div_10c scale_div_36c scale_bigchild scale_poolbig"
i=0
for s in 1 2; do for man in $MANS; do
  g=$((i%8))
  CUDA_VISIBLE_DEVICES=$g $PY -B src/train_region_mil.py --manifest manifests/$man.parquet --region-cache emb_reg --mode plain --epochs 20 --seed $s --out runs/G_sc_${man}_s$s > logs/sc_${man}_s$s.log 2>&1 &
  i=$((i+1)); [ $((i%8)) -eq 0 ] && wait
done; done
wait
echo "=== SCALING seeds eval ==="
for s in 1 2; do for man in $MANS; do
  echo -n "${man}_s$s: "; $PY -B src/eval_model.py G_sc_${man}_s$s manifests/eval_frames_konkle.parquet emb_konkle 2>/dev/null | grep -oE "4AFC=[0-9.]+"
done; done
echo "SCALING_SEEDS_DONE"
