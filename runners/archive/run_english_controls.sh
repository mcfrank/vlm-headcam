#!/bin/bash
# Why does the English-filtered arm sit BELOW the scaling curve? Matched-count controls.
#   english   790,519 pairs / 28 children (>=80% English)      -> 67.5 (have)
#   ctl_qty   790,519 pairs / 36 children, random              -> isolates pair count
#   ctl_kids  790,519 pairs / 28 children, 8 RANDOM dropped    -> isolates count + diversity
set -u
cd /data2/mcfrank/vlm-headcam
export PYTHONPATH=src
PY=/data2/mcfrank/ladder/condaenv/bin/python
TRAIN=emb_dv3_grid_877k
EVAL="--eval-cache emb_enc_grid_eval/dinov3b_ots_konkle --eval-frames manifests/eval_frames_konkle.parquet"
DEV="--dev-cache emb_dv3_konkle_dev16 --dev-frames manifests/eval_frames_konkle_dev.parquet"
COV="--min-coverage 0.9"

while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1<2000' | wc -l)" -lt 3 ]; do
  echo "$(date +%H:%M) waiting for 3 idle GPUs"; sleep 600; done
FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk '$2<2000{print $1}' | tr '\n' ' ')
set -- $FREE; G1=$1; G2=${2:-$1}; G3=${3:-$1}
echo "GPUs: $FREE"

# ctl_qty: 3 seeds of the same random-pair manifest
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$G1 $PY -B src/train_frame_mil.py --window 0 \
    --manifest manifests/ctl_qty.parquet --caches $TRAIN $EVAL $DEV $COV \
    --seed $s --out runs/EC_ctl_qty_s$s > logs/ec_qty_s$s.log 2>&1
  [ -f runs/EC_ctl_qty_s$s/metrics.json ] || echo "FAILED EC_ctl_qty_s$s"
done &

# ctl_kids: one seed per child-drop draw, so the average is over DRAWS not just inits
for s in 0 1 2; do
  g=$([ $s -eq 0 ] && echo $G2 || echo $G3)
  CUDA_VISIBLE_DEVICES=$g $PY -B src/train_frame_mil.py --window 0 \
    --manifest manifests/ctl_kids_s$s.parquet --caches $TRAIN $EVAL $DEV $COV \
    --seed $s --out runs/EC_ctl_kids_s$s > logs/ec_kids_s$s.log 2>&1
  [ -f runs/EC_ctl_kids_s$s/metrics.json ] || echo "FAILED EC_ctl_kids_s$s"
done
wait

$PY src/scrape_runs.py > /dev/null 2>&1
$PY - <<'EOF'
import json, glob, numpy as np
def fam(p):
    v=[json.load(open(f))["reported_test_acc"] for f in glob.glob(f"runs/{p}_s*/metrics.json")]
    return (np.mean(v), np.std(v, ddof=1) if len(v)>1 else 0.0, len(v)) if v else (float("nan"),0,0)
print("\n=== WHY IS THE ENGLISH ARM BELOW THE CURVE? (790,519 pairs everywhere) ===")
for lab, p in [("english >=80%  (28 kids)", "P5_lad_region_en"),
               ("ctl_qty        (36 kids)", "EC_ctl_qty"),
               ("ctl_kids       (28 kids, random drop)", "EC_ctl_kids")]:
    m,s,n = fam(p); print(f"  {lab:38s} {m:6.2f} ± {s:4.2f}  (n={n})")
print("  full corpus 904,812 / 36 kids            72.55")
print("  curve prediction at 790k (all 36 kids)   70.80")
EOF
echo "EC_DONE"
