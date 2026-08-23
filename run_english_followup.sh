#!/bin/bash
# Which children the language filter removes is what costs 4.3 pts — not count, not diversity.
# Prime suspect: S00240001 (61,002 pairs, 68% English), half of everything the >=80% cut removes.
#   en50_matched  >=50% English, matched to 790,519 (keeps S00240001 + S00680001)
#   ctl_drop_big  drop ONLY S00240001, matched to 790,519
set -u
cd /data2/mcfrank/vlm-headcam
export PYTHONPATH=src
PY=/data2/mcfrank/ladder/condaenv/bin/python
TRAIN=emb_dv3_grid_877k
EVAL="--eval-cache emb_enc_grid_eval/dinov3b_ots_konkle --eval-frames manifests/eval_frames_konkle.parquet"
DEV="--dev-cache emb_dv3_konkle_dev16 --dev-frames manifests/eval_frames_konkle_dev.parquet"
COV="--min-coverage 0.9"
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1<2000'|wc -l)" -lt 2 ]; do
  echo "$(date +%H:%M) waiting for GPUs"; sleep 600; done
FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk '$2<2000{print $1}' | tr '\n' ' ')
set -- $FREE; A=$1; B=${2:-$1}
for cfg in "en50_matched $A" "ctl_drop_big $B"; do
  set -- $cfg; man=$1; gpu=$2
  ( for s in 0 1 2; do
      CUDA_VISIBLE_DEVICES=$gpu $PY -B src/train_frame_mil.py --window 0 \
        --manifest manifests/$man.parquet --caches $TRAIN $EVAL $DEV $COV \
        --seed $s --out runs/EC_${man}_s$s > logs/ec_${man}_s$s.log 2>&1
      [ -f runs/EC_${man}_s$s/metrics.json ] || echo "FAILED EC_${man}_s$s"
    done ) &
done
wait
$PY src/scrape_runs.py > /dev/null 2>&1
$PY - <<'EOF'
import json, glob, numpy as np
def f(p):
    v=[json.load(open(x))["reported_test_acc"] for x in glob.glob(f"runs/{p}_s*/metrics.json")]
    return (np.mean(v), np.std(v,ddof=1) if len(v)>1 else 0, len(v)) if v else (float("nan"),0,0)
print("\n=== all at 790,519 pairs ===")
for lab,p in [("full corpus (904,812 / 36 kids)","P5_lad_region"),
              ("ctl_qty      random pairs, 36 kids","EC_ctl_qty"),
              ("ctl_kids     8 RANDOM kids dropped, 28","EC_ctl_kids"),
              ("ctl_drop_big drop S00240001 only, 35","EC_ctl_drop_big"),
              ("en50         >=50% English, 30 kids","EC_en50_matched"),
              ("en80         >=80% English, 28 kids","P5_lad_region_en")]:
    m,s,n=f(p); print(f"  {lab:42s} {m:6.2f} ± {s:4.2f} (n={n})")
EOF
echo "EC2_DONE"
