#!/bin/bash
# Bundle 2 on the consolidated 2026.1 release — SINGLE ARM: the lightly English-filtered set
# (English utterances only, videos >=50% measured English) is the corpus for everything.
#   ladder (full corpus + matched subsamples at 300k/100k), scaling (per-seed subsamples,
#   8 draws <=100k / 5 at 300k / 3 at 1M), diversity (random child draws, 5 seeds).
# Run AFTER build (this script only trains): bash run_2026_studies.sh
set -u
cd "$(dirname "$0")"
export HF_HOME=/ccn2/u/khaiaw/.cache/huggingface HF_HUB_OFFLINE=1
PY=/ccn2/u/khaiaw/miniconda3/envs/ccwm/bin/python
CACHES=/ccn2b/dataset/babyview/2026.1/outputs/image_embeddings/dinov3b_grid4x4
EVAL="--eval-cache emb_enc_grid_eval/dinov3b_ots_konkle --eval-frames manifests/eval_frames_konkle.parquet"
DEV="--dev-cache emb_dv3_konkle_dev16 --dev-frames manifests/eval_frames_konkle_dev.parquet"
COV="--min-coverage 0.9"

while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1<2000'|wc -l)" -lt 3 ]; do
  echo "$(date +%H:%M) waiting for 3 idle GPUs"; sleep 600; done
FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk '$2<2000{print $1}' | tr '\n' ' ')
echo "GPUs: $FREE"; GPUS=($FREE); NG=${#GPUS[@]}; i=0

one () {  # one <tag> <manifest> <seed>
  local tag=$1 man=$2 s=$3
  [ -f "runs/B26_${tag}_s$s/metrics.json" ] && return          # resumable
  [ -f "manifests/$man.parquet" ] || { echo "  MISSING manifests/$man.parquet"; return; }
  local g=${GPUS[$((i % NG))]}; i=$((i+1))
  CUDA_VISIBLE_DEVICES=$g $PY -B src/train_frame_mil.py --window 0 \
    --manifest manifests/$man.parquet --caches $CACHES $EVAL $DEV $COV \
    --seed $s --out runs/B26_${tag}_s$s > logs/b26_${tag}_s$s.log 2>&1 &
  [ $((i % NG)) -eq 0 ] && wait
}

echo "=== 1. scaling (queued first) ==="
for N in 3000 10000 30000 100000; do for s in 0 1 2 3 4 5 6 7; do one rand_$N bv26_rand_${N}_s$s $s; done; done
for s in 0 1 2 3 4; do one rand_300000 bv26_rand_300000_s$s $s; done
for s in 0 1 2;     do one rand_1000000 bv26_rand_1000000_s$s $s; done
wait
echo "=== 2. diversity ==="
for k in 1 3 10 25 50; do for s in 0 1 2 3 4; do one div_${k}c bv26_div_${k}c_s$s $s; done; done
wait
echo "=== 3. ladder: full corpus (fixed manifest, 3 init seeds) ==="
for r in base filtnat t15 t2; do for s in 0 1 2; do one lad_$r bv26_$r $s; done; done
wait
echo "=== 4. ladder at reduced scale (matched subsample per seed) ==="
for N in 300000 100000; do for r in base filtnat t15 t2; do for s in 0 1 2; do
  one lad${N}_$r bv26_lad${N}_${r}_s$s $s; done; done; done
wait
n=$(ls -d runs/B26_* 2>/dev/null | wc -l); m=$(ls runs/B26_*/metrics.json 2>/dev/null | wc -l)
echo "B26_DONE: $n dirs, $m with metrics.json ($(date))"
