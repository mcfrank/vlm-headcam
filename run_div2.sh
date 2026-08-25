#!/bin/bash
# Diversity at production floors: 100k x k in {1,3,10,25,50}, 300k x k in {10,25,50}, 5 draws
# each = 40 runs. Runs on whatever GPUs are free at launch; resumable.
set -u
cd "$(dirname "$0")"
export HF_HOME=/ccn2/u/khaiaw/.cache/huggingface HF_HUB_OFFLINE=1
PY=/ccn2/u/khaiaw/miniconda3/envs/ccwm/bin/python
CACHES=/ccn2b/dataset/babyview/2026.1/outputs/image_embeddings/dinov3b_grid4x4
until [ "$(ls manifests/bv26_div100k_* manifests/bv26_div300k_* 2>/dev/null | wc -l)" -ge 30 ]; do
  echo "$(date +%H:%M) waiting for div2 manifests"; sleep 300; done
# wait for at least one usable GPU (an A40 with <20G used has ample room for these runs);
# an empty list once made NG=0 and every launch died on a modulo — never assume GPUs exist
while :; do
  FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F", " '$2<20000{print $1}' | tr '\n' ' ')
  GPUS=($FREE); NG=${#GPUS[@]}
  [ "$NG" -ge 1 ] && break
  echo "$(date +%H:%M) no usable GPU"; sleep 600
done
i=0; echo "GPUs: $FREE"
one () {
  local tag=$1 s=$2
  [ -f "runs/B26_${tag}_s$s/metrics.json" ] && return
  [ -f "manifests/bv26_${tag}_s$s.parquet" ] || return
  local g=${GPUS[$((i % NG))]}; i=$((i+1))
  CUDA_VISIBLE_DEVICES=$g $PY -B src/train_frame_mil.py --window 0 \
    --manifest manifests/bv26_${tag}_s$s.parquet --caches $CACHES \
    --eval-cache emb_enc_grid_eval/dinov3b_ots_konkle --eval-frames manifests/eval_frames_konkle.parquet \
    --dev-cache emb_dv3_konkle_dev16 --dev-frames manifests/eval_frames_konkle_dev.parquet \
    --min-coverage 0.9 --seed $s --out runs/B26_${tag}_s$s > logs/b26_${tag}_s$s.log 2>&1 &
  [ $((i % NG)) -eq 0 ] && wait
}
for k in 1 3 10 25 50; do for s in 0 1 2 3 4; do one div100k_${k}c $s; done; done
for k in 10 25 50;     do for s in 0 1 2 3 4; do one div300k_${k}c $s; done; done
wait
echo "DIV2_DONE: $(ls runs/B26_div1*/metrics.json runs/B26_div3*/metrics.json 2>/dev/null | wc -l) ($(date))"
