#!/bin/bash
# Figures-session request (2): diversity 30k series on the final rig — completes figS2's
# third budget line. F_dinov3l_div30k_{1,3,10,25,48}c x 5 seeds = 25 runs, L-OTS caches.
set -u
cd "$(dirname "$0")/.."   # repo root
export HF_HOME=/ccn2/u/khaiaw/.cache/huggingface HF_HUB_OFFLINE=1
PY=/ccn2/u/khaiaw/miniconda3/envs/ccwm/bin/python
EMB=/ccn2b/dataset/babyview/2026.1/outputs/image_embeddings
while :; do
  FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F", " '$2<41000{print $1}' | tr '\n' ' ')
  GPUS=($FREE); NG=${#GPUS[@]}; [ "$NG" -ge 1 ] && break
  sleep 600
done
i=0; echo "GPUs: $FREE"
one () {
  local k=$1 s=$2
  [ -f "runs/F_dinov3l_div30k_${k}c_s$s/metrics.json" ] && return
  pgrep -f "out runs/F_dinov3l_div30k_${k}c_s$s( |$)" > /dev/null && return
  local g=${GPUS[$((i % NG))]}; i=$((i+1))
  CUDA_VISIBLE_DEVICES=$g $PY -B src/train_frame_mil.py --window 0 \
    --manifest manifests/bv26a_div30k_${k}c_s$s.parquet \
    --caches $EMB/dinov3l_grid4x4/shard_0 $EMB/dinov3l_grid4x4/shard_1 $EMB/dinov3l_grid4x4/shard_2 $EMB/dinov3l_grid4x4/shard_3 \
    --eval-cache emb_ch8_eval/dinov3l_grid4x4_konkle --eval-frames manifests/eval_frames_konkle.parquet \
    --dev-cache emb_ch8_eval/dinov3l_grid4x4_konkle_dev --dev-frames manifests/eval_frames_konkle_dev.parquet \
    --min-coverage 0.9 --seed $s --out runs/F_dinov3l_div30k_${k}c_s$s > logs/f_div30k_${k}c_s$s.log 2>&1 &
  [ $((i % NG)) -eq 0 ] && wait
}
for k in 1 3 10 25 48; do for s in 0 1 2 3 4; do one $k $s; done; done
wait
echo "DIV30K_DONE: $(ls runs/F_dinov3l_div30k_*/metrics.json 2>/dev/null | wc -l)/25 ($(date))"
