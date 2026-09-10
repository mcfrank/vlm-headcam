#!/bin/bash
# Alignment-selection controls, round 2 (post-audit; see src/build_aligned_controls2.py):
#   rand172k    plain random |A| draw, 3 seeds     (neutral size-matched reference for matchrand)
#   minusmatch  full corpus minus matchrand_s, 3 seeds (the removal the text describes)
# all four encoders -> 24 runs. Same rig as run_controls.sh (window 0, dev-selected epoch).
set -u
cd "$(dirname "$0")/.."   # repo root
export HF_HOME=/ccn2/u/khaiaw/.cache/huggingface HF_HUB_OFFLINE=1
PY=/ccn2/u/khaiaw/miniconda3/envs/ccwm/bin/python
EMB=/ccn2b/dataset/babyview/2026.1/outputs/image_embeddings
declare -A CACHE EV DV
CACHE[dinov3b]="$EMB/dinov3b_grid4x4"; EV[dinov3b]=emb_enc_grid_eval/dinov3b_ots_konkle; DV[dinov3b]=emb_dv3_konkle_dev16
CACHE[dinov3l]="$(ls -d $EMB/dinov3l_grid4x4/shard_* | tr '\n' ' ')"; EV[dinov3l]=emb_ch8_eval/dinov3l_grid4x4_konkle; DV[dinov3l]=emb_ch8_eval/dinov3l_grid4x4_konkle_dev
CACHE[vits_bv]="$(ls -d $EMB/vits_bv_grid4x4/shard_* | tr '\n' ' ')"; EV[vits_bv]=emb_ch8_eval/vits_bv_konkle; DV[vits_bv]=emb_ch8_eval/vits_bv_konkle_dev
CACHE[vitb_bv]="$(ls -d $EMB/vitb_bv_grid4x4/shard_* | tr '\n' ' ')"; EV[vitb_bv]=emb_ch8_eval/vitb_bv_konkle; DV[vitb_bv]=emb_ch8_eval/vitb_bv_konkle_dev
while :; do
  FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F", " '$2<41000{print $1}' | tr '\n' ' ')
  GPUS=($FREE); NG=${#GPUS[@]}; [ "$NG" -ge 1 ] && break; sleep 600
done
i=0; echo "GPUs: $FREE"
one () {  # one <enc> <tag> <manifest> <seed> <window>
  local e=$1 tag=$2 man=$3 s=$4 w=$5
  [ -f "runs/F_${e}_${tag}_s$s/metrics.json" ] && return
  pgrep -f "out runs/F_${e}_${tag}_s$s( |$)" >/dev/null && return
  [ -f "manifests/$man.parquet" ] || { echo "MISSING $man"; return; }
  local g=${GPUS[$((i % NG))]}; i=$((i+1))
  CUDA_VISIBLE_DEVICES=$g $PY -B src/train_frame_mil.py --window $w \
    --manifest manifests/$man.parquet --caches ${CACHE[$e]} \
    --eval-cache ${EV[$e]} --eval-frames manifests/eval_frames_konkle.parquet \
    --dev-cache ${DV[$e]} --dev-frames manifests/eval_frames_konkle_dev.parquet \
    --min-coverage 0.9 --seed $s --out runs/F_${e}_${tag}_s$s > logs/f_${e}_${tag}_s$s.log 2>&1 &
  [ $((i % NG)) -eq 0 ] && wait
}
echo "=== B1. plain random 172k, all four encoders ==="
for e in dinov3l dinov3b vitb_bv vits_bv; do for s in 0 1 2; do
  one $e rand172k bv26a_rand172k_s$s $s 0
done; done
wait
echo "=== B2. full minus matched, all four encoders ==="
for e in dinov3l dinov3b vitb_bv vits_bv; do for s in 0 1 2; do
  one $e minusmatch bv26a_minusmatch_s$s $s 0
done; done
wait
echo "CONTROLS_B_DONE: $(ls runs/F_*_{rand172k,minusmatch}_s*/metrics.json 2>/dev/null | wc -l)/24 ($(date))"
