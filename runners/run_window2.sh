#!/bin/bash
# +-5 s temporal-window MIL control at LARGE scale (SI): 4 encoders x {1M x3, full x3} = 24 runs,
# paired to the region runs' manifests (rand_1000000_s<s>, base). Caches = main region cache +
# round-1 neighbors (emb_win5) + round-2 residual neighbors (emb_win5b) -> complete windows.
# Gated on the round-2 join marker. GPUs chosen at start: only cards with <5 GB used (never the
# L-BV trainer's), two runs per card.
set -u
cd "$(dirname "$0")/.."   # repo root
export HF_HOME=/ccn2/u/khaiaw/.cache/huggingface HF_HUB_OFFLINE=1
PY=/ccn2/u/khaiaw/miniconda3/envs/ccwm/bin/python
EMB=/ccn2b/dataset/babyview/2026.1/outputs/image_embeddings; W5=/data2/mcfrank/emb_win5; W5B=/data2/mcfrank/emb_win5b
until grep -q "WIN5B_EMBED_DONE: 8/8" logs/win5_embed.log 2>/dev/null; do sleep 900; done
declare -A CACHE EV DV
CACHE[dinov3b]="$EMB/dinov3b_grid4x4 $(ls -d $W5/dinov3b/shard_* $W5B/dinov3b/shard_* | tr '\n' ' ')"; EV[dinov3b]=emb_enc_grid_eval/dinov3b_ots_konkle; DV[dinov3b]=emb_dv3_konkle_dev16
CACHE[dinov3l]="$(ls -d $EMB/dinov3l_grid4x4/shard_* $W5/dinov3l/shard_* $W5B/dinov3l/shard_* | tr '\n' ' ')"; EV[dinov3l]=emb_ch8_eval/dinov3l_grid4x4_konkle; DV[dinov3l]=emb_ch8_eval/dinov3l_grid4x4_konkle_dev
CACHE[vits_bv]="$(ls -d $EMB/vits_bv_grid4x4/shard_* $W5/vits_bv/shard_* $W5B/vits_bv/shard_* | tr '\n' ' ')"; EV[vits_bv]=emb_ch8_eval/vits_bv_konkle; DV[vits_bv]=emb_ch8_eval/vits_bv_konkle_dev
CACHE[vitb_bv]="$(ls -d $EMB/vitb_bv_grid4x4/shard_* $W5/vitb_bv/shard_* $W5B/vitb_bv/shard_* | tr '\n' ' ')"; EV[vitb_bv]=emb_ch8_eval/vitb_bv_konkle; DV[vitb_bv]=emb_ch8_eval/vitb_bv_konkle_dev
while :; do
  FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F", " '$2<5000{print $1}' | tr '\n' ' ')
  GPUS=($FREE $FREE); NG=${#GPUS[@]}; [ "$NG" -ge 2 ] && break; sleep 600
done
i=0; echo "GPUs (x2 per card): ${GPUS[*]}"
one () {
  local e=$1 tag=$2 man=$3 s=$4
  [ -f "runs/F_${e}_${tag}_s$s/metrics.json" ] && return
  pgrep -f "out runs/F_${e}_${tag}_s$s( |$)" >/dev/null && return
  local g=${GPUS[$((i % NG))]}; i=$((i+1))
  CUDA_VISIBLE_DEVICES=$g $PY -B src/train_frame_mil.py --window 5 \
    --manifest manifests/$man.parquet --caches ${CACHE[$e]} \
    --eval-cache ${EV[$e]} --eval-frames manifests/eval_frames_konkle.parquet \
    --dev-cache ${DV[$e]} --dev-frames manifests/eval_frames_konkle_dev.parquet \
    --min-coverage 0.9 --seed $s --out runs/F_${e}_${tag}_s$s > logs/f_${e}_${tag}_s$s.log 2>&1 &
  [ $((i % NG)) -eq 0 ] && wait
}
for e in dinov3l dinov3b vitb_bv vits_bv; do
  for s in 0 1 2; do one $e win5_1000000 bv26a_rand_1000000_s$s $s; done
  for s in 0 1 2; do one $e win5_full    bv26a_base               $s; done
done
wait
echo "WINDOW2_DONE: $(ls runs/F_*_win5_{1000000,full}_s*/metrics.json 2>/dev/null | wc -l)/24 ($(date))"
