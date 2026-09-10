#!/bin/bash
# +-5 s temporal-window MIL control (SI), complete windows: 4 encoders x {30k x5, 300k x3}.
# Caches = main region cache + neighbor cache (trainer unions cache dirs). Gated on embeds.
set -u
cd "$(dirname "$0")/.."   # repo root
export HF_HOME=/ccn2/u/khaiaw/.cache/huggingface HF_HUB_OFFLINE=1
PY=/ccn2/u/khaiaw/miniconda3/envs/ccwm/bin/python
EMB=/ccn2b/dataset/babyview/2026.1/outputs/image_embeddings; W5=/data2/mcfrank/emb_win5
until grep -q "WIN5_EMBED_DONE: 8/8" logs/win5_embed.log 2>/dev/null; do sleep 900; done
declare -A CACHE EV DV
CACHE[dinov3b]="$EMB/dinov3b_grid4x4 $W5/dinov3b/shard_0 $W5/dinov3b/shard_1"; EV[dinov3b]=emb_enc_grid_eval/dinov3b_ots_konkle; DV[dinov3b]=emb_dv3_konkle_dev16
CACHE[dinov3l]="$(ls -d $EMB/dinov3l_grid4x4/shard_* $W5/dinov3l/shard_* | tr '\n' ' ')"; EV[dinov3l]=emb_ch8_eval/dinov3l_grid4x4_konkle; DV[dinov3l]=emb_ch8_eval/dinov3l_grid4x4_konkle_dev
CACHE[vits_bv]="$(ls -d $EMB/vits_bv_grid4x4/shard_* | tr '\n' ' ') $W5/vits_bv/shard_0 $W5/vits_bv/shard_1"; EV[vits_bv]=emb_ch8_eval/vits_bv_konkle; DV[vits_bv]=emb_ch8_eval/vits_bv_konkle_dev
CACHE[vitb_bv]="$(ls -d $EMB/vitb_bv_grid4x4/shard_* | tr '\n' ' ') $W5/vitb_bv/shard_0 $W5/vitb_bv/shard_1"; EV[vitb_bv]=emb_ch8_eval/vitb_bv_konkle; DV[vitb_bv]=emb_ch8_eval/vitb_bv_konkle_dev
# GPUs 0-5 belong to the L-BV DINO trainer (rank 0 needs ~38 GB; a <41 GB-used rule would land
# on top of it and OOM it). Pin to GPUs 6-7, two runs per GPU.
GPUS=(6 7 6 7); NG=4
i=0; echo "GPUs: ${GPUS[*]}"
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
  for s in 0 1 2 3 4; do one $e win5_30000 bv26a_rand_30000_s$s $s; done
  for s in 0 1 2;     do one $e win5_300000 bv26a_rand_300000_s$s $s; done
done
wait
echo "WINDOW_DONE: $(ls runs/F_*win5*/metrics.json 2>/dev/null | wc -l)/32 ($(date))"
