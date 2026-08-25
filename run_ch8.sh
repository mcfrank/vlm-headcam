#!/bin/bash
# ch8 rerun, stage 2: encoder comparison on the clean 2026.1 rig.
# Encoders: DINOv3-L OTS (internet) vs DINOv3-L-BV (developmental-only), each at
# full corpus + 300k + 100k x 3 seeds -> encoder x scale, directly comparable to the
# B26 DINOv3-B arms. Waits for the embedding driver. Trainer unions the 4 shard dirs.
set -u
cd "$(dirname "$0")"
export HF_HOME=/ccn2/u/khaiaw/.cache/huggingface HF_HUB_OFFLINE=1
PY=/ccn2/u/khaiaw/miniconda3/envs/ccwm/bin/python
OUTROOT=/ccn2b/dataset/babyview/2026.1/outputs/image_embeddings
until grep -q "CH8_EMBED_ALL_DONE" logs/ch8_embed_driver.log 2>/dev/null; do
  echo "$(date +%H:%M) waiting for ch8 embeddings"; sleep 900; done
while :; do
  FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F", " '$2<20000{print $1}' | tr '\n' ' ')
  GPUS=($FREE); NG=${#GPUS[@]}
  [ "$NG" -ge 1 ] && break
  echo "$(date +%H:%M) no usable GPU"; sleep 600
done
i=0; echo "GPUs: $FREE"
one () {  # one <enc> <tag> <manifest> <seed>
  local enc=$1 tag=$2 man=$3 s=$4
  [ -f "runs/C8_${enc}_${tag}_s$s/metrics.json" ] && return
  local g=${GPUS[$((i % NG))]}; i=$((i+1))
  CUDA_VISIBLE_DEVICES=$g $PY -B src/train_frame_mil.py --window 0 \
    --manifest manifests/$man.parquet \
    --caches $OUTROOT/$enc/shard_0 $OUTROOT/$enc/shard_1 $OUTROOT/$enc/shard_2 $OUTROOT/$enc/shard_3 \
    --eval-cache emb_ch8_eval/${enc}_konkle --eval-frames manifests/eval_frames_konkle.parquet \
    --dev-cache emb_ch8_eval/${enc}_konkle_dev --dev-frames manifests/eval_frames_konkle_dev.parquet \
    --min-coverage 0.9 --seed $s --out runs/C8_${enc}_${tag}_s$s > logs/c8_${enc}_${tag}_s$s.log 2>&1 &
  [ $((i % NG)) -eq 0 ] && wait
}
for enc in dinov3l_grid4x4 dinov3l_bv_grid4x4; do
  for s in 0 1 2; do one $enc base bv26_base $s; done
  for s in 0 1 2; do one $enc rand_300000 bv26_rand_300000_s$s $s; done
  for s in 0 1 2; do one $enc rand_100000 bv26_rand_100000_s$s $s; done
done
wait
echo "CH8_DONE: $(ls runs/C8_*/metrics.json 2>/dev/null | wc -l)/18 ($(date))"
