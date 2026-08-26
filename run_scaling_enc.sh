#!/bin/bash
# PREVIEW scaling curves for the encoder comparison: DINOv3-L OTS (internet) and DINOv3-L-BV
# (developmental) on the SAME manifests/draws as the B-OTS curve (transcript-filter corpus) so
# all three encoders are point-for-point comparable. Final versions rerun on the audio-filtered
# corpus. Fills the cells C8 didn't cover; resumable; in-flight-guarded.
set -u
cd "$(dirname "$0")"
export HF_HOME=/ccn2/u/khaiaw/.cache/huggingface HF_HUB_OFFLINE=1
PY=/ccn2/u/khaiaw/miniconda3/envs/ccwm/bin/python
OUTROOT=/ccn2b/dataset/babyview/2026.1/outputs/image_embeddings
while :; do
  FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F", " '$2<20000{print $1}' | tr '\n' ' ')
  GPUS=($FREE); NG=${#GPUS[@]}
  [ "$NG" -ge 1 ] && break
  echo "$(date +%H:%M) no usable GPU"; sleep 600
done
i=0; echo "GPUs: $FREE"
one () {  # one <enc> <N> <seed>
  local enc=$1 N=$2 s=$3
  [ -f "runs/C8_${enc}_rand_${N}_s$s/metrics.json" ] && return
  pgrep -f "out runs/C8_${enc}_rand_${N}_s$s( |$)" > /dev/null && return
  [ -f "manifests/bv26_rand_${N}_s$s.parquet" ] || { echo "MISSING bv26_rand_${N}_s$s"; return; }
  local g=${GPUS[$((i % NG))]}; i=$((i+1))
  CUDA_VISIBLE_DEVICES=$g $PY -B src/train_frame_mil.py --window 0 \
    --manifest manifests/bv26_rand_${N}_s$s.parquet \
    --caches $OUTROOT/$enc/shard_0 $OUTROOT/$enc/shard_1 $OUTROOT/$enc/shard_2 $OUTROOT/$enc/shard_3 \
    --eval-cache emb_ch8_eval/${enc}_konkle --eval-frames manifests/eval_frames_konkle.parquet \
    --dev-cache emb_ch8_eval/${enc}_konkle_dev --dev-frames manifests/eval_frames_konkle_dev.parquet \
    --min-coverage 0.9 --seed $s --out runs/C8_${enc}_rand_${N}_s$s > logs/c8_${enc}_rand_${N}_s$s.log 2>&1 &
  [ $((i % NG)) -eq 0 ] && wait
}
for enc in dinov3l_grid4x4 dinov3l_bv_grid4x4; do
  for N in 3000 10000 30000; do for s in 0 1 2 3 4 5 6 7; do one $enc $N $s; done; done
  for s in 3 4 5 6 7; do one $enc 100000 $s; done
  for s in 3 4;       do one $enc 300000 $s; done
  for s in 0 1 2;     do one $enc 1000000 $s; done
done
wait
echo "SCALING_ENC_DONE: $(ls runs/C8_*rand*/metrics.json 2>/dev/null | wc -l) rand cells ($(date))"
