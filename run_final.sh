#!/bin/bash
# FINAL rerun (family prefix F_): audio-filtered corpus (bv26a manifests, 1.686M pairs,
# 48 children), four encoders. Per encoder: scaling (7 pts, per-seed draws) + ladder (full)
# + ALIGNED scaling (4 pts x3). L-OTS additionally: ladder@300k/@100k + diversity v2.
# An encoder's arms run only when BOTH its frame cache and konkle eval caches exist, so the
# OTS arms start immediately and the new BV encoders join when the C9/eval caches land.
# Resumable; in-flight-guarded; frame-coverage gated. Relaunch anytime to fill gaps.
set -u
cd "$(dirname "$0")"
export HF_HOME=/ccn2/u/khaiaw/.cache/huggingface HF_HUB_OFFLINE=1
PY=/ccn2/u/khaiaw/miniconda3/envs/ccwm/bin/python
EMB=/ccn2b/dataset/babyview/2026.1/outputs/image_embeddings

declare -A CACHE EVALC DEVC
CACHE[dinov3b]="$EMB/dinov3b_grid4x4"
EVALC[dinov3b]="emb_enc_grid_eval/dinov3b_ots_konkle";  DEVC[dinov3b]="emb_dv3_konkle_dev16"
CACHE[dinov3l]="$EMB/dinov3l_grid4x4/shard_0 $EMB/dinov3l_grid4x4/shard_1 $EMB/dinov3l_grid4x4/shard_2 $EMB/dinov3l_grid4x4/shard_3"
EVALC[dinov3l]="emb_ch8_eval/dinov3l_grid4x4_konkle"; DEVC[dinov3l]="emb_ch8_eval/dinov3l_grid4x4_konkle_dev"
CACHE[vits_bv]="$EMB/vits_bv_grid4x4"
EVALC[vits_bv]="emb_ch8_eval/vits_bv_konkle";  DEVC[vits_bv]="emb_ch8_eval/vits_bv_konkle_dev"
CACHE[vitb_bv]="$EMB/vitb_bv_grid4x4"
EVALC[vitb_bv]="emb_ch8_eval/vitb_bv_konkle";  DEVC[vitb_bv]="emb_ch8_eval/vitb_bv_konkle_dev"

while :; do
  FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F", " '$2<20000{print $1}' | tr '\n' ' ')
  GPUS=($FREE); NG=${#GPUS[@]}; [ "$NG" -ge 1 ] && break
  echo "$(date +%H:%M) no usable GPU"; sleep 600
done
i=0; echo "GPUs: $FREE"

enc_ready () {  # cache dir(s) + eval caches all present
  local e=$1
  for d in ${CACHE[$e]}; do
    { [ -f "$d/index.parquet" ] || [ -f "$d/shard_0/index.parquet" ] || ls $d/shard_*/index.parquet >/dev/null 2>&1 || [ -f "$d/emb.f16.npy" ]; } || return 1
  done
  [ -f "${EVALC[$e]}/index.parquet" ] && [ -f "${DEVC[$e]}/index.parquet" ]
}

one () {  # one <enc> <tag> <manifest> <seed>
  local e=$1 tag=$2 man=$3 s=$4
  [ -f "runs/F_${e}_${tag}_s$s/metrics.json" ] && return
  pgrep -f "out runs/F_${e}_${tag}_s$s( |$)" > /dev/null && return
  [ -f "manifests/$man.parquet" ] || { echo "MISSING manifests/$man.parquet"; return; }
  local caches=""
  for d in ${CACHE[$e]}; do
    if [ -f "$d/emb.f16.npy" ]; then caches="$caches $d"
    else for sh in $d/shard_*; do [ -d "$sh" ] && caches="$caches $sh"; done; fi
  done
  local g=${GPUS[$((i % NG))]}; i=$((i+1))
  CUDA_VISIBLE_DEVICES=$g $PY -B src/train_frame_mil.py --window 0 \
    --manifest manifests/$man.parquet --caches $caches \
    --eval-cache ${EVALC[$e]} --eval-frames manifests/eval_frames_konkle.parquet \
    --dev-cache ${DEVC[$e]} --dev-frames manifests/eval_frames_konkle_dev.parquet \
    --min-coverage 0.9 --seed $s --out runs/F_${e}_${tag}_s$s > logs/f_${e}_${tag}_s$s.log 2>&1 &
  [ $((i % NG)) -eq 0 ] && wait
}

for pass in 1 2 3 4 5 6; do
  did_any=0
  for e in dinov3l dinov3b vits_bv vitb_bv; do
    enc_ready $e || { echo "pass $pass: $e caches not ready, skipping"; continue; }
    did_any=1
    echo "=== pass $pass: $e ($(date +%H:%M)) ==="
    for N in 3000 10000 30000 100000; do for s in 0 1 2 3 4 5 6 7; do one $e rand_$N bv26a_rand_${N}_s$s $s; done; done
    for s in 0 1 2 3 4; do one $e rand_300000 bv26a_rand_300000_s$s $s; done
    for s in 0 1 2;     do one $e rand_1000000 bv26a_rand_1000000_s$s $s; done
    for s in 0 1 2;     do one $e base bv26a_base $s; done
    for r in filtnat t15 t2; do for s in 0 1 2; do one $e lad_$r bv26a_$r $s; done; done
    for N in 10000 30000 100000 170000; do for s in 0 1 2; do one $e align_$N bv26a_align_$N $s; done; done
  done
  wait
  # L-OTS extras
  if enc_ready dinov3l; then
    for N in 300000 100000; do for r in base filtnat t15 t2; do for s in 0 1 2; do
      one dinov3l lad${N}_$r bv26a_lad${N}_${r}_s$s $s; done; done; done
    for k in 1 3 10 25 48; do for s in 0 1 2 3 4; do one dinov3l div100k_${k}c bv26a_div100k_${k}c_s$s $s; done; done
    for k in 10 25 48;     do for s in 0 1 2 3 4; do one dinov3l div300k_${k}c bv26a_div300k_${k}c_s$s $s; done; done
    wait
  fi
  n=$(ls runs/F_*/metrics.json 2>/dev/null | wc -l)
  echo "pass $pass done: $n/320 metrics ($(date))"
  [ "$n" -ge 320 ] && break
  [ "$did_any" -eq 0 ] && sleep 1800 || sleep 300
done
echo "F_DONE: $(ls runs/F_*/metrics.json 2>/dev/null | wc -l)/320 ($(date))"
