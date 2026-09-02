#!/bin/bash
# Whole-frame (no-MIL) SI comparison, final rig: mean-over-grid R=1 caches (the final frame
# caches are drop-CLS, so cls_only would read a corner cell — mean-pool is the correct
# whole-frame baseline, matching the book's original "meanpatch"). 4 encoders x 3 scales
# (30k / 300k / full) x 5 seeds. GPU threshold relaxed to <41G: these runs are ~5G and
# coexist with the DINO training campaign on 48G cards — no need to clear anyone., paired to the region runs' manifests, plus region base
# top-ups s3/s4 so the full-scale pairing is n=5 on both sides. 68 runs.
set -u
cd "$(dirname "$0")"
export HF_HOME=/ccn2/u/khaiaw/.cache/huggingface HF_HUB_OFFLINE=1
PY=/ccn2/u/khaiaw/miniconda3/envs/ccwm/bin/python
EMB=/ccn2b/dataset/babyview/2026.1/outputs/image_embeddings
until grep -q "WF_CACHES_DONE" logs/wfprep.log 2>/dev/null; do
  echo "$(date +%H:%M) waiting for wf caches"; sleep 300; done
while :; do
  FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F", " '$2<41000{print $1}' | tr '\n' ' ')
  GPUS=($FREE); NG=${#GPUS[@]}; [ "$NG" -ge 1 ] && break
  echo "$(date +%H:%M) no usable GPU"; sleep 600
done
i=0; echo "GPUs: $FREE"

declare -A RCACHE
RCACHE[dinov3b]="$EMB/dinov3b_grid4x4"
RCACHE[dinov3l]="$EMB/dinov3l_grid4x4/shard_0 $EMB/dinov3l_grid4x4/shard_1 $EMB/dinov3l_grid4x4/shard_2 $EMB/dinov3l_grid4x4/shard_3"
RCACHE[vits_bv]="$(ls -d $EMB/vits_bv_grid4x4/shard_* 2>/dev/null | tr '\n' ' ')"
RCACHE[vitb_bv]="$(ls -d $EMB/vitb_bv_grid4x4/shard_* 2>/dev/null | tr '\n' ' ')"

one () {  # one <enc> <tag> <manifest> <seed> <mode:wf|region>
  local e=$1 tag=$2 man=$3 s=$4 mode=$5
  [ -f "runs/F_${e}_${tag}_s$s/metrics.json" ] && return
  pgrep -f "out runs/F_${e}_${tag}_s$s( |$)" > /dev/null && return
  local caches ev dv
  if [ "$mode" = wf ]; then
    caches=$(ls -d emb_wf/$e/shard_* 2>/dev/null | tr '\n' ' '); [ -z "$caches" ] && caches="emb_wf/$e"
    ev="emb_wf_eval/${e}_konkle"; dv="emb_wf_eval/${e}_konkle_dev"
  else
    caches="${RCACHE[$e]}"
    case $e in
      dinov3b) ev=emb_enc_grid_eval/dinov3b_ots_konkle; dv=emb_dv3_konkle_dev16;;
      dinov3l) ev=emb_ch8_eval/dinov3l_grid4x4_konkle; dv=emb_ch8_eval/dinov3l_grid4x4_konkle_dev;;
      *) ev=emb_ch8_eval/${e}_konkle; dv=emb_ch8_eval/${e}_konkle_dev;;
    esac
  fi
  local g=${GPUS[$((i % NG))]}; i=$((i+1))
  CUDA_VISIBLE_DEVICES=$g $PY -B src/train_frame_mil.py --window 0 \
    --manifest manifests/$man.parquet --caches $caches \
    --eval-cache $ev --eval-frames manifests/eval_frames_konkle.parquet \
    --dev-cache $dv --dev-frames manifests/eval_frames_konkle_dev.parquet \
    --min-coverage 0.9 --seed $s --out runs/F_${e}_${tag}_s$s > logs/f_${e}_${tag}_s$s.log 2>&1 &
  [ $((i % NG)) -eq 0 ] && wait
}

for e in dinov3l dinov3b vitb_bv vits_bv; do
  for s in 0 1 2 3 4; do
    one $e wf30000 bv26a_rand_30000_s$s $s wf
    one $e wf300000 bv26a_rand_300000_s$s $s wf
    one $e wffull bv26a_base $s wf
  done
  for s in 3 4; do one $e base bv26a_base $s region; done   # top-up full-scale region to n=5
done
wait
echo "WF_DONE: $(ls runs/F_*wf*/metrics.json 2>/dev/null | wc -l)/60 wf + $(ls runs/F_*_base_s[34]/metrics.json 2>/dev/null | wc -l)/8 topups ($(date))"
