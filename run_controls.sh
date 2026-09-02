#!/bin/bash
# Supplementary controls on the final rig (L-OTS unless noted):
#  A. alignment-selection controls (12 runs): alignedonly / matchrand (exposure+length
#     matched) / minusaligned / minusrand, 3 seeds each
#  B. temporal-window MIL control (60 runs): --window 5 (max over regions x cached frames
#     within +-5 s), 4 encoders x {30k,300k,full} x 5 seeds, paired to the region manifests
#     like run_wf.sh. Caches hold only pair-frames, so window occupancy is partial — logged.
set -u
cd "$(dirname "$0")"
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
echo "=== A. alignment-selection controls ==="
for s in 0 1 2; do
  one dinov3l alignedonly  bv26a_alignedonly      $s 0
  one dinov3l matchrand    bv26a_matchrand_s$s    $s 0
  one dinov3l minusaligned bv26a_minusaligned     $s 0
  one dinov3l minusrand    bv26a_minusrand_s$s    $s 0
done
wait
echo "=== B. temporal-window (+-5) control ==="
for e in dinov3l dinov3b vitb_bv vits_bv; do for s in 0 1 2 3 4; do
  one $e win5_30000  bv26a_rand_30000_s$s  $s 5
  one $e win5_300000 bv26a_rand_300000_s$s $s 5
  one $e win5_full   bv26a_base            $s 5
done; done
wait
echo "CONTROLS_DONE: A=$(ls runs/F_dinov3l_{alignedonly,matchrand,minusaligned,minusrand}_s*/metrics.json 2>/dev/null | wc -l)/12 B=$(ls runs/F_*win5*/metrics.json 2>/dev/null | wc -l)/60 ($(date))"
