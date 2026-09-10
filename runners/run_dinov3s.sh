#!/bin/bash
# S-OTS (dinov3s) — the complete per-encoder sequence (113 runs), mirror of run_vitl.sh.
# Gates: S_MAIN_DONE (dinov3s_driver.sh), WF_ENC_DONE dinov3s, S_WIN5_DONE: 8/8.
# GPUs: every card, two runs per card (window embeds for this encoder, ~2 GB each, coexist).
set -u
cd "$(dirname "$0")/.."   # repo root
export HF_HOME=/ccn2/u/khaiaw/.cache/huggingface HF_HUB_OFFLINE=1
PY=/ccn2/u/khaiaw/miniconda3/envs/ccwm/bin/python
EMB=/ccn2b/dataset/babyview/2026.1/outputs/image_embeddings
E=dinov3s
EV=emb_ch8_eval/dinov3s_konkle; DV=emb_ch8_eval/dinov3s_konkle_dev
until grep -q "S_MAIN_DONE" logs/dinov3s_driver.log 2>/dev/null; do sleep 600; done
RCACHE="$(ls -d $EMB/dinov3s_grid4x4/shard_* | tr '\n' ' ')"
[ -n "$RCACHE" ] || { echo "no dinov3s frame cache"; exit 1; }
GPUS=(0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7); NG=${#GPUS[@]}
i=0; echo "dinov3s runs start $(date); GPUs x2: ${GPUS[*]}"
one () {  # one <tag> <manifest> <seed> <window> <caches> <ev> <dv>
  local tag=$1 man=$2 s=$3 w=$4 caches=$5 ev=$6 dv=$7
  [ -f "runs/F_${E}_${tag}_s$s/metrics.json" ] && return
  pgrep -f "out runs/F_${E}_${tag}_s$s( |$)" >/dev/null && return
  [ -f "manifests/$man.parquet" ] || { echo "MISSING manifests/$man.parquet"; return; }
  local g=${GPUS[$((i % NG))]}; i=$((i+1))
  CUDA_VISIBLE_DEVICES=$g $PY -B src/train_frame_mil.py --window $w \
    --manifest manifests/$man.parquet --caches $caches \
    --eval-cache $ev --eval-frames manifests/eval_frames_konkle.parquet \
    --dev-cache $dv --dev-frames manifests/eval_frames_konkle_dev.parquet \
    --min-coverage 0.9 --seed $s --out runs/F_${E}_${tag}_s$s > logs/f_${E}_${tag}_s$s.log 2>&1 &
  [ $((i % NG)) -eq 0 ] && wait
}
echo "=== A. scaling / ladder / aligned / alignment controls ==="
for N in 3000 10000 30000 100000; do for s in 0 1 2 3 4 5 6 7; do one rand_$N bv26a_rand_${N}_s$s $s 0 "$RCACHE" $EV $DV; done; done
for s in 0 1 2 3 4; do one rand_300000 bv26a_rand_300000_s$s $s 0 "$RCACHE" $EV $DV; done
for s in 0 1 2;     do one rand_1000000 bv26a_rand_1000000_s$s $s 0 "$RCACHE" $EV $DV; done
for s in 0 1 2 3 4; do one base bv26a_base $s 0 "$RCACHE" $EV $DV; done
for r in filtnat t15 t2; do for s in 0 1 2; do one lad_$r bv26a_$r $s 0 "$RCACHE" $EV $DV; done; done
for N in 10000 30000 100000 170000; do for s in 0 1 2; do one align_$N bv26a_align_$N $s 0 "$RCACHE" $EV $DV; done; done
for s in 0 1 2; do
  one alignedonly  bv26a_alignedonly      $s 0 "$RCACHE" $EV $DV
  one rand172k     bv26a_rand172k_s$s     $s 0 "$RCACHE" $EV $DV
  one matchrand    bv26a_matchrand_s$s    $s 0 "$RCACHE" $EV $DV
  one minusaligned bv26a_minusaligned     $s 0 "$RCACHE" $EV $DV
  one minusmatch   bv26a_minusmatch_s$s   $s 0 "$RCACHE" $EV $DV
  one minusrand    bv26a_minusrand_s$s    $s 0 "$RCACHE" $EV $DV
done
wait
echo "A done: $(ls runs/F_${E}_*/metrics.json | wc -l) ($(date))"
echo "=== B. no-MIL (whole-frame) ==="
until grep -q "WF_ENC_DONE dinov3s" logs/wfprep_dinov3s.log 2>/dev/null; do sleep 600; done
WFC="$(ls -d emb_wf/dinov3s/shard_* 2>/dev/null | tr '\n' ' ')"; [ -z "$WFC" ] && WFC=emb_wf/dinov3s
for s in 0 1 2 3 4; do
  one wf30000  bv26a_rand_30000_s$s  $s 0 "$WFC" emb_wf_eval/dinov3s_konkle emb_wf_eval/dinov3s_konkle_dev
  one wf300000 bv26a_rand_300000_s$s $s 0 "$WFC" emb_wf_eval/dinov3s_konkle emb_wf_eval/dinov3s_konkle_dev
  one wffull   bv26a_base            $s 0 "$WFC" emb_wf_eval/dinov3s_konkle emb_wf_eval/dinov3s_konkle_dev
done
wait
echo "B done: $(ls runs/F_${E}_*/metrics.json | wc -l) ($(date))"
echo "=== C. temporal window +-5 s ==="
until grep -q "S_WIN5_DONE: 8/8" logs/win5_embed.log 2>/dev/null; do sleep 900; done
WCACHE="$RCACHE $(ls -d /data2/mcfrank/emb_win5/dinov3s/shard_* | tr '\n' ' ')"
for s in 0 1 2 3 4; do one win5_30000 bv26a_rand_30000_s$s $s 5 "$WCACHE" $EV $DV; done
for s in 0 1 2; do
  one win5_300000  bv26a_rand_300000_s$s  $s 5 "$WCACHE" $EV $DV
  one win5_1000000 bv26a_rand_1000000_s$s $s 5 "$WCACHE" $EV $DV
  one win5_full    bv26a_base             $s 5 "$WCACHE" $EV $DV
done
wait
echo "S_RUNS_DONE: $(ls runs/F_${E}_*/metrics.json | wc -l)/113 ($(date))"
