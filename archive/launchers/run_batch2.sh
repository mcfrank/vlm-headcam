#!/bin/bash
# Batch 2 — region-MIL bootstrap variants that attack the E-step signal:
#   lang (E2), distinct/base-rate (E8), distinct+lang. Dynamic GPUs. Needs emb_reg (Batch 1).
set -u
W=/data2/mcfrank/vlm-headcam
PY=/data2/mcfrank/ladder/condaenv/bin/python
EV=$W/manifests/eval_frames.parquet
RC=$W/emb_reg
mkdir -p $W/runs

# name variant-flags manifest
JOBS=(
"R2_lang_across140 --lang-prior boot_across_140000"
"R2_distinct_across140 --score-mode|distinct boot_across_140000"
"R2_distlang_across140 --score-mode|distinct|--lang-prior boot_across_140000"
"R2_lang_within110 --lang-prior boot_within_110000"
"R2_distinct_within110 --score-mode|distinct boot_within_110000"
"R2_distlang_within110 --score-mode|distinct|--lang-prior boot_within_110000"
"R2_distlang_across60 --score-mode|distinct|--lang-prior boot_across_60000"
)
mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk '$2<8000{print $1}')
NG=${#GPUS[@]}; [ "$NG" -eq 0 ] && GPUS=(0 1 2 3) && NG=4
echo "[b2] GPUs: ${GPUS[*]}"

for gi in "${!GPUS[@]}"; do
 (
  for k in "${!JOBS[@]}"; do
    [ $((k % NG)) -ne "$gi" ] && continue
    set -- ${JOBS[$k]}; name=$1; flags=$(echo $2 | tr '|' ' '); man=$3
    out=$W/runs/$name; [ -f $out/DONE ] && continue
    env CUDA_VISIBLE_DEVICES=${GPUS[$gi]} PYTHONDONTWRITEBYTECODE=1 $PY -B $W/src/train_region_mil.py \
      --manifest $W/manifests/$man.parquet --region-cache $RC --eval-frames $EV --out $out \
      --mode boot --warmup 5 --rounds 6 --epochs-per-round 2 $flags > $out.log 2>&1 \
      && touch $out/DONE || touch $out/FAILED
  done
 ) &
done
wait
touch $W/runs/BATCH2_DONE
echo "[b2] done"
