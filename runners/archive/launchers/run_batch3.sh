#!/bin/bash
# Batch 3 — cross-situational prototype (E9) region-MIL bootstrap. Dynamic GPUs.
set -u
W=/data2/mcfrank/vlm-headcam; PY=/data2/mcfrank/ladder/condaenv/bin/python
EV=$W/manifests/eval_frames.parquet; RC=$W/emb_reg; mkdir -p $W/runs
JOBS=(
"R3_proto_across140 --proto boot_across_140000"
"R3_proto_within110 --proto boot_within_110000"
"R3_protolang_across140 --proto|--lang-prior boot_across_140000"
"R3_protolang_within110 --proto|--lang-prior boot_within_110000"
"R3_proto_across60 --proto boot_across_60000"
)
mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk '$2<8000{print $1}')
NG=${#GPUS[@]}; [ "$NG" -eq 0 ] && GPUS=(0 1 2 3) && NG=4
echo "[b3] GPUs: ${GPUS[*]}"
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
touch $W/runs/BATCH3_DONE
echo "[b3] done"
