#!/bin/bash
# Per-GPU sequential-queue dispatcher for A/B/C training runs.
# Usage (on ccn2, detached): bash dispatch.sh "1 2 3 4"   # space-sep GPU ids
set -u
GPUS=($1)
NG=${#GPUS[@]}
W=/data2/mcfrank/vlm-headcam
PY=/data2/mcfrank/ladder/condaenv/bin/python
JOBS=$W/manifests/jobs.tsv
mkdir -p $W/runs
mapfile -t LINES < "$JOBS"

for gi in "${!GPUS[@]}"; do
  G=${GPUS[$gi]}
  (
    for li in "${!LINES[@]}"; do
      [ $(( li % NG )) -ne $gi ] && continue
      IFS=$'\t' read -r name man ev seed <<< "${LINES[$li]}"
      out=$W/runs/$name
      [ -f "$out/DONE" ] && continue
      env CUDA_VISIBLE_DEVICES=$G PYTHONDONTWRITEBYTECODE=1 $PY -B $W/src/train.py \
        --manifest "$man" --eval-frames "$ev" --out "$out" --emb-dir $W/emb_full \
        --epochs 20 --batch 256 --lr 3e-4 --min-freq 5 --seed "$seed" \
        > "$out.log" 2>&1 && touch "$out/DONE" || touch "$out/FAILED"
    done
  ) &
done
wait
touch $W/runs/ABC_ALL_DONE
echo done
