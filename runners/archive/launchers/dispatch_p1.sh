#!/bin/bash
# P1 grid dispatcher. jobs_p1.tsv: name <TAB> type(boot|plain) <TAB> manifest
# Usage (on ccn2, detached): bash dispatch_p1.sh "0 1 2 3 4 5 6 7"
set -u
GPUS=($1); NG=${#GPUS[@]}
W=/data2/mcfrank/vlm-headcam
PY=/data2/mcfrank/ladder/condaenv/bin/python
EMB=$W/emb_full
EV=$W/manifests/eval_frames.parquet
mapfile -t LINES < "$W/manifests/jobs_p1.tsv"

for gi in "${!GPUS[@]}"; do
  G=${GPUS[$gi]}
  (
    for li in "${!LINES[@]}"; do
      [ $(( li % NG )) -ne $gi ] && continue
      IFS=$'\t' read -r name type man <<< "${LINES[$li]}"
      out=$W/runs/$name
      [ -f "$out/DONE" ] && continue
      if [ "$type" = "boot" ]; then
        env CUDA_VISIBLE_DEVICES=$G PYTHONDONTWRITEBYTECODE=1 $PY -B $W/src/train_boot.py \
          --manifest "$W/manifests/$man" --emb-dir $EMB --eval-frames $EV --out "$out" \
          --warmup 5 --rounds 6 --epochs-per-round 2 > "$out.log" 2>&1 && touch "$out/DONE" || touch "$out/FAILED"
      else
        env CUDA_VISIBLE_DEVICES=$G PYTHONDONTWRITEBYTECODE=1 $PY -B $W/src/train.py \
          --manifest "$W/manifests/$man" --emb-dir $EMB --eval-frames $EV --out "$out" \
          --epochs 20 > "$out.log" 2>&1 && touch "$out/DONE" || touch "$out/FAILED"
      fi
    done
  ) &
done
wait
touch $W/runs/P1_ALL_DONE
echo P1_done
