#!/bin/bash
# Batch 4 — curriculum (E5): bootstrap within-kid, then transfer to across-kid.
set -u
W=/data2/mcfrank/vlm-headcam; PY=/data2/mcfrank/ladder/condaenv/bin/python
EV=$W/manifests/eval_frames.parquet; RC=$W/emb_reg; V=$W/manifests/shared_vocab.json
COMMON="--mode boot --warmup 5 --rounds 6 --epochs-per-round 2 --proto --lang-prior --region-cache $RC --eval-frames $EV --vocab-json $V"
run(){ env CUDA_VISIBLE_DEVICES=$1 PYTHONDONTWRITEBYTECODE=1 $PY -B $W/src/train_region_mil.py $COMMON --manifest $W/manifests/$2.parquet --out $W/runs/$3 ${4:-} > $W/runs/$3.log 2>&1 && touch $W/runs/$3/DONE || touch $W/runs/$3/FAILED; }
# stage1 (within) + matched from-scratch across control, in parallel
run 0 boot_within_110000 R4_stage1_within &
run 1 boot_across_140000 R4_scratch_across &
wait
# stage2: across, initialized from within-bootstrapped model
run 0 boot_across_140000 R4_stage2_across "--init-from $W/runs/R4_stage1_within/model.pt"
touch $W/runs/BATCH4_DONE
echo "[b4] done"
