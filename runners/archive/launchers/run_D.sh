#!/bin/bash
# Exp D: region grounding 2x2 (train in {frame,crop} x eval in {frame,crop}).
# Usage (on ccn2): bash run_D.sh <GPU>
set -u
G=${1:-2}
W=/data2/mcfrank/vlm-headcam
PY=/data2/mcfrank/ladder/condaenv/bin/python
M=$W/manifests
COM="--epochs 20 --batch 256 --lr 3e-4 --min-freq 5 --seed 0"
run(){ env CUDA_VISIBLE_DEVICES=$G PYTHONDONTWRITEBYTECODE=1 $PY -B $W/src/train.py "$@" $COM; }

# frame-train / frame-eval
run --manifest $M/region_frame_S00360001.parquet --emb-dir $W/emb_full \
    --eval-frames $M/eval_frames.parquet --out $W/runs/D_frame_frameeval
# crop-train / crop-eval
run --manifest $M/region_crop_S00360001_keyed.parquet --emb-dir $W/emb_crop_train \
    --eval-frames $M/eval_crop_S00360001_keyed.parquet --eval-emb-dir $W/emb_crop_eval \
    --out $W/runs/D_crop_cropeval
# frame-train / crop-eval
run --manifest $M/region_frame_S00360001.parquet --emb-dir $W/emb_full \
    --eval-frames $M/eval_crop_S00360001_keyed.parquet --eval-emb-dir $W/emb_crop_eval \
    --out $W/runs/D_frame_cropeval
# crop-train / frame-eval
run --manifest $M/region_crop_S00360001_keyed.parquet --emb-dir $W/emb_crop_train \
    --eval-frames $M/eval_frames.parquet --eval-emb-dir $W/emb_full \
    --out $W/runs/D_crop_frameeval
touch $W/runs/D_ALL_DONE
echo D_done
