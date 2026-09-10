#!/bin/bash
cd /data2/mcfrank/vlm-headcam
export CUDA_VISIBLE_DEVICES=7
PY=/data2/mcfrank/ladder/condaenv/bin/python
t () { $PY -B src/train.py --manifest manifests/$1.parquet --emb-dir emb_full --out runs/$2 --epochs 20 --seed ${3:-0} 2>&1 | tail -1; echo "  ^ $2"; }
echo "=== no-MIL baseline ==="
t grid_baseline_train      G_base_wf_s0
echo "=== topline1 (Gemini filter sweep) ==="
for thr in 50 70 80 90 100; do t grid_t1_ge${thr}_train G_t1_ge${thr}_wf_s0; done
echo "=== topline2 (Gemini labels) ==="
t grid_t2_labels_train     G_t2_wf_s0
echo "GRID_WF_DONE"
