#!/bin/bash
cd /data2/mcfrank/vlm-headcam
export CUDA_VISIBLE_DEVICES=7
PY=/data2/mcfrank/ladder/condaenv/bin/python
# wait for region embedding to finish (index jumps to full count at completion)
until [ "$($PY -c "import pandas as pd;print(len(pd.read_parquet(chr(39)+chr(101)+chr(39).join([chr(39),chr(109),chr(98)]) if False else \"emb_reg/index.parquet\")))" 2>/dev/null)" -ge 1080000 ]; do sleep 60; done
echo "emb_reg complete: $($PY -c "import pandas as pd;print(len(pd.read_parquet(\"emb_reg/index.parquet\")))")"
rmil () { $PY -B src/train_region_mil.py --manifest manifests/$1.parquet --region-cache emb_reg --out runs/$2 --epochs 20 --mode plain 2>&1 | tail -1; echo "  ^ $2"; }
rmil grid_baseline_train  G_base_mil_s0
for thr in 50 70 80 90 100; do rmil grid_t1_ge${thr}_train G_t1_ge${thr}_mil_s0; done
rmil grid_t2_labels_train G_t2_mil_s0
echo "=== GRID_MIL eval (test-60 / dev-117) ==="
for run in G_base_mil_s0 G_t1_ge50_mil_s0 G_t1_ge70_mil_s0 G_t1_ge80_mil_s0 G_t1_ge90_mil_s0 G_t1_ge100_mil_s0 G_t2_mil_s0; do
  echo -n "$run  test60: "; $PY -B src/eval_model.py $run manifests/eval_frames_konkle.parquet emb_konkle 2>/dev/null | grep -oE "4AFC=[0-9.]+ .*"
  echo -n "$run  dev117: "; $PY -B src/eval_model.py $run manifests/eval_frames_konkle_dev.parquet emb_konkle_dev 2>/dev/null | grep -oE "4AFC=[0-9.]+ .*"
done
echo "GRID_MIL_DONE"
