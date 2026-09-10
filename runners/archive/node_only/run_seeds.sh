#!/bin/bash
cd /data2/mcfrank/vlm-headcam
export CUDA_VISIBLE_DEVICES=4
PY=/data2/mcfrank/ladder/condaenv/bin/python
wf () { $PY -B src/train.py --manifest manifests/$1.parquet --emb-dir emb_full --out runs/$2 --epochs 20 --seed $3 2>&1 | tail -1; echo "  ^ $2"; }
mil () { $PY -B src/train_region_mil.py --manifest manifests/$1.parquet --region-cache emb_reg --out runs/$2 --epochs 20 --mode plain --seed $3 2>&1 | tail -1; echo "  ^ $2"; }
echo "=== whole-frame seeds ==="
for s in 1 2; do
  wf grid_baseline_train   G_base_wf_s${s}    $s
  wf grid_t1_ge50_train    G_t1_ge50_wf_s${s} $s
  wf grid_t2_labels_train  G_t2_wf_s${s}      $s
done
echo "=== region-MIL seeds ==="
for s in 1 2; do
  mil grid_baseline_train  G_base_mil_s${s}    $s
  mil grid_t1_ge50_train   G_t1_ge50_mil_s${s} $s
  mil grid_t2_labels_train G_t2_mil_s${s}      $s
done
# whole-frame dev eval needs a CLS slice of the dev region cache
$PY -c "import numpy as np,os,shutil; os.makedirs(chr(101)+chr(109)+chr(98)+chr(95)+chr(107)+chr(111)+chr(110)+chr(107)+chr(108)+chr(101)+chr(95)+chr(100)+chr(101)+chr(118)+chr(95)+chr(99)+chr(108)+chr(115),exist_ok=True)"
$PY -c "import numpy as np,os,shutil; d=\"emb_konkle_dev_cls\"; os.makedirs(d,exist_ok=True); r=np.load(\"emb_konkle_dev/emb.f16.npy\"); np.save(d+\"/emb.f16.npy\", r[:,0,:]); shutil.copy(\"emb_konkle_dev/index.parquet\", d+\"/index.parquet\"); print(\"built\", d, r[:,0,:].shape)"
echo SEEDS_TRAIN_DONE
