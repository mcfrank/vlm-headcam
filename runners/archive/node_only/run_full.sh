#!/bin/bash
cd /data2/mcfrank/vlm-headcam
export PYTHONPATH=src
PY=/data2/mcfrank/ladder/condaenv/bin/python
until [ $(for i in 0 1 2 3 4 5 6 7; do [ -f emb_reg_ho_$i/index.parquet ] && echo x; done | wc -l) -eq 8 ]; do sleep 20; done
C="emb_reg emb_reg_ho_0 emb_reg_ho_1 emb_reg_ho_2 emb_reg_ho_3 emb_reg_ho_4 emb_reg_ho_5 emb_reg_ho_6 emb_reg_ho_7"
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$s $PY -B src/train_frame_mil.py --manifest manifests/grid_baseline_full.parquet --caches $C --eval-cache emb_konkle --eval-frames manifests/eval_frames_konkle.parquet --window 0 --seed $s --out runs/G_base_mil_full_s$s > logs/full_s$s.log 2>&1 &
done
wait
echo "=== region-MIL 100% (all 1.14M pairs) ==="
for s in 0 1 2; do echo "full_s$s $(grep -oE \"best [0-9.]+\" logs/full_s$s.log | tail -1)"; done
echo FULL_DONE
