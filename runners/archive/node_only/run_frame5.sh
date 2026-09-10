#!/bin/bash
cd /data2/mcfrank/vlm-headcam
export PYTHONPATH=src
PY=/data2/mcfrank/ladder/condaenv/bin/python
until [ $(for i in 0 1 2 3 4 5 6 7; do [ -f emb_win5_$i/index.parquet ] && echo x; done | wc -l) -eq 8 ]; do sleep 30; done
C="emb_reg emb_win_0 emb_win_1 emb_win_2 emb_win_3 emb_win5_0 emb_win5_1 emb_win5_2 emb_win5_3 emb_win5_4 emb_win5_5 emb_win5_6 emb_win5_7"
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$s $PY -B src/train_frame_mil.py --manifest manifests/grid_baseline_train.parquet --caches $C --eval-cache emb_konkle --eval-frames manifests/eval_frames_konkle.parquet --window 5 --seed $s --out runs/G_framereg5_s$s > logs/framereg5_s$s.log 2>&1 &
done
wait
echo "=== pm5s frame-MIL (region+frame, +-5s) ==="
for s in 0 1 2; do echo -n "framereg5_s$s "; $PY -B src/eval_model.py G_framereg5_s$s manifests/eval_frames_konkle.parquet emb_konkle 2>/dev/null | grep -oE "4AFC=[0-9.]+"; done
echo FRAME5_DONE
