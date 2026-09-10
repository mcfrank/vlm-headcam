#!/bin/bash
# S-OTS (dinov3s = facebook/dinov3-vits16-pretrain-lvd1689m, 22M params) cache campaign, queued
# BEHIND the L-BV runs (gate: VITL_RUNS_DONE in logs/run_vitl.log). Same recipe as L-OTS:
#   eval caches  embed_konkle.py (R=17, CLS+grid): Konkle test / dev / LEVANTE  -> emb_ch8_eval/dinov3s_*, emb_lev_dinov3s
#   main cache   embed_regions.py --drop-cls, bv26_frames_all, 8 shards        -> $EMB/dinov3s_grid4x4/shard_0-7
#   wf caches    make_wf_caches.py dinov3s (CPU)                                -> emb_wf/dinov3s, emb_wf_eval/dinov3s_*
#   window       embed_regions.py --drop-cls, all 4.92M neighbor frames, 8 shards -> /data2/mcfrank/emb_win5/dinov3s/shard_0-7
# Markers: S_MAIN_DONE (logs/dinov3s_driver.log), WF_ENC_DONE dinov3s (logs/wfprep_dinov3s.log),
#          S_WIN5_DONE: 8/8 (logs/win5_embed.log). run_dinov3s.sh gates on these.
set -u
cd /data2/mcfrank/vlm-headcam
export HF_HOME=/ccn2/u/khaiaw/.cache/huggingface HF_HUB_OFFLINE=1
export BABYVIEW_ROOT=/ccn2b/dataset/babyview/2026.1 BABYVIEW_FRAMES=/data2/mcfrank/frames_1fps_local
P=/ccn2/u/khaiaw/miniconda3/envs/ccwm/bin/python
M=facebook/dinov3-vits16-pretrain-lvd1689m
EMB=/ccn2b/dataset/babyview/2026.1/outputs/image_embeddings
MAIN=$EMB/dinov3s_grid4x4
W5=/data2/mcfrank/emb_win5/dinov3s
LOG=logs/dinov3s_driver.log
until grep -q "VITL_RUNS_DONE" logs/run_vitl.log 2>/dev/null; do sleep 900; done
echo "=== S-OTS campaign start $(date) ===" >> $LOG
# combined neighbor list (early ∪ late = round-1 ∪ round-2, 4.92M frames)
$P - <<'EOF'
import pandas as pd
w = pd.concat([pd.read_parquet("manifests/bv26a_win5_vitl_early.parquet"), pd.read_parquet("manifests/bv26a_win5_vitl_late.parquet")], ignore_index=True)
w.to_parquet("manifests/bv26a_win5_all.parquet", index=False); print("neighbor frames", len(w))
EOF
# eval caches (sequential, GPU 0)
CUDA_VISIBLE_DEVICES=0 $P -B src/embed_konkle.py --manifest manifests/konkle_manifest.parquet --model $M --grid 4 --out emb_ch8_eval/dinov3s_konkle > logs/s_konkle.log 2>&1 || { echo "S_FAIL konkle" >> $LOG; exit 1; }
CUDA_VISIBLE_DEVICES=0 $P -B src/embed_konkle.py --manifest manifests/eval_frames_konkle_dev.parquet --model $M --grid 4 --out emb_ch8_eval/dinov3s_konkle_dev > logs/s_konkle_dev.log 2>&1 || { echo "S_FAIL konkle_dev" >> $LOG; exit 1; }
CUDA_VISIBLE_DEVICES=0 $P -B src/embed_konkle.py --manifest manifests/lev_vocab_manifest.parquet --model $M --grid 4 --out emb_lev_dinov3s > logs/s_lev.log 2>&1 || { echo "S_FAIL lev" >> $LOG; exit 1; }
echo "S eval caches done $(date)" >> $LOG
# main frame cache, 8 shards
mkdir -p $MAIN
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i $P -B src/embed_regions.py --frames manifests/bv26_frames_all.parquet --out $MAIN/shard_$i \
    --model $M --grid 4 --drop-cls --shard $i --nshards 8 > logs/s_main_$i.log 2>&1 &
done
wait
n=$(ls $MAIN/shard_*/index.parquet 2>/dev/null | wc -l); [ "$n" -eq 8 ] || { echo "S_FAIL main cache $n/8" >> $LOG; exit 1; }
echo "S_MAIN_DONE ($(date))" >> $LOG
# wf caches (CPU) in the background; window embeds on all 8 GPUs
OMP_NUM_THREADS=8 setsid nohup $P -B src/make_wf_caches.py dinov3s > logs/wfprep_dinov3s.log 2>&1 < /dev/null &
mkdir -p $W5
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i $P -B src/embed_regions.py --frames manifests/bv26a_win5_all.parquet --out $W5/shard_$i \
    --model $M --grid 4 --drop-cls --shard $i --nshards 8 > logs/win5emb_dinov3s_$i.log 2>&1 &
done
wait
n=$(ls $W5/shard_*/index.parquet 2>/dev/null | wc -l); [ "$n" -eq 8 ] || { echo "S_FAIL window $n/8" >> $LOG; exit 1; }
echo "S_WIN5_DONE: 8/8 $(date)" >> logs/win5_embed.log
echo "S caches all done ($(date))" >> $LOG
