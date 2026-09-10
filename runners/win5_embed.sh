#!/bin/bash
# Neighbor-frame embeddings for the +-5 s window control: 3,425,375 frames x 4 encoders.
# Frames read from the ccn2b release (the /data2 mirror is sparse); caches to /data2.
# OTS via embed_regions.py, BV via the DINO session's native embedder; 2 shards/encoder,
# encoders sequential (only ~2 GPUs free while L-BV trains).
set -u
cd /data2/mcfrank/vlm-headcam
export HF_HOME=/ccn2/u/khaiaw/.cache/huggingface HF_HUB_OFFLINE=1
export BABYVIEW_ROOT=/ccn2b/dataset/babyview/2026.1 BABYVIEW_FRAMES=/ccn2b/dataset/babyview/2026.1/extracted_frames_1fps
P=/ccn2/u/khaiaw/miniconda3/envs/ccwm/bin/python
OUT=/data2/mcfrank/emb_win5; mkdir -p $OUT
M=manifests/bv26a_win5_frames.parquet
pick2 () { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F", " '$2<20000{print $1}' | head -2 | tr '\n' ' '; }
run_ots () {  # <enc> <hf model>
  local G=($(pick2)); [ ${#G[@]} -lt 2 ] && G=(${G[0]} ${G[0]})
  for i in 0 1; do
    CUDA_VISIBLE_DEVICES=${G[$i]} $P -B src/embed_regions.py --frames $M --out $OUT/$1/shard_$i \
      --model $2 --grid 4 --drop-cls --shard $i --nshards 2 > logs/win5emb_$1_$i.log 2>&1 &
  done; wait
}
run_bv () {   # <enc> <run dir>
  local G=($(pick2)); [ ${#G[@]} -lt 2 ] && G=(${G[0]} ${G[0]})
  for i in 0 1; do
    CUDA_VISIBLE_DEVICES=${G[$i]} $P -B /data2/mcfrank/dinov3/embed_native_dino.py --run $2 --ckpt 199999 \
      --manifest $M --out $OUT/$1/shard_$i --drop-cls --grid 4 --shard $i --nshards 2 > logs/win5emb_$1_$i.log 2>&1 &
  done; wait
}
echo "=== dinov3b $(date) ===";  run_ots dinov3b facebook/dinov3-vitb16-pretrain-lvd1689m
echo "=== vits_bv $(date) ===";  run_bv  vits_bv /data2/mcfrank/dino_s2_vits
echo "=== vitb_bv $(date) ===";  run_bv  vitb_bv /data2/mcfrank/dino_s3_vitb
echo "=== dinov3l $(date) ===";  run_ots dinov3l facebook/dinov3-vitl16-pretrain-lvd1689m
n=$(ls $OUT/*/shard_*/index.parquet 2>/dev/null | wc -l)
echo "WIN5_EMBED_DONE: $n/8 shard indexes ($(date))"
