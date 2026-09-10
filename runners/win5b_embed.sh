#!/bin/bash
# Round-2 neighbor embeddings (residual frames for FULL-corpus +-5 s windows), all four
# encoders, 2 shards each, on GPUs 6-7 only (0-5 belong to the L-BV DINO trainer).
# Per GPU: the L-OTS shard (long pole) runs alongside a sequential chain of the three
# lighter encoders. Embedders are partly I/O-bound, so two per GPU is a net win.
# Join watcher writes "WIN5B_EMBED_DONE: 8/8" to logs/win5_embed.log; run_window2.sh gates on it.
set -u
cd /data2/mcfrank/vlm-headcam
export HF_HOME=/ccn2/u/khaiaw/.cache/huggingface HF_HUB_OFFLINE=1
export BABYVIEW_ROOT=/ccn2b/dataset/babyview/2026.1 BABYVIEW_FRAMES=/ccn2b/dataset/babyview/2026.1/extracted_frames_1fps
P=/ccn2/u/khaiaw/miniconda3/envs/ccwm/bin/python
M=manifests/bv26a_win5_frames2.parquet
OUT=/data2/mcfrank/emb_win5b
GPUS=(6 7)
[ -f "$M" ] || { echo "MISSING $M"; exit 1; }
for i in 0 1; do
  g=${GPUS[$i]}
  CUDA_VISIBLE_DEVICES=$g setsid nohup $P -B src/embed_regions.py --frames $M --out $OUT/dinov3l/shard_$i \
    --model facebook/dinov3-vitl16-pretrain-lvd1689m --grid 4 --drop-cls --shard $i --nshards 2 \
    > logs/win5bemb_dinov3l_$i.log 2>&1 < /dev/null &
  ( CUDA_VISIBLE_DEVICES=$g $P -B /data2/mcfrank/dinov3/embed_native_dino.py --run /data2/mcfrank/dino_s3_vitb --ckpt 199999 \
      --manifest $M --out $OUT/vitb_bv/shard_$i --drop-cls --shard $i --nshards 2 > logs/win5bemb_vitb_bv_$i.log 2>&1
    CUDA_VISIBLE_DEVICES=$g $P -B /data2/mcfrank/dinov3/embed_native_dino.py --run /data2/mcfrank/dino_s2_vits --ckpt 199999 \
      --manifest $M --out $OUT/vits_bv/shard_$i --drop-cls --shard $i --nshards 2 > logs/win5bemb_vits_bv_$i.log 2>&1
    CUDA_VISIBLE_DEVICES=$g $P -B src/embed_regions.py --frames $M --out $OUT/dinov3b/shard_$i \
      --model facebook/dinov3-vitb16-pretrain-lvd1689m --grid 4 --drop-cls --shard $i --nshards 2 > logs/win5bemb_dinov3b_$i.log 2>&1
  ) > logs/win5bemb_chain_$i.log 2>&1 < /dev/null &
  disown -a
done
echo "=== win5b: 2 x (dinov3l + chain vitb_bv->vits_bv->dinov3b) on GPUs ${GPUS[*]} $(date) ===" >> logs/win5_embed.log
cat > /data2/mcfrank/win5b_join.sh <<'EOJ'
#!/bin/bash
cd /data2/mcfrank/vlm-headcam
until [ "$(ls /data2/mcfrank/emb_win5b/*/shard_*/index.parquet 2>/dev/null | wc -l)" -ge 8 ]; do sleep 600; done
echo "WIN5B_EMBED_DONE: 8/8 (all encoders, 2 shards each) $(date)" >> logs/win5_embed.log
EOJ
setsid nohup bash /data2/mcfrank/win5b_join.sh > /dev/null 2>&1 < /dev/null &
disown -a
echo "launched; procs: $(pgrep -f "emb_win5b" | wc -l)"
