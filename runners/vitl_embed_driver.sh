#!/bin/bash
# L-BV (vitl_bv) cache campaign, everything the DINO session's c9_vitl.sh does NOT cover:
#   (1) +-5 s window neighbor caches (4.92M frames): EARLY part = 2 shards on GPUs 6-7 now;
#       LATE part = 6 shards on GPUs 0-5, gated on the main frame cache finishing
#       (VITL_C9_ALL_DONE_VERIFIED in logs/c9_vitl_driver.log). -> /data2/mcfrank/emb_win5/vitl_bv
#   (2) whole-frame (no-MIL) mean caches (CPU), same gate. -> emb_wf/vitl_bv, emb_wf_eval/vitl_bv_*
# Markers (logs/win5_embed.log): VITL_WIN5_DONE: 8/8 ; (logs/wfprep_vitl.log): WF_ENC_DONE vitl_bv
set -u
cd /data2/mcfrank/vlm-headcam
export HF_HOME=/ccn2/u/khaiaw/.cache/huggingface HF_HUB_OFFLINE=1
export BABYVIEW_ROOT=/ccn2b/dataset/babyview/2026.1 BABYVIEW_FRAMES=/data2/mcfrank/frames_1fps_local
P=/ccn2/u/khaiaw/miniconda3/envs/ccwm/bin/python
R=/data2/mcfrank/dino_s4_vitl
OUT=/data2/mcfrank/emb_win5/vitl_bv
for m in manifests/bv26a_win5_vitl_early.parquet manifests/bv26a_win5_vitl_late.parquet; do
  [ -f "$m" ] || { echo "MISSING $m"; exit 1; }; done
mkdir -p $OUT
# (1a) early shards now, GPUs 6-7
for i in 0 1; do
  CUDA_VISIBLE_DEVICES=$((6 + i)) setsid nohup $P -B /data2/mcfrank/dinov3/embed_native_dino.py --run $R --ckpt 199999 \
    --manifest manifests/bv26a_win5_vitl_early.parquet --out $OUT/shard_$i --drop-cls --shard $i --nshards 2 \
    > logs/win5emb_vitl_bv_$i.log 2>&1 < /dev/null &
done
echo "=== vitl_bv window embeds: early shards 0-1 on GPUs 6-7 $(date) ===" >> logs/win5_embed.log
# (1b)+(2) gated on the main cache
cat > /data2/mcfrank/vitl_post_cache.sh <<'EOJ'
#!/bin/bash
cd /data2/mcfrank/vlm-headcam
export HF_HOME=/ccn2/u/khaiaw/.cache/huggingface HF_HUB_OFFLINE=1
export BABYVIEW_ROOT=/ccn2b/dataset/babyview/2026.1 BABYVIEW_FRAMES=/data2/mcfrank/frames_1fps_local
P=/ccn2/u/khaiaw/miniconda3/envs/ccwm/bin/python
R=/data2/mcfrank/dino_s4_vitl
OUT=/data2/mcfrank/emb_win5/vitl_bv
until grep -q "VITL_C9_ALL_DONE_VERIFIED" logs/c9_vitl_driver.log 2>/dev/null; do sleep 600; done
for i in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$i setsid nohup $P -B /data2/mcfrank/dinov3/embed_native_dino.py --run $R --ckpt 199999 \
    --manifest manifests/bv26a_win5_vitl_late.parquet --out $OUT/shard_$((i + 2)) --drop-cls --shard $i --nshards 6 \
    > logs/win5emb_vitl_bv_$((i + 2)).log 2>&1 < /dev/null &
done
echo "=== vitl_bv window embeds: late shards 2-7 on GPUs 0-5 $(date) ===" >> logs/win5_embed.log
OMP_NUM_THREADS=8 $P -B src/make_wf_caches.py vitl_bv > logs/wfprep_vitl.log 2>&1
until [ "$(ls $OUT/shard_*/index.parquet 2>/dev/null | wc -l)" -ge 8 ]; do sleep 600; done
echo "VITL_WIN5_DONE: 8/8 $(date)" >> logs/win5_embed.log
EOJ
setsid nohup bash /data2/mcfrank/vitl_post_cache.sh > logs/vitl_post_cache.log 2>&1 < /dev/null &
disown -a
sleep 3; echo "launched: $(pgrep -f "emb_win5/vitl_b[v]" | wc -l) early embedders, post-cache waiter $(pgrep -f "vitl_post_cache.s[h]" | wc -l)"
