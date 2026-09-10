#!/bin/bash
# BUNDLE 1 — DATA for 2026.1. Nothing here depends on an analysis decision, so it is safe to run
# unattended. The STUDIES bundle (ladder / scaling / diversity) is deliberately NOT chained: we
# look at the first results before committing compute to the rest.
#
#   setsid bash -c "bash run_2026_data.sh > /data2/mcfrank/data2026.log 2>&1" < /dev/null &
set -u
cd /data2/mcfrank/vlm-headcam
export PYTHONPATH=src
PY=/data2/mcfrank/ladder/condaenv/bin/python
# DINOv3 is gated; embed in khaiaw's env against his cache (see run_phase5.sh)
EMBPY=/ccn2/u/khaiaw/miniconda3/envs/ccwm/bin/python
export HF_HOME=/ccn2/u/khaiaw/.cache/huggingface HF_HUB_OFFLINE=1
export BABYVIEW_ROOT=/ccn2b/dataset/babyview/2026.1
FR=$BABYVIEW_ROOT/extracted_frames_1fps
# general release artefact -> lives with the release, not in personal scratch
OUT=/ccn2b/dataset/babyview/2026.1/outputs/image_embeddings/dinov3b_grid4x4
DV3=facebook/dinov3-vitb16-pretrain-lvd1689m
NSH=8

# ---- 0. discard the register-corrupted shards from 2026-08-22 --------------------
for d in emb_dv3_2026_0 emb_dv3_2026_1; do
  [ -d "$d" ] && { echo "removing register-corrupted $d"; rm -rf "$d"; }
done

# ---- 1. DINOv3 embeddings, R=16 to match the 2025.2 readout ---------------------
mkdir -p "$OUT"
echo "=== embedding 2026.1 midpoint frames -> $OUT ==="
while :; do
  FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk '$2<2000{print $1}')
  NF=$(echo $FREE | wc -w); [ "$NF" -ge 2 ] && break
  echo "$(date +%H:%M) waiting for >=2 idle GPUs (have $NF)"; sleep 600
done
echo "GPUs: $FREE"
i=0
for g in $FREE; do
  [ "$i" -ge "$NSH" ] && break
  ( [ -f "$OUT/shard_$i/index.parquet" ] || \
    CUDA_VISIBLE_DEVICES=$g $EMBPY -B src/embed_regions.py \
      --frames manifests/bv2026_frames.parquet --out "$OUT/shard_$i" --model $DV3 \
      --grid 4 --drop-cls --shard $i --nshards $NSH > logs/emb2026_$i.log 2>&1
    echo "  shard $i done" ) &
  i=$((i+1))
done
wait
NS=$(ls -d "$OUT"/shard_*/index.parquet 2>/dev/null | wc -l)
echo "shards complete: $NS/$i"
[ "$NS" -eq "$i" ] || { echo "EMBEDDING INCOMPLETE — stopping before language"; exit 1; }
# ---- 2. language annotation for 2026.1 ------------------------------------------
echo "=== language annotation 2026.1 ==="
LANGOUT=/ccn2/dataset/babyview/annotations/language
bash ~/bv-annotations/language/run_release.sh 2026.1 \
     /ccn2a/dataset/babyview/2026.1/outputs/merged_transcripts_parsed.csv "$LANGOUT" \
     || { echo "LANGUAGE FAILED"; exit 1; }

echo "DATA_2026_DONE"
echo "next (deliberate, not chained): bash run_2026_studies.sh   # ladder + scaling + diversity"
