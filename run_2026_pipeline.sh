#!/bin/bash
# 2026.1 pipeline: frames -> (Gemini annotations + DINOv3 embeddings) -> scaling runs.
# The core experiments stay on 2025.2 (run_phase5.sh); this adds the scale/diversity extension.
#
#   setsid bash -c "bash run_2026_pipeline.sh > /data2/mcfrank/bv2026.log 2>&1" < /dev/null &
#
# Every stage is resumable and skips itself if its output exists, so it is safe to re-launch.
set -u
cd /data2/mcfrank/vlm-headcam
export PYTHONPATH=src
PY=/data2/mcfrank/ladder/condaenv/bin/python
# Vertex credentials for the Gemini annotation step (service account; never echoed, never in git)
if [ -f "$HOME/.secrets/vlm-headcam.env" ]; then
  set -a; . "$HOME/.secrets/vlm-headcam.env"; set +a
  export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/.secrets/vlm-headcam-sa.json}"
else
  echo "WARNING: no ~/.secrets/vlm-headcam.env — the Gemini stage will fail"
fi
# DINOv3 is a GATED HuggingFace repo and this node has no HF auth. Khai has accepted the
# licence and his cache holds the weights, and his `ccwm` env has transformers 4.57 (ours is 4.49,
# which predates the dinov3 model type). So embedding runs in HIS env against HIS cache, offline;
# training stays in our env. Same precedent as the pose pipeline symlinking his model weights.
EMBPY=/ccn2/u/khaiaw/miniconda3/envs/ccwm/bin/python
export HF_HOME=/ccn2/u/khaiaw/.cache/huggingface HF_HUB_OFFLINE=1
PREFIX=manifests/bv2026
N_VIDEOS_EXPECTED=16008          # from the 2026.1 transcript

# ---- 0. locate the frames and WAIT for extraction to finish -------------------------
find_root () {
  for d in /ccn2b/dataset/babyview/2026.1/extracted_frames_1fps \
           /ccn2b/dataset/babyview/_mcfrank_2026.1_staging/extracted_frames_1fps \
           /ccn2a/dataset/babyview/2026.1/extracted_frames_1fps; do
    [ -d "$d" ] && { echo "$d"; return; }
  done
}
FR=$(find_root); [ -n "$FR" ] || { echo "no 2026.1 frame dir yet — rerun once extraction starts"; exit 1; }
echo "frames: $FR"
# settle: stop waiting when the dir count stops growing for 3 consecutive 10-min checks
stable=0
while [ $stable -lt 3 ]; do
  a=$(ls "$FR" | wc -l); sleep 600; b=$(ls "$FR" | wc -l)
  if [ "$a" -eq "$b" ]; then stable=$((stable+1)); else stable=0; fi
  echo "  $(date +%H:%M) dirs=$b (expected ~$N_VIDEOS_EXPECTED) stable=$stable"
  # if the frames moved to their permanent home mid-wait, follow them
  NEW=$(find_root); [ "$NEW" = "$FR" ] || { FR=$NEW; echo "  frames moved -> $FR"; stable=0; }
done
echo "extraction settled at $(ls "$FR" | wc -l) video dirs"

# ---- 1. pairs manifest (midpoint pairing, same convention as 2025.2) -----------------
if [ ! -f ${PREFIX}_pairs.parquet ]; then
  $PY -B src/build_pairs_2026.py --frames-root "$FR" --out-prefix $PREFIX --require-frame \
      > logs/bv2026_pairs.log 2>&1 || { echo "PAIRS FAILED"; exit 1; }
fi
grep -E "pairs over|children" logs/bv2026_pairs.log

# ---- 2a. Gemini referential annotation (network-bound; runs alongside the GPU work) ---
#      thinking is OFF by default (thinking_budget=0) — the cheap path.
# gemini_align skips already-scored keys from its JSONL checkpoint, so re-running is cheap and
# resumes. A parquet on disk may be PARTIAL (2026-08-23: 55.8% scored), so never skip on its
# existence alone.
( if true; then
    BABYVIEW_FRAMES="$FR" $PY -B src/gemini_align.py --manifest ${PREFIX}_pairs.parquet \
       --out scored/bv2026_gemini.parquet --workers 48 > logs/bv2026_gemini.log 2>&1 \
      || { echo "GEMINI FAILED (see logs/bv2026_gemini.log)"; exit 1; }
  fi
  echo "GEMINI_DONE" ) &
GEM=$!

# ---- 2b. DINOv3-B region embeddings, only the midpoint frames -------------------------
DV3=facebook/dinov3-vitb16-pretrain-lvd1689m
if [ ! -f emb_dv3_2026_0/index.parquet ]; then
  # shard across whatever GPUs are idle (pose/phase5 may be using some)
  # wait for idle GPUs rather than colliding with pose / run_phase5
  while :; do
    FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk '$2<2000{print $1}')
    NF=$(echo $FREE | wc -w); [ "$NF" -ge 2 ] && break
    echo "  $(date +%H:%M) waiting for >=2 idle GPUs (have $NF)"; sleep 600
  done
  echo "embedding on GPUs: $FREE"
  i=0
  for g in $FREE; do
    BABYVIEW_FRAMES="$FR" CUDA_VISIBLE_DEVICES=$g PYTHONPATH=src $EMBPY -B src/embed_regions.py \
      --frames ${PREFIX}_frames.parquet --out emb_dv3_2026_$i --model $DV3 --grid 4 \
      --shard $i --nshards $NF > logs/bv2026_emb_$i.log 2>&1 &
    i=$((i+1))
  done
  wait
  echo "EMBED_DONE"
fi
wait $GEM

# ---- 3. scaling + diversity on 2026.1 --------------------------------------------------
# 51 children (vs 36) is the reason to do this: the diversity result is the one that gains.
CACHES=$(ls -d emb_dv3_2026_* 2>/dev/null | tr '\n' ' ')
EVAL="--eval-cache emb_enc_eval/dinov3b_ots_konkle --eval-frames manifests/eval_frames_konkle.parquet"
DEV="--dev-cache emb_dv3_konkle_dev --dev-frames manifests/eval_frames_konkle_dev.parquet"
# sizes go further than 2025.2 (1.84M pairs vs 1.14M) and diversity goes to 51 children (vs 36)
$PY -B src/build_scaling_manifests.py \
    --scored scored/bv2026_gemini.parquet --pairs ${PREFIX}_pairs.parquet --prefix bv2026 \
    --sizes 10000,30000,100000,300000,1000000 --aligned-sizes 10000,30000,85000,150000 \
    --kids 1,3,10,25,51 > logs/bv2026_scalemans.log 2>&1 || { echo "SCALING MANIFESTS FAILED"; exit 1; }
i=0
for man in $(ls manifests/bv2026_{rand,align,div}_*.parquet manifests/bv2026_{bigchild,poolbig}.parquet 2>/dev/null | xargs -n1 basename | sed 's/.parquet//'); do
  for s in 0 1 2; do
    g=$((i % 8)); i=$((i+1))
    CUDA_VISIBLE_DEVICES=$g $PY -B src/train_frame_mil.py --window 0 \
      --manifest manifests/$man.parquet --caches $CACHES $EVAL $DEV \
      --seed $s --out runs/B26_${man}_s$s > logs/b26_${man}_s$s.log 2>&1 &
    [ $((i % 6)) -eq 0 ] && wait
  done
done
wait
$PY src/scrape_runs.py
echo "BV2026_PIPELINE_DONE"
