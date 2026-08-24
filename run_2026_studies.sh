#!/bin/bash
# BUNDLE 2 — the three studies we actually want to nail, on 2026.1:
#   ladder · scaling · diversity        (cues and item analysis are deliberately parked)
#
# NOT chained to bundle 1 on purpose. Run it only after looking at the data bundle's output —
# in particular the supplemental English table, because the video-level filter's threshold
# determines every manifest below, and getting it wrong means redoing all of these.
#
#   bash run_2026_studies.sh            # after: bash run_2026_data.sh has printed DATA_2026_DONE
set -u
cd /data2/mcfrank/vlm-headcam
export PYTHONPATH=src
PY=/data2/mcfrank/ladder/condaenv/bin/python
EMB=/ccn2b/dataset/babyview/2026.1/outputs/image_embeddings/dinov3b_grid4x4
LANG=/ccn2/dataset/babyview/annotations/language/lang_2026.1.parquet

# ---- 0. refuse to run on incomplete data ----------------------------------------
NS=$(ls -d $EMB/shard_*/index.parquet 2>/dev/null | wc -l)
[ "$NS" -ge 1 ] || { echo "ABORT: no DINOv3 shards at $EMB — run run_2026_data.sh first"; exit 1; }
[ -f "$LANG" ] || { echo "ABORT: no language annotation at $LANG"; exit 1; }
CACHES=$(ls -d $EMB/shard_* | tr '\n' ' ')
echo "caches: $NS shards | language: $LANG"

# ---- 1. English filter (PRIMARY per Mike: do not train on non-English) -----------
$PY ~/bv-annotations/language/build_english_filter.py \
    --lang "$LANG" --pairs manifests/bv2026_pairs.parquet \
    --out manifests/bv26_en_pairs.parquet \
    --report /ccn2/dataset/babyview/annotations/language/video_decisions_2026.1.csv \
  || { echo "ENGLISH FILTER FAILED"; exit 1; }
$PY ~/bv-annotations/language/supplemental_table.py --lang "$LANG" \
    --out /ccn2/dataset/babyview/annotations/language/supp_english_by_child_2026.1.csv

# ---- 2. manifests: English-filtered (primary) and unfiltered (robustness) --------
$PY src/build_ladder_manifests.py --scored scored/bv2026_gemini.parquet --prefix bv26en \
    --english-filter manifests/bv26_en_pairs.parquet \
    --sizes 10000,30000,100000,300000,1000000 --kids 1,3,10,25,51 || exit 1
$PY src/build_ladder_manifests.py --scored scored/bv2026_gemini.parquet --prefix bv26 \
    --sizes 10000,30000,100000,300000,1000000 --kids 1,3,10,25,51 || exit 1

# ---- 3. run ---------------------------------------------------------------------
EVAL="--eval-cache emb_enc_grid_eval/dinov3b_ots_konkle --eval-frames manifests/eval_frames_konkle.parquet"
DEV="--dev-cache emb_dv3_konkle_dev16 --dev-frames manifests/eval_frames_konkle_dev.parquet"
COV="--min-coverage 0.9"
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1<2000'|wc -l)" -lt 3 ]; do
  echo "$(date +%H:%M) waiting for 3 idle GPUs"; sleep 600; done
FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk '$2<2000{print $1}' | tr '\n' ' ')
echo "GPUs: $FREE"
GPUS=($FREE); NG=${#GPUS[@]}; i=0

run () {  # run <tag> <manifest>
  local tag=$1 man=$2
  for s in 0 1 2; do
    local g=${GPUS[$((i % NG))]}; i=$((i+1))
    CUDA_VISIBLE_DEVICES=$g $PY -B src/train_frame_mil.py --window 0 \
      --manifest manifests/$man.parquet --caches $CACHES $EVAL $DEV $COV \
      --seed $s --out runs/B26_${tag}_s$s > logs/b26_${tag}_s$s.log 2>&1 &
    [ $((i % NG)) -eq 0 ] && wait
  done
}

for PFX in bv26en bv26; do              # English-filtered first: it is the primary arm
  echo "=== $PFX: ladder ==="
  for r in base filtnat t15 t2; do run ${PFX}_lad_$r ${PFX}_$r; done; wait
  echo "=== $PFX: scaling ==="
  for N in 10000 30000 100000 300000 1000000; do
    [ -f manifests/${PFX}_rand_$N.parquet ] && run ${PFX}_rand_$N ${PFX}_rand_$N
  done; wait
  echo "=== $PFX: diversity ==="
  for k in 1 3 10 25 51; do
    [ -f manifests/${PFX}_div_${k}c.parquet ] && run ${PFX}_div_${k}c ${PFX}_div_${k}c
  done; wait
done

n=$(ls -d runs/B26_* 2>/dev/null | wc -l); m=$(ls runs/B26_*/metrics.json 2>/dev/null | wc -l)
echo "B26 runs: $n dirs, $m with metrics.json"
$PY src/scrape_runs.py
echo "STUDIES_2026_DONE"
