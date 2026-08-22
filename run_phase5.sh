#!/bin/bash
# PHASE 5 re-runs: rebuild every headline number on the DINOv3-B clean rig, with dev-split
# epoch selection and per-run metrics.json. Queued 2026-08-21; the node is busy with pose
# detection, so this waits for free GPUs before starting.
#
#   setsid bash -c "bash run_phase5.sh > /data2/mcfrank/phase5.log 2>&1" < /dev/null &
#
# Everything writes metrics.json, so results/ rebuilds by scraping run dirs — no stdout archaeology.
set -u
cd /data2/mcfrank/vlm-headcam
export PYTHONPATH=src
PY=/data2/mcfrank/ladder/condaenv/bin/python
# DINOv3 is a GATED HuggingFace repo and this node has no HF auth. Khai has accepted the
# licence and his cache holds the weights, and his `ccwm` env has transformers 4.57 (ours is 4.49,
# which predates the dinov3 model type). So embedding runs in HIS env against HIS cache, offline;
# training stays in our env. Same precedent as the pose pipeline symlinking his model weights.
EMBPY=/ccn2/u/khaiaw/miniconda3/envs/ccwm/bin/python
export HF_HOME=/ccn2/u/khaiaw/.cache/huggingface HF_HUB_OFFLINE=1
OAK=oak-dtn:/oak/stanford/groups/mcfrank/vlm-headcam-archive/home-runs/vlm_enc

# ---- wait for the node (pose detection is using it) --------------------------------
need_free() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1<2000' | wc -l; }
echo "waiting for >=3 free GPUs ..."
while [ "$(need_free)" -lt 3 ]; do sleep 600; done
FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk '$2<2000{print $1}' | tr '\n' ' ')
echo "GPUs free: $FREE"; set -- $FREE; G1=$1; G2=${2:-$1}; G3=${3:-$1}

# ---- 0. eval caches: restore DINOv3-B test-60, embed dev-117 -----------------------
mkdir -p emb_enc_eval
mkdir -p emb_enc_eval emb_enc_grid_eval
[ -f emb_enc_eval/dinov3b_ots_konkle/index.parquet ] || rsync -rlt $OAK/emb_enc_eval/dinov3b_ots_konkle emb_enc_eval/
[ -f emb_enc_grid_eval/dinov3b_ots_konkle/index.parquet ] || rsync -rlt $OAK/emb_enc_grid_eval/dinov3b_ots_konkle emb_enc_grid_eval/
if [ ! -f emb_dv3_konkle_dev/index.parquet ]; then
  echo "=== embedding Konkle dev-117 with DINOv3-B ==="
  rm -rf emb_dv3_konkle_dev   # a previous failure can leave an empty dir
  CUDA_VISIBLE_DEVICES=$G1 PYTHONPATH=src $EMBPY -B src/embed_konkle.py \
     --manifest manifests/eval_frames_konkle_dev.parquet --out emb_dv3_konkle_dev \
     --model facebook/dinov3-vitb16-pretrain-lvd1689m > logs/p5_embed_dev.log 2>&1
fi

for need in emb_enc_eval/dinov3b_ots_konkle emb_enc_grid_eval/dinov3b_ots_konkle \
            emb_dv3_konkle_dev16 emb_dv3_grid_877k emb_enc/dinov3b_ots; do
  [ -f "$need/index.parquet" ] || { echo "ABORT: missing prerequisite $need (see logs/p5_*.log)"; exit 1; }
done
echo "prerequisites present"

# embed_konkle writes CLS + 4x4 (R=17); Khai's grid readout is 16 cells with no CLS. Slice the dev
# cache to match, or max-over-regions gets an extra region the projection never saw in training.
if [ ! -f emb_dv3_konkle_dev16/index.parquet ]; then
  mkdir -p emb_dv3_konkle_dev16
  $PY - <<'PYX'
import numpy as np, shutil
a = np.load("emb_dv3_konkle_dev/emb.f16.npy", mmap_mode="r")
np.save("emb_dv3_konkle_dev16/emb.f16.npy", np.asarray(a[:, 1:, :]))   # drop the CLS row
shutil.copy("emb_dv3_konkle_dev/index.parquet", "emb_dv3_konkle_dev16/index.parquet")
print("dev cache sliced to", np.load("emb_dv3_konkle_dev16/emb.f16.npy", mmap_mode="r").shape)
PYX
fi

# The FULL 877,802-frame DINOv3-B grid. emb_enc_grid_top is the TOPLINE subset (82,811 frames) —
# using it silently trained every rung on ~9% of its manifest on 2026-08-22.
TRAIN=emb_dv3_grid_877k
PUREC=emb_enc/dinov3b_ots                   # whole-frame MEANPATCH (R=1), the honest pure baseline
EVAL="--eval-cache emb_enc_grid_eval/dinov3b_ots_konkle --eval-frames manifests/eval_frames_konkle.parquet"
EVAL1="--eval-cache emb_enc_eval/dinov3b_ots_konkle --eval-frames manifests/eval_frames_konkle.parquet"
DEV="--dev-cache emb_dv3_konkle_dev16 --dev-frames manifests/eval_frames_konkle_dev.parquet"
COV="--min-coverage 0.9"

# build English-dominant variants of every manifest the ladder/scaling use
for man in grid_baseline_train grid_t15_filtnat_train grid_t15_train grid_t2_labels_train \
           grid_baseline_full scale_rand_10000 scale_rand_30000 scale_rand_100000 scale_rand_300000; do
  [ -f manifests/${man}_en.parquet ] || $PY -B src/filter_english.py --in manifests/$man.parquet \
      --out manifests/${man}_en.parquet --min 80 >> logs/p5_english.log 2>&1
done
echo "english-dominant manifests built"

run () {  # run <tag> <manifest> <gpu> [extra flags]
  local tag=$1 man=$2 gpu=$3; shift 3
  for s in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$gpu $PY -B src/train_frame_mil.py --window 0 \
      --manifest manifests/$man.parquet --caches $TRAIN $EVAL $DEV $COV \
      --seed $s --out runs/P5_${tag}_s$s "$@" > logs/p5_${tag}_s$s.log 2>&1
  done
}

# ---- 1. the ladder (the UNRECOVERABLE display item) --------------------------------
echo "=== ladder ==="
purerun () {  # whole-frame meanpatch baseline: its own R=1 cache and matching R=1 eval
  local tag=$1 man=$2 gpu=$3
  for s in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$gpu $PY -B src/train_frame_mil.py --window 0 \
      --manifest manifests/$man.parquet --caches $PUREC $EVAL1 $DEV $COV \
      --seed $s --out runs/P5_${tag}_s$s > logs/p5_${tag}_s$s.log 2>&1
  done
}
purerun lad_pure grid_baseline_train $G1 &
run lad_region   grid_baseline_train      $G2 &              # + region MIL
run lad_filter   grid_t15_filtnat_train   $G3 &              # + alignment filter
wait
run lad_word     grid_t15_train           $G1 &              # + word selection
run lad_vision   grid_t2_labels_train     $G2 &              # + vision binding (clean-label ceiling)
wait
# English-dominant arm
purerun lad_pure_en grid_baseline_train_en $G1 &
run lad_region_en grid_baseline_train_en    $G2 &
run lad_filter_en grid_t15_filtnat_train_en $G3 &
wait
run lad_word_en   grid_t15_train_en         $G1 &
run lad_vision_en grid_t2_labels_train_en   $G2 &
wait
echo "LADDER_DONE"

# ---- 2. the scaling curve (so one axis = one rig) -----------------------------------
echo "=== scaling ==="
i=0
for man in scale_rand_10000 scale_rand_30000 scale_rand_100000 scale_rand_300000 \
           grid_baseline_train grid_baseline_full \
           scale_rand_10000_en scale_rand_30000_en scale_rand_100000_en scale_rand_300000_en \
           grid_baseline_train_en grid_baseline_full_en; do
  g=$(echo $FREE | cut -d" " -f$(( i % 3 + 1 ))); run sc_$man $man $g & i=$((i+1))
  [ $((i % 3)) -eq 0 ] && wait
done
wait
echo "SCALING_DONE"

# ---- 3. rebuild the results tables --------------------------------------------------
$PY src/scrape_runs.py && $PY src/scrape_evals.py
n=$(ls -d runs/P5_* 2>/dev/null | wc -l)
echo "P5 run dirs created: $n"
[ "$n" -gt 0 ] || { echo "PHASE5 PRODUCED NOTHING — check logs/p5_*.log"; exit 1; }
echo "PHASE5_DONE"
