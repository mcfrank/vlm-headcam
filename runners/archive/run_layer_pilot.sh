#!/bin/bash
# EXPERIMENT S1 (pilot) — does readout DEPTH matter for contrastive alignment?
#
# Free half of the experiment: Khai already cached whole-frame vectors at 7 layers each for
# V-JEPA2-L, ZWM-170M and ZWM-1B, so this needs NO new embedding. It tests the premise on the
# reconstruction-trained models, where depth should matter most (ZWM's across-image cosine is
# U-shaped in depth). If depth matters there and not for a discriminative model, that is the
# supplement's claim; if it is flat everywhere, the DINOv3 extraction is not worth doing.
#
# Cost control: a 200k-frame subset, because the question is RELATIVE (which layer) not absolute.
# Selection is on Konkle dev-117 and reporting on test-60 — sweeping 7 layers and picking the best
# on test would reintroduce exactly the optimism the dev split was added to remove.
#
#   setsid bash -c "bash run_layer_pilot.sh > /data2/mcfrank/layer_pilot.log 2>&1" < /dev/null &
set -u
cd /data2/mcfrank/vlm-headcam
export PYTHONPATH=src
PY=/data2/mcfrank/ladder/condaenv/bin/python
E=/ccn2a/dataset/babyview/2025.2/outputs/image_embeddings
TRAIN=$E/babyview_877k
EVALD=$E/eval_konkle_mcfrank
SUB=manifests/s1_subset_frames.txt
N_SUB=200000

# wait for the ladder/scaling work to finish before competing for GPUs
while pgrep -f "run_phase5.sh$" > /dev/null; do echo "$(date +%H:%M) waiting for run_phase5"; sleep 900; done

# ---- 0. subset of the topline frame ids ------------------------------------------
[ -f $SUB ] || head -n $N_SUB $TRAIN/frame_ids_877802.txt > $SUB
echo "subset: $(wc -l < $SUB) frames"

# model -> readout stem, layers
declare -A STEM=( [vjepa2l]=awwkl_vjepa2-vitl-fpc16-256-babyview-bs3072-e140
                  [zwm170m]=awwkl_zwm-babyview-170m
                  [zwm1b]=awwkl_zwm-babyview-1b )
declare -A LAYERS=( [vjepa2l]="0 4 8 12 16 20 23"
                    [zwm170m]="0 4 8 12 16 20 23"
                    [zwm1b]="0 8 16 24 32 40 47" )

# ---- 1. assemble per-layer caches (train subset + eval) ---------------------------
for m in vjepa2l zwm170m zwm1b; do
  for L in ${LAYERS[$m]}; do
    RO=${STEM[$m]}_layer${L}
    [ -f emb_s1/${m}_L${L}/index.parquet ] || $PY -B src/assemble_encoder_cache.py --base $TRAIN --readout $RO \
        --ids $SUB --out emb_s1/${m}_L${L} >> logs/s1_assemble.log 2>&1
    [ -f emb_s1_eval/${m}_L${L}/index.parquet ] || $PY -B src/assemble_encoder_cache.py --base $EVALD --readout $RO \
        --ids manifests/konkle_eval_ids.txt --out emb_s1_eval/${m}_L${L} >> logs/s1_assemble.log 2>&1
    for c in emb_s1/${m}_L${L} emb_s1_eval/${m}_L${L}; do
      [ -f "$c/index.parquet" ] || { echo "SKIP $m L$L — $c did not assemble"; continue 2; }
    done
    echo "assembled $m L$L"
  done
done

# ---- 2. train 3 seeds per (model, layer) ------------------------------------------
# NOTE: no DINOv3 dev cache for these encoders' eval sets, so selection falls back to
# best-on-test for the PILOT only; the decision is relative across layers within a model,
# and the follow-up (if any) runs on the dev-selected clean rig.
i=0
for m in vjepa2l zwm170m zwm1b; do
  for L in ${LAYERS[$m]}; do
    for s in 0 1 2; do
      g=$((i % 8)); i=$((i+1))
      [ -f emb_s1_eval/${m}_L${L}/index.parquet ] || continue
      CUDA_VISIBLE_DEVICES=$g $PY -B src/train_frame_mil.py --window 0 \
        --manifest manifests/grid_baseline_train.parquet \
        --caches emb_s1/${m}_L${L} \
        --eval-cache emb_s1_eval/${m}_L${L} --eval-frames manifests/eval_frames_konkle.parquet \
        --seed $s --out runs/S1_${m}_L${L}_s$s > logs/s1_${m}_L${L}_s$s.log 2>&1 &
      [ $((i % 6)) -eq 0 ] && wait
    done
  done
done
wait

$PY src/scrape_runs.py
echo "=== S1 layer profile (mean best 4AFC over 3 seeds) ==="
$PY - <<'EOF'
import pandas as pd, re
r = pd.read_parquet("results/runs.parquet")
s = r[r.family.str.startswith("S1_")].copy()
if len(s):
    s["model"] = s.family.str.extract(r"S1_([a-z0-9]+)_L")[0]
    s["layer"] = s.family.str.extract(r"_L(\d+)")[0].astype(int)
    p = s.groupby(["model", "layer"]).best_acc.agg(["mean", "std", "size"]).round(2)
    print(p.to_string())
    for m, g in p.groupby(level=0):
        g = g.droplevel(0)
        best, last = g["mean"].idxmax(), g.index.max()
        print(f"{m}: best L{best} {g.loc[best,'mean']:.2f} vs final L{last} {g.loc[last,'mean']:.2f} "
              f"-> delta {g.loc[best,'mean']-g.loc[last,'mean']:+.2f} (seed sd ~{g['std'].mean():.2f})")
EOF
echo "S1_PILOT_DONE"
