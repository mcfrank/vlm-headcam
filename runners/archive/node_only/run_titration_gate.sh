#!/bin/bash
# Track B, gate variant — the cue used as a HARD filter (keep top gate-frac as positives,
# drop the rest), mirroring the ch.3 CLIP filter. rho=1.0 gate should approach oracle 49.9,
# anchoring the titration to a known reference. The rho where gate 4AFC lifts clearly above
# the unweighted baseline (~34.8) is the ignition threshold for a filter-style cue.
set -u
W=/data2/mcfrank/vlm-headcam; PY=/data2/mcfrank/ladder/condaenv/bin/python
EV=$W/manifests/eval_frames.parquet; RC=$W/emb_reg
MAN=$W/manifests/boot_across_140000.parquet
COMMON="--mode boot --warmup 5 --rounds 6 --epochs-per-round 2 --region-cache $RC --eval-frames $EV --manifest $MAN --prior-mode gate --titrate-cov 1.0 --gate-frac 0.12"
GPUS=(0 1 2 3 4 5); NG=${#GPUS[@]}

JOBS=(
  "G_r10_s0 0.1 0" "G_r10_s1 0.1 1"
  "G_r20_s0 0.2 0" "G_r20_s1 0.2 1"
  "G_r30_s0 0.3 0" "G_r30_s1 0.3 1"
  "G_r50_s0 0.5 0" "G_r50_s1 0.5 1"
  "G_r70_s0 0.7 0" "G_r70_s1 0.7 1"
  "G_r100_s0 1.0 0" "G_r100_s1 1.0 1"
)
run(){ # gpu name rho seed
  env CUDA_VISIBLE_DEVICES=$1 PYTHONDONTWRITEBYTECODE=1 $PY -B $W/src/train_region_mil.py $COMMON \
    --titrate-rho $3 --seed $4 --out $W/runs/$2 > $W/runs/$2.log 2>&1 && touch $W/runs/$2/DONE || touch $W/runs/$2/FAILED; }

i=0
for spec in "${JOBS[@]}"; do
  set -- $spec; name=$1; rho=$2; seed=$3
  gpu=${GPUS[$((i % NG))]}
  run $gpu $name $rho $seed &
  i=$((i+1))
  if (( i % NG == 0 )); then wait; fi
done
wait
touch $W/runs/TITRATION_GATE_DONE
echo "[titration-gate] done"
