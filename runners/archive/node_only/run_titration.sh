#!/bin/bash
# Track B — titration: how much aligned-ness information does the bootstrap need to ignite?
# Inject a synthetic cue of controlled quality (Spearman rho vs held-out CLIP) and coverage,
# on the plain region-MIL boot recipe (no proto/lang-prior, so the cue is the only added signal).
# References on this pool: unweighted region-MIL boot ~34.8, region-MIL oracle 49.9, chance 25.
set -u
W=/data2/mcfrank/vlm-headcam; PY=/data2/mcfrank/ladder/condaenv/bin/python
EV=$W/manifests/eval_frames.parquet; RC=$W/emb_reg
MAN=$W/manifests/boot_across_140000.parquet
COMMON="--mode boot --warmup 5 --rounds 6 --epochs-per-round 2 --region-cache $RC --eval-frames $EV --manifest $MAN"
GPUS=(0 1 2 3 4 5); NG=${#GPUS[@]}
mkdir -p $W/runs

# job spec: name  rho  cov  prior-mode  seed
JOBS=(
  # quality curve (fixed cue = the weight every round), cov=1.0, two seeds
  "T_q_r00_s0 0.0 1.0 fixed 0"   "T_q_r00_s1 0.0 1.0 fixed 1"
  "T_q_r05_s0 0.05 1.0 fixed 0"  "T_q_r05_s1 0.05 1.0 fixed 1"
  "T_q_r10_s0 0.1 1.0 fixed 0"   "T_q_r10_s1 0.1 1.0 fixed 1"
  "T_q_r20_s0 0.2 1.0 fixed 0"   "T_q_r20_s1 0.2 1.0 fixed 1"
  "T_q_r30_s0 0.3 1.0 fixed 0"   "T_q_r30_s1 0.3 1.0 fixed 1"
  "T_q_r50_s0 0.5 1.0 fixed 0"   "T_q_r50_s1 0.5 1.0 fixed 1"
  "T_q_r70_s0 0.7 1.0 fixed 0"   "T_q_r70_s1 0.7 1.0 fixed 1"
  "T_q_r100_s0 1.0 1.0 fixed 0"  "T_q_r100_s1 1.0 1.0 fixed 1"
  # amplification test: cue seeds warmup only, then endogenous EM (does final rho > seed rho?)
  "T_seed_r10 0.1 1.0 seed 0" "T_seed_r20 0.2 1.0 seed 0"
  "T_seed_r30 0.3 1.0 seed 0" "T_seed_r50 0.5 1.0 seed 0"
  # coverage axis: sparse-strong vs dense-weak (fixed mode)
  "T_cov_r100c05 1.0 0.05 fixed 0" "T_cov_r100c10 1.0 0.1 fixed 0"
  "T_cov_r70c10 0.7 0.1 fixed 0"   "T_cov_r50c20 0.5 0.2 fixed 0"
)

run(){ # gpu name rho cov mode seed
  env CUDA_VISIBLE_DEVICES=$1 PYTHONDONTWRITEBYTECODE=1 $PY -B $W/src/train_region_mil.py $COMMON \
    --titrate-rho $3 --titrate-cov $4 --prior-mode $5 --seed $6 \
    --out $W/runs/$2 > $W/runs/$2.log 2>&1 && touch $W/runs/$2/DONE || touch $W/runs/$2/FAILED; }

# launch in fixed waves of NG (one job per GPU per wave); full wait between waves so no
# two jobs ever share a GPU. Runs are ~equal length, so wave imbalance is small.
i=0
for spec in "${JOBS[@]}"; do
  set -- $spec; name=$1; rho=$2; cov=$3; mode=$4; seed=$5
  gpu=${GPUS[$((i % NG))]}
  run $gpu $name $rho $cov $mode $seed &
  i=$((i+1))
  if (( i % NG == 0 )); then wait; fi
done
wait
touch $W/runs/TITRATION_DONE
echo "[titration] all done"
