#!/bin/bash
# Batch 1 (region-MIL), dynamic-GPU version. Waits for the (already running) region embed,
# merges shards, then runs the grid across ALL currently-free GPUs.
set -u
W=/data2/mcfrank/vlm-headcam
PY=/data2/mcfrank/ladder/condaenv/bin/python
EV=$W/manifests/eval_frames.parquet
mkdir -p $W/runs

echo "[b1b] waiting for region embedding..."
until grep -q "^done" $W/embed_reg_0.log 2>/dev/null && grep -q "^done" $W/embed_reg_1.log 2>/dev/null; do sleep 20; done
echo "[b1b] merging"
$PY - <<'PYEOF'
import numpy as np, pandas as pd
from pathlib import Path
W=Path("/data2/mcfrank/vlm-headcam"); parts=[]; idxs=[]; base=0
for i in [0,1]:
    e=np.load(W/f"emb_reg_{i}/emb.f16.npy"); ix=pd.read_parquet(W/f"emb_reg_{i}/index.parquet").copy()
    ix["row"]=base+np.arange(len(ix)); base+=len(e); parts.append(e); idxs.append(ix)
emb=np.concatenate(parts,0); idx=pd.concat(idxs,ignore_index=True)
(W/"emb_reg").mkdir(exist_ok=True)
np.save(W/"emb_reg/emb.f16.npy", emb); idx.to_parquet(W/"emb_reg/index.parquet")
print("merged", emb.shape)
PYEOF

# detect free GPUs now (memory < 8000 MB)
mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk '$2<8000{print $1}')
NG=${#GPUS[@]}
[ "$NG" -eq 0 ] && GPUS=(1 6) && NG=2
echo "[b1b] grid on GPUs: ${GPUS[*]}"

JOBS=(
"R_oracle_across plain aligned_S00360001"
"R_oracle_within plain oracle_within"
"R_plain_across_140000 plain boot_across_140000"
"R_plain_within_110000 plain boot_within_110000"
"R_boot_across_20000 boot boot_across_20000"
"R_boot_across_60000 boot boot_across_60000"
"R_boot_across_140000 boot boot_across_140000"
"R_boot_within_20000 boot boot_within_20000"
"R_boot_within_60000 boot boot_within_60000"
"R_boot_within_110000 boot boot_within_110000"
)
for gi in "${!GPUS[@]}"; do
 (
  for k in "${!JOBS[@]}"; do
    [ $((k % NG)) -ne "$gi" ] && continue
    set -- ${JOBS[$k]}; name=$1; mode=$2; man=$3
    out=$W/runs/$name; [ -f $out/DONE ] && continue
    if [ "$mode" = plain ]; then EX="--mode plain --epochs 20"; else EX="--mode boot --warmup 5 --rounds 6 --epochs-per-round 2"; fi
    env CUDA_VISIBLE_DEVICES=${GPUS[$gi]} PYTHONDONTWRITEBYTECODE=1 $PY -B $W/src/train_region_mil.py \
      --manifest $W/manifests/$man.parquet --region-cache $W/emb_reg --eval-frames $EV \
      --out $out $EX > $out.log 2>&1 && touch $out/DONE || touch $out/FAILED
  done
 ) &
done
wait
touch $W/runs/BATCH1_DONE
echo "[b1b] done"
