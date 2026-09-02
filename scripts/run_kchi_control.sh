#!/bin/bash
cd /data2/mcfrank/vlm-headcam
/data2/mcfrank/ladder/condaenv/bin/python - <<'PY'
import pandas as pd
en = pd.read_parquet("manifests/bv26_pairs_en_audio.parquet")
no = en[en.speaker != "KCHI"]
COLS = ["video_id", "frame_idx", "text"]
no[COLS].to_parquet("manifests/bv26a_nokchi.parquet", index=False)
print(f"noKCHI: {len(no):,} of {len(en):,} pairs ({en.speaker.eq('KCHI').mean()*100:.1f}% KCHI removed)")
for s in range(3):
    en.sample(len(no), random_state=5000 + s)[COLS].to_parquet(
        f"manifests/bv26a_randmatch_s{s}.parquet", index=False)
print("3 matched-N random draws written")
PY
export HF_HOME=/ccn2/u/khaiaw/.cache/huggingface HF_HUB_OFFLINE=1
PY=/ccn2/u/khaiaw/miniconda3/envs/ccwm/bin/python
EMB=/ccn2b/dataset/babyview/2026.1/outputs/image_embeddings
CACHES="$EMB/dinov3l_grid4x4/shard_0 $EMB/dinov3l_grid4x4/shard_1 $EMB/dinov3l_grid4x4/shard_2 $EMB/dinov3l_grid4x4/shard_3"
while :; do
  FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F", " '$2<41000{print $1}' | tr '\n' ' ')
  GPUS=($FREE); NG=${#GPUS[@]}; [ "$NG" -ge 1 ] && break; sleep 600
done
i=0
one () {
  local tag=$1 man=$2 s=$3
  [ -f "runs/F_dinov3l_${tag}_s$s/metrics.json" ] && return
  pgrep -f "out runs/F_dinov3l_${tag}_s$s( |$)" >/dev/null && return
  local g=${GPUS[$((i % NG))]}; i=$((i+1))
  CUDA_VISIBLE_DEVICES=$g $PY -B src/train_frame_mil.py --window 0 \
    --manifest manifests/$man.parquet --caches $CACHES \
    --eval-cache emb_ch8_eval/dinov3l_grid4x4_konkle --eval-frames manifests/eval_frames_konkle.parquet \
    --dev-cache emb_ch8_eval/dinov3l_grid4x4_konkle_dev --dev-frames manifests/eval_frames_konkle_dev.parquet \
    --min-coverage 0.9 --seed $s --out runs/F_dinov3l_${tag}_s$s > logs/f_${tag}_s$s.log 2>&1 &
  [ $((i % NG)) -eq 0 ] && wait
}
for s in 0 1 2; do one nokchi bv26a_nokchi $s; one randmatch bv26a_randmatch_s$s $s; done
wait
echo "KCHI_CONTROL_DONE: $(ls runs/F_dinov3l_nokchi_s*/metrics.json runs/F_dinov3l_randmatch_s*/metrics.json 2>/dev/null | wc -l)/6"
