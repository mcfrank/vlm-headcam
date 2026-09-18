#!/bin/bash
# LEVANTE fix (2026-09-18): the original image manifest globbed *.webp, so the two .jpg targets
# (rubber band, turnstile) were never embedded and both items were unplayable for every model.
#
# Step 1 embeds the 629 item images with the paper's recipe (R=17, CLS + 4x4 grid) into
# emb_lev_<enc>_v2. Re-embedding changes batch composition, which perturbs the other images in
# the last bits (cosine >= 0.99998) — harmless, but it would blur what the fix did. So step 2
# APPENDS only the two new images' rows to the paper's caches (old rows byte-identical), and
# step 3 re-evaluates every checkpoint into results/lev_scaling_final_v2.csv for comparison.
set -u
cd /data2/mcfrank/vlm-headcam
export HF_HOME=/ccn2/u/khaiaw/.cache/huggingface HF_HUB_OFFLINE=1
P=/ccn2/u/khaiaw/miniconda3/envs/ccwm/bin/python
R=/ccn2b/dataset/babyview/eval_assets
M=/data2/mcfrank/oak_stage/lev_manifest_abs.parquet
LOG=logs/lev_fix_20260918.log
ENCS="dinov3s dinov3b dinov3l vits_bv vitb_bv vitl_bv"
echo "start $(date)" >> $LOG

# 1. embed (skipped if the _v2 caches already exist)
if [ ! -f emb_lev_vitl_bv_v2/index.parquet ]; then
  $P - <<EOF >> $LOG 2>&1
import pandas as pd
m = pd.read_parquet("$R/manifests/lev_vocab_manifest.parquet"); m["path"] = "$R/" + m.path
m.to_parquet("$M", index=False); print("manifest", len(m))
EOF
  declare -A HF=([dinov3s]=facebook/dinov3-vits16-pretrain-lvd1689m [dinov3b]=facebook/dinov3-vitb16-pretrain-lvd1689m [dinov3l]=facebook/dinov3-vitl16-pretrain-lvd1689m)
  declare -A NATIVE=([vits_bv]=/data2/mcfrank/dino_s2_vits [vitb_bv]=/data2/mcfrank/dino_s3_vitb [vitl_bv]=/data2/mcfrank/dino_s4_vitl)
  g=2
  for e in dinov3s dinov3b dinov3l; do
    CUDA_VISIBLE_DEVICES=$g $P -B src/embed_konkle.py --manifest $M --model ${HF[$e]} --grid 4 --out emb_lev_${e}_v2 > logs/lev_fix_$e.log 2>&1 & g=$((g+1))
  done
  for e in vits_bv vitb_bv vitl_bv; do
    CUDA_VISIBLE_DEVICES=$g $P -B /data2/mcfrank/dinov3/embed_native_dino.py --run ${NATIVE[$e]} --ckpt 199999 --manifest $M --out emb_lev_${e}_v2 > logs/lev_fix_$e.log 2>&1 & g=$((g+1))
  done
  wait
  echo "embeds done $(date)" >> $LOG
fi

# 2. append the two new rows to the paper's caches -> emb_lev_<enc>_fix; verify; swap
$P - <<'EOF' >> $LOG 2>&1
import numpy as np, pandas as pd, sys, shutil
from pathlib import Path
NEW = ["rubberBand", "turnstile"]; bad = 0
for e in ["dinov3s", "dinov3b", "dinov3l", "vits_bv", "vitb_bv", "vitl_bv"]:
    oi = pd.read_parquet(f"emb_lev_{e}/index.parquet"); oe = np.load(f"emb_lev_{e}/emb.f16.npy")
    vi = pd.read_parquet(f"emb_lev_{e}_v2/index.parquet"); ve = np.load(f"emb_lev_{e}_v2/emb.f16.npy")
    assert not oi.video_id.isin(NEW).any() and (oi.row.values == np.arange(len(oi))).all()
    add = vi.set_index("video_id").loc[NEW]
    emb = np.concatenate([oe, ve[add.row.values]])
    idx = pd.concat([oi, pd.DataFrame(dict(video_id=NEW, frame_idx=add.frame_idx.values,
                                           row=np.arange(len(oi), len(oi) + 2)))], ignore_index=True)
    out = Path(f"emb_lev_{e}_fix"); out.mkdir(exist_ok=True)
    np.save(out / "emb.f16.npy", emb); idx.to_parquet(out / "index.parquet")
    chk = np.load(out / "emb.f16.npy")
    ok = (np.array_equal(chk[:len(oi)], oe) and chk.shape == (len(oi) + 2,) + oe.shape[1:]
          and np.array_equal(chk[len(oi):], ve[add.row.values]))
    bad += not ok
    print(f"APPEND {e}: {oe.shape} -> {chk.shape} | old rows byte-identical {np.array_equal(chk[:len(oi)], oe)} | {'OK' if ok else 'FAIL'}")
sys.exit(1 if bad else 0)
EOF
[ $? -eq 0 ] || { echo "LEV_FIX_FAIL append" >> $LOG; exit 1; }
for e in $ENCS; do
  mv emb_lev_$e emb_lev_${e}.pre_20260918 && mv emb_lev_${e}_fix emb_lev_$e
done
echo "caches swapped $(date)" >> $LOG

# 3. re-evaluate every checkpoint
CUDA_VISIBLE_DEVICES=2 $P -B src/eval_lev_scaling.py --out results/lev_scaling_final_v2.csv > logs/lev_fix_eval.log 2>&1 \
  && echo "LEV_FIX_DONE $(date)" >> $LOG || echo "LEV_FIX_FAIL eval" >> $LOG
