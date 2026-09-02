"""Whole-frame (no-MIL) caches: mean over grid cells -> [N,1,D]. For 17-row eval caches
(CLS+grid) the mean is over rows 1:; for 16-row frame caches over all rows."""
import glob
import numpy as np
import pandas as pd
import shutil
from pathlib import Path

EMB = "/ccn2b/dataset/babyview/2026.1/outputs/image_embeddings"
JOBS = []
for enc in ["dinov3b", "dinov3l", "vits_bv", "vitb_bv"]:
    d = f"{EMB}/{enc}_grid4x4" if enc != "dinov3b" else f"{EMB}/dinov3b_grid4x4"
    srcs = [d] if Path(d, "emb.f16.npy").exists() else sorted(glob.glob(f"{d}/shard_*"))
    JOBS.append((enc, srcs, f"/data2/mcfrank/vlm-headcam/emb_wf/{enc}"))
EVAL = {"dinov3b": ("emb_enc_grid_eval/dinov3b_ots_konkle", "emb_dv3_konkle_dev16"),
        "dinov3l": ("emb_ch8_eval/dinov3l_grid4x4_konkle", "emb_ch8_eval/dinov3l_grid4x4_konkle_dev"),
        "vits_bv": ("emb_ch8_eval/vits_bv_konkle", "emb_ch8_eval/vits_bv_konkle_dev"),
        "vitb_bv": ("emb_ch8_eval/vitb_bv_konkle", "emb_ch8_eval/vitb_bv_konkle_dev")}

def meancache(src, dst):
    e = np.load(f"{src}/emb.f16.npy", mmap_mode="r")
    R = e.shape[1]
    start = 1 if R == 17 else 0          # 17 = CLS+grid: drop CLS; 16 = grid only
    out = np.zeros((e.shape[0], 1, e.shape[2]), dtype=np.float16)
    B = 100_000
    for i in range(0, len(e), B):
        out[i:i+B, 0] = e[i:i+B, start:].astype(np.float32).mean(1).astype(np.float16)
    Path(dst).mkdir(parents=True, exist_ok=True)
    np.save(f"{dst}/emb.f16.npy", out)
    shutil.copy(f"{src}/index.parquet", f"{dst}/index.parquet")
    print(f"  {src} R={R} -> {dst} {out.shape}", flush=True)

import os
import sys
os.chdir("/data2/mcfrank/vlm-headcam")
ONLY = sys.argv[1] if len(sys.argv) > 1 else None
for enc, srcs, dstroot in JOBS:
    if ONLY and enc != ONLY:
        continue
    e0 = np.load(f"{srcs[0]}/emb.f16.npy", mmap_mode="r")
    print(f"{enc}: frame cache R={e0.shape[1]} D={e0.shape[2]}", flush=True)
    for k, s in enumerate(srcs):
        meancache(s, f"{dstroot}/shard_{k}" if len(srcs) > 1 else dstroot)
    for tag, src in zip(["konkle", "konkle_dev"], EVAL[enc]):
        meancache(src, f"emb_wf_eval/{enc}_{tag}")
print(f"WF_ENC_DONE {ONLY or chr(97)}")
