"""Emit jobs.tsv for the A/B/C experiment dispatcher.
Columns (tab-sep): name  manifest  eval_frames  seed
"""
from pathlib import Path
from common import MANIFEST_DIR

M = str(MANIFEST_DIR)
EVAL = {"S00360001": f"{M}/eval_frames.parquet",
        "S00240001": f"{M}/eval_frames_S00240001.parquet",
        "S00370002": f"{M}/eval_frames_S00370002.parquet"}
SEEDS = [0, 1, 2]
jobs = []

# Exp B: aligned vs random, 3 children x 3 seeds
for ch in EVAL:
    for arm in ["aligned", "random"]:
        for s in SEEDS:
            jobs.append((f"B_{arm}_{ch}_s{s}", f"{M}/{arm}_{ch}.parquet", EVAL[ch], s))

# Exp C: threshold sweep at S00360001 (0.24 == aligned_S00360001, already in B)
for thr in ["0.26", "0.28"]:
    for s in SEEDS:
        jobs.append((f"C_thr{thr}_s{s}", f"{M}/thresh_{thr}.parquet", EVAL["S00360001"], s))

# Exp A (unfiltered, large) runs standalone on its own GPU, not via this dispatcher.

with open(f"{M}/jobs.tsv", "w") as f:
    for name, man, ev, s in jobs:
        f.write(f"{name}\t{man}\t{ev}\t{s}\n")
print(f"wrote {len(jobs)} jobs -> {M}/jobs.tsv")
