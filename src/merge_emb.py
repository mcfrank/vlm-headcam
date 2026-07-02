"""Merge embedding caches into one. Usage: merge_emb.py <out> <src1> <src2> ...
Dedups by (video_id, frame_idx) keeping first. Writes <out>/emb.f16.npy + index.parquet.
Include <out> among the sources if it already holds data you want to keep."""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

out = Path(sys.argv[1])
srcs = [Path(d) for d in sys.argv[2:]]

parts_idx, parts_emb = [], []
for si, d in enumerate(srcs):
    if not (d / "index.parquet").exists():
        continue
    idx = pd.read_parquet(d / "index.parquet")[["video_id", "frame_idx"]].copy()
    idx["frame_idx"] = idx.frame_idx.astype(int)
    idx["glob"] = len(parts_emb) and sum(len(e) for e in parts_emb) or 0
    idx["glob"] = idx["glob"] + np.arange(len(idx))
    parts_idx.append(idx)
    parts_emb.append(np.load(d / "emb.f16.npy"))
    print(f"{d}: {len(idx)}")

emb = np.concatenate(parts_emb, axis=0)
idx = pd.concat(parts_idx, ignore_index=True)
idx = idx.drop_duplicates(subset=["video_id", "frame_idx"], keep="first").reset_index(drop=True)
final = emb[idx["glob"].to_numpy()]
final_idx = idx[["video_id", "frame_idx"]].copy()
final_idx["row"] = np.arange(len(final_idx))

out.mkdir(parents=True, exist_ok=True)
np.save(out / "emb.f16.npy", final)
final_idx.to_parquet(out / "index.parquet")
print(f"merged -> {out}: {len(final)} unique frames")
