"""Assemble the union of all frames any experiment needs, subtract what's already cached,
and shard the remainder for parallel embedding.

Union = all 37 children's utterance mid-frames (from full_clip_results) + all eval-set
frames. Output: manifests/shard_{i}.parquet for i in 0..N-1 (only NEW frames).
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from common import CLIP_RESULTS, MANIFEST_DIR, EMB_DIR

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5

df = pd.read_csv(CLIP_RESULTS, usecols=["video_name", "utterance_start_time", "utterance_end_time"])
df = df.dropna()
df["frame_idx"] = ((df.utterance_start_time + df.utterance_end_time) / 2).astype(int)
frames = df.rename(columns={"video_name": "video_id"})[["video_id", "frame_idx"]]

# add every eval set's frames
for p in Path(MANIFEST_DIR).glob("eval_frames*.parquet"):
    frames = pd.concat([frames, pd.read_parquet(p)[["video_id", "frame_idx"]]])

frames = frames.drop_duplicates().reset_index(drop=True)
print("total union frames:", len(frames))

# subtract already-cached
idx_path = Path(EMB_DIR) / "index.parquet"
if idx_path.exists():
    have = pd.read_parquet(idx_path)
    have_keys = set(zip(have.video_id, have.frame_idx.astype(int)))
    mask = [(v, int(f)) not in have_keys for v, f in zip(frames.video_id, frames.frame_idx)]
    frames = frames[mask].reset_index(drop=True)
    print("cached:", len(have), "| new to embed:", len(frames))

# shuffle so shards are balanced in per-video locality, then split
frames = frames.sample(frac=1.0, random_state=0).reset_index(drop=True)
for i in range(N):
    sh = frames.iloc[i::N].reset_index(drop=True)
    sh.to_parquet(Path(MANIFEST_DIR) / f"shard_{i}.parquet")
    print(f"shard_{i}: {len(sh)}")
