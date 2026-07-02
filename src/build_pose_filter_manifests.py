"""Decisive test: is a pose cue a good TRAINING FILTER (better word learner), regardless of
CLIP? Build size-matched training sets from the pose-covered pool:
  child   : child's own hand present
  adultface: adult face present
  clippos : clip>0.24 (POSITIVE CONTROL -- known to beat random)
  random  : random subset
Same N, same pool. If child ~ clippos > random, the cue selects good moments CLIP-blindly."""
import numpy as np, pandas as pd
from pathlib import Path
W = Path("/data2/mcfrank/vlm-headcam")
df = pd.read_parquet(W / "manifests/pose_cues_full.parquet")
# keep only pairs with a region embedding (they come from boot_across, so all do) and non-empty text
df = df[df.text.astype(str).str.len() > 0].reset_index(drop=True)
rng = np.random.default_rng(0)

child = df[df.child_hand == 1]
clippos = df[df.clip_score_max > 0.24]
adultface = df[df.adult_face == 1]
N = min(len(child), len(clippos), 8000)
print(f"pool={len(df)}  child={len(child)}  clip>0.24={len(clippos)}  adultface={len(adultface)}  -> matched N={N}")

def sub(d): return d.sample(N, random_state=0)[["video_id", "frame_idx", "text", "clip_score_max", "child_id"]]
sets = {"child": sub(child), "adultface": sub(adultface), "clippos": sub(clippos), "random": sub(df)}
for k, v in sets.items():
    v.reset_index(drop=True).to_parquet(W / f"manifests/PF_{k}.parquet")
    print(f"  PF_{k}: {len(v)}  mean_clip={v.clip_score_max.mean():.3f}  aligned%={(v.clip_score_max>0.24).mean():.1%}")
