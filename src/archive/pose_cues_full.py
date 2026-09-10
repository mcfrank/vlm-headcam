"""Compute pose cues for ALL pose-covered training pairs (with keys), so the cues can be
used as TRAINING FILTERS and screened against downstream 4AFC instead of CLIP. Three
independent cue families all gave rho~0 vs CLIP; the mean-CLIP-identical pattern says CLIP is
blind to them, so the real test is whether a cue selects better *learning* moments."""
import os, re, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "/data2/mcfrank/vlm-headcam/src")
from pose_lib import load_pose, POSE_ROOT
from pose_hands_split import frame_cues, recid

W = "/data2/mcfrank/vlm-headcam"
man = pd.read_parquet(f"{W}/manifests/boot_across_140000.parquet")
xw = pd.read_csv(f"{W}/metadata/videos.csv", dtype=str, low_memory=False)
xw.columns = [c.lstrip("﻿") for c in xw.columns]
rec2gcp = dict(zip(xw.unique_video_id, xw.superseded_gcp_name_feb25))
posedirs = set(os.listdir(POSE_ROOT))
man["dir"] = man.video_id.map(lambda v: (lambda g: f"{g}_processed" if isinstance(g, str) else None)(rec2gcp.get(recid(v))))
cov = man[man.dir.map(lambda d: isinstance(d, str) and d in posedirs)].copy()
print(f"pose-covered pairs: {len(cov)}", flush=True)

rows = []; t0 = time.time()
for j, (dd, g) in enumerate(cov.groupby("dir")):
    for r in g.itertuples(index=False):
        pp = f"{POSE_ROOT}/{dd}/{int(r.frame_idx):05d}.pkl"
        if not os.path.exists(pp):
            continue
        try:
            c = frame_cues(load_pose(pp))
        except Exception:
            continue
        c.update(video_id=r.video_id, frame_idx=int(r.frame_idx), text=r.text,
                 clip_score_max=r.clip_score_max, child_id=r.child_id)
        rows.append(c)
    if j % 500 == 0:
        print(f"  {j} vids {len(rows)} frames {time.time()-t0:.0f}s", flush=True)
df = pd.DataFrame(rows)
df.to_parquet(f"{W}/manifests/pose_cues_full.parquet")
print(f"wrote pose_cues_full: {len(df)} pairs")
print(f"  child_hand={df.child_hand.mean():.1%}  adult_hand={df.adult_hand.mean():.1%}  adult_face={df.adult_face.mean():.1%}")
