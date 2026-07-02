"""Draw skeleton overlays for the strict-pointing candidates so we can eyeball whether the
'extended straight arm' detections are real referential points or just reaching/gesturing."""
import os, sys
import pandas as pd
sys.path.insert(0, "/data2/mcfrank/vlm-headcam/src")
from pose_lib import load_pose, frame_path, draw_overlay
W = "/data2/mcfrank/vlm-headcam"
OUT = f"{W}/runs/point_overlays"; os.makedirs(OUT, exist_ok=True)
cand = pd.read_csv(f"{W}/runs/point_candidates.csv")
for k, r in enumerate(cand.itertuples(index=False)):
    sec = f"{int(r.frame_idx):05d}"
    pose = load_pose(f"/ccn2/dataset/babyview/outputs_20250312/pose/4M_frames_old/{r.dir}/{sec}.pkl")
    fp = frame_path(r.dir, sec)
    out = f"{OUT}/{k:02d}_reach{r.reach:.1f}_{r.dir[:18]}_{sec}.jpg"
    draw_overlay(fp if os.path.exists(fp) else "", pose, out)
print(f"wrote {len(cand)} overlays to {OUT}")
print(cand[["reach", "straight", "clip_score_max", "text"]].to_string())
