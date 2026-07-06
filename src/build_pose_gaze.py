"""Caregiver gaze/head direction from raw face keypoints (pkls): pitch (nose below eyes = looking
down, e.g. at a held/low object) and yaw (head turned). Does gaze direction predict Gemini
alignment? Sharded pkl read over a 150k-pair sample. --shard i --nshard N -> gaze_shard_i.parquet."""
import argparse
import glob
import os
import numpy as np
import pandas as pd
from pose_lib import load_pose

PKL = "/ccn2/dataset/babyview/2025.2/outputs/pose_1fps"
ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
a = ap.parse_args()

posevids = {os.path.basename(d) for d in glob.glob(PKL + "/*") if os.path.isdir(d)}
G = pd.read_parquet("scored/gemini_full.parquet")[["video_id", "frame_idx", "alignment"]]
G = G[G.video_id.isin(posevids)].sample(150000, random_state=0)
mine = G.iloc[a.shard::a.nshard]

rows = []
for r in mine.itertuples(index=False):
    path = f"{PKL}/{r.video_id}/{int(r.frame_idx):05d}.pkl"
    if not os.path.exists(path):
        continue
    try:
        p = load_pose(path)
    except Exception:
        continue
    if not p.get("persons"):
        continue
    bb = np.asarray(p["bboxes"], float)
    i = int(((bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])).argmax())
    per = p["persons"][i]
    kp = np.asarray(per["kp"], float); sc = np.asarray(per["score"], float)
    if len(kp) < 5 or sc[:5].min() < 0.3:
        continue
    nose, le, re, lear, rear = kp[0], kp[1], kp[2], kp[3], kp[4]
    scale = abs(lear[0] - rear[0]) + 1e-6
    pitch = (nose[1] - (le[1] + re[1]) / 2) / scale        # >0 nose below eyes = looking down
    yaw = abs(nose[0] - (lear[0] + rear[0]) / 2) / scale    # head turned off-frontal
    rows.append((r.video_id, int(r.frame_idx), float(r.alignment), float(pitch), float(yaw)))

pd.DataFrame(rows, columns=["video_id", "frame_idx", "alignment", "pitch", "yaw"]).to_parquet(
    f"manifests/gaze_shard_{a.shard}.parquet", index=False)
print(f"shard {a.shard}: {len(mine)} sampled, {len(rows)} with gaze", flush=True)
