"""Caregiver-close utterance cue: apparent size of the nearest caregiver. Largest person with
a visible face/torso whose bbox centroid is in the upper 2/3 of the frame (excludes the child's
own hands, which enter low from the bottom). Bigger => closer, more engaged => more referential."""
import os, re, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "/data2/mcfrank/vlm-headcam/src")
from pose_lib import load_pose, POSE_ROOT, NOSE, LEYE, REYE, LEAR, REAR, LSHO, RSHO, LHIP, RHIP
W = "/data2/mcfrank/vlm-headcam"; FW, FH = 512.0, 910.0; TH = 0.3

def recid(v):
    m = re.search(r"rec[A-Za-z0-9]+", str(v)); return m.group(0) if m else None

def cg_close(pose):
    best = 0.0
    for P in pose["persons"]:
        i = P["id"]
        if not (isinstance(i, int) and i < len(pose["bboxes"]) and P["kp"].shape[0] == 133):
            continue
        sc = P["score"]
        face = np.mean(sc[[NOSE, LEYE, REYE, LEAR, REAR]]); torso = np.mean(sc[[LSHO, RSHO, LHIP, RHIP]])
        if not (face > 0.3 or torso > 0.3):
            continue
        b = pose["bboxes"][i]; cy = 0.5 * (b[1] + b[3]) / FH
        if cy > 0.66:                          # centroid in lower third -> likely child's own body/hands
            continue
        best = max(best, float(max(0, b[2]-b[0]) * max(0, b[3]-b[1]) / (FW*FH)))
    return best

man = pd.read_parquet(f"{W}/manifests/boot_across_140000.parquet")
xw = pd.read_csv(f"{W}/metadata/videos.csv", dtype=str, low_memory=False); xw.columns = [c.lstrip("﻿") for c in xw.columns]
rec2gcp = dict(zip(xw.unique_video_id, xw.superseded_gcp_name_feb25))
posedirs = set(os.listdir(POSE_ROOT))
man["dir"] = man.video_id.map(lambda v: (lambda g: f"{g}_processed" if isinstance(g, str) else None)(rec2gcp.get(recid(v))))
cov = man[man.dir.map(lambda d: isinstance(d, str) and d in posedirs)].copy()

rows = []; t0 = time.time()
for j, (dd, g) in enumerate(cov.groupby("dir")):
    for r in g.itertuples(index=False):
        pp = f"{POSE_ROOT}/{dd}/{int(r.frame_idx):05d}.pkl"
        if not os.path.exists(pp): continue
        try: c = cg_close(load_pose(pp))
        except Exception: continue
        rows.append(dict(video_id=r.video_id, frame_idx=int(r.frame_idx), cg_close=c, clip_score_max=r.clip_score_max))
    if j % 800 == 0: print(f"  {j} vids {len(rows)} {time.time()-t0:.0f}s", flush=True)
df = pd.DataFrame(rows); df.to_parquet(f"{W}/manifests/cg_close_pairs.parquet")
clip = df.clip_score_max.to_numpy()
print(f"cg_close pairs: {len(df)}  nonzero {int((df.cg_close>0).sum())} ({(df.cg_close>0).mean():.0%})")
print(f"rho(cg_close, clip) = {np.corrcoef(pd.Series(df.cg_close).rank(), pd.Series(clip).rank())[0,1]:+.3f}")
print(f"mean clip | cg_close>median: {clip[df.cg_close>df.cg_close.median()].mean():.3f} vs <=: {clip[df.cg_close<=df.cg_close.median()].mean():.3f}")
