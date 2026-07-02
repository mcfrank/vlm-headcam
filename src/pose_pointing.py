"""Detect adult POINTING (the rare-but-precise cue: FTF-2013 precision .78, recall .10) and
map how sparse it is at increasing strictness. A point is an adult (attached face/torso) with
an EXTENDED arm (wrist far from shoulder), optionally a STRAIGHT elbow, optionally an EXTENDED
INDEX finger (hand shape). Saves a per-pair point score for the combined filter, and the
strictest candidates for a human look at real frames."""
import os, re, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "/data2/mcfrank/vlm-headcam/src")
from pose_lib import (load_pose, POSE_ROOT, NOSE, LEYE, REYE, LEAR, REAR, LSHO, RSHO,
                      LELB, RELB, LWRI, RWRI, LHIP, RHIP, LHAND, RHAND)
W = "/data2/mcfrank/vlm-headcam"; TH = 0.3
ARMS = [(LSHO, LELB, LWRI, LHAND), (RSHO, RELB, RWRI, RHAND)]

def recid(v):
    m = re.search(r"rec[A-Za-z0-9]+", str(v)); return m.group(0) if m else None

def index_extended(kp, sc, hand_idx):
    """hand_idx: 21 COCO-WholeBody hand kpts (0 wrist; index tip=8, middle=12, ring=16, pinky=20)."""
    if not all(sc[hand_idx[j]] > TH for j in [0, 8, 12, 16, 20]):
        return np.nan
    w = kp[hand_idx[0]]
    d = lambda t: float(np.hypot(*(kp[hand_idx[t]] - w)))
    idx, others = d(8), np.mean([d(12), d(16), d(20)])
    return float(idx > 1.25 * others and idx >= max(d(12), d(16), d(20)))  # index protrudes, others curled

def point_features(pose):
    best = dict(reach=0.0, straight=0.0, index=np.nan, has_adult=0)
    for P in pose["persons"]:
        i = P["id"]
        if not (isinstance(i, int) and i < len(pose["bboxes"]) and P["kp"].shape[0] == 133):
            continue
        kp, sc = P["kp"], P["score"]; vis = sc > TH
        face = np.mean(sc[[NOSE, LEYE, REYE, LEAR, REAR]]); torso = np.mean(sc[[LSHO, RSHO, LHIP, RHIP]])
        if not (face > 0.3 or torso > 0.3):     # adults only (attached to a body)
            continue
        best["has_adult"] = 1
        b = pose["bboxes"][i]; bh = float(b[3] - b[1])
        # robust body scale: torso length (shoulder->hip), else bbox height; guards the
        # degenerate case where shoulders collapse (sideways/mis-detected person)
        torso = [np.hypot(*(kp[s] - kp[h])) for s, h in [(LSHO, LHIP), (RSHO, RHIP)]
                 if vis[s] and vis[h]]
        scale = float(np.median(torso)) if torso else 0.45 * bh
        if not (scale == scale and scale > 15):
            continue
        for sh, el, wr, hand in ARMS:
            if not (vis[sh] and vis[wr]):
                continue
            reach = float(np.hypot(*(kp[wr] - kp[sh])) / scale)
            if reach > 3.0:                      # implausible for a real arm -> skip degenerate
                continue
            straight = np.nan
            if vis[el]:
                a, b = kp[el] - kp[sh], kp[wr] - kp[el]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                straight = float((a @ b) / (na * nb)) if na > 1 and nb > 1 else np.nan
            hidx = LHAND[0] and (list(range(91, 112)) if hand is LHAND else list(range(112, 133)))
            ix = index_extended(kp, sc, hidx)
            if reach > best["reach"]:
                best.update(reach=reach, straight=(straight if straight == straight else 0.0),
                            index=ix)
    return best

def main(cap=80000):
    man = pd.read_parquet(f"{W}/manifests/boot_across_140000.parquet")
    xw = pd.read_csv(f"{W}/metadata/videos.csv", dtype=str, low_memory=False)
    xw.columns = [c.lstrip("﻿") for c in xw.columns]
    rec2gcp = dict(zip(xw.unique_video_id, xw.superseded_gcp_name_feb25))
    posedirs = set(os.listdir(POSE_ROOT))
    man["dir"] = man.video_id.map(lambda v: (lambda g: f"{g}_processed" if isinstance(g, str) else None)(rec2gcp.get(recid(v))))
    cov = man[man.dir.map(lambda d: isinstance(d, str) and d in posedirs)].copy()
    rows = []; t0 = time.time()
    for j, (dd, g) in enumerate(cov.groupby("dir")):
        for r in g.itertuples(index=False):
            pp = f"{POSE_ROOT}/{dd}/{int(r.frame_idx):05d}.pkl"
            if not os.path.exists(pp):
                continue
            try:
                b = point_features(load_pose(pp))
            except Exception:
                continue
            b.update(video_id=r.video_id, frame_idx=int(r.frame_idx), dir=dd,
                     clip_score_max=r.clip_score_max, text=r.text, child_id=r.child_id)
            rows.append(b)
        if j % 500 == 0:
            print(f"  {j} vids {len(rows)} frames {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df["point_score"] = df.reach * np.clip(df.straight, 0, 1)
    df.to_parquet(f"{W}/manifests/pose_pointing.parquet")
    n = len(df)
    print(f"\n=== POINTING SPARSITY MAP ({n} pose-covered pairs) ===")
    print(f"  adult present (attached body):        {df.has_adult.mean():.1%}")
    for lvl, m in [("L1 extended arm (reach>1.0 torso)", df.reach > 1.0),
                   ("L2 + straight elbow (cos>0.6)", (df.reach > 1.0) & (df.straight > 0.6)),
                   ("L3 strong+straight (reach>1.2)", (df.reach > 1.2) & (df.straight > 0.6)),
                   ("L3b + index extended", (df.reach > 1.0) & (df.straight > 0.6) & (df.index == 1))]:
        print(f"  {lvl:38s} {m.mean():.2%}  (n={int(m.sum())})")
    clip = df.clip_score_max.to_numpy()
    def sp(x):
        mm = ~np.isnan(x); return float(np.corrcoef(pd.Series(x[mm]).rank(), pd.Series(clip[mm]).rank())[0,1])
    print(f"\n  rho(point_score, clip) = {sp(df.point_score.to_numpy(float)):+.3f}")
    print(f"  mean clip | L2 point: {clip[(df.reach>1.0)&(df.straight>0.6)].mean():.3f}  overall: {clip.mean():.3f}")
    print(f"  reach distribution: p50={df.reach.median():.2f} p90={df.reach.quantile(.9):.2f} p99={df.reach.quantile(.99):.2f} max={df.reach.max():.2f}")
    # save strict candidates for a human look (real extended straight arms)
    cand = df[(df.reach > 1.15) & (df.reach < 2.5) & (df.straight > 0.7)].nlargest(24, "point_score")
    cand[["video_id", "frame_idx", "dir", "reach", "straight", "index", "clip_score_max", "text"]].to_csv(f"{W}/runs/point_candidates.csv", index=False)
    print(f"  saved {len(cand)} strict-point candidates -> runs/point_candidates.csv")

if __name__ == "__main__":
    main()
