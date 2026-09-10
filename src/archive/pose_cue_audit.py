"""Track A, pose extension: do DIRECTIONAL social cues from whole-body pose carry alignment
information that presence-boxes lacked? For each pose-covered training pair we compute, for
the most prominent (largest) detected person — the caregiver in most egocentric frames —
gaze/face-orientation, pointing/reaching, and showing cues, then audit rho vs held-out CLIP
alignment (same yardstick as the box audit, where hands gave rho~0.005)."""
import os, re, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "/data2/mcfrank/vlm-headcam/src")
from pose_lib import (load_pose, pose_path, POSE_ROOT, NOSE, LEYE, REYE, LEAR, REAR,
                      LSHO, RSHO, LELB, RELB, LWRI, RWRI, LHIP, RHIP, FACE, LHAND, RHAND)

W = "/data2/mcfrank/vlm-headcam"
FW, FH = 512.0, 910.0
TH = 0.3

def recid(v):
    m = re.search(r"rec[A-Za-z0-9]+", str(v)); return m.group(0) if m else None

def cues_for_frame(pose):
    """Cues from the largest detected person. Returns None if no usable person."""
    if len(pose["bboxes"]) == 0 or not pose["persons"]:
        return None
    # largest person that also has keypoints
    best, barea = None, -1
    for P in pose["persons"]:
        i = P["id"]
        if not isinstance(i, int) or i >= len(pose["bboxes"]):
            continue
        b = pose["bboxes"][i]; a = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
        if P["kp"].shape[0] == 133 and a > barea:
            best, barea = (P, b), a
    if best is None:
        return None
    P, b = best; kp, sc = P["kp"], P["score"]; vis = sc > TH
    scale = abs(kp[LSHO, 0]-kp[RSHO, 0]) if vis[LSHO] and vis[RSHO] else (b[2]-b[0])
    scale = max(scale, 1e-3)
    def d(i, j): return float(np.hypot(*(kp[i]-kp[j])))

    c = {}
    c["person_area"] = float(barea / (FW*FH))
    c["face_score"] = float(np.mean(sc[FACE]))                       # face landmarks confident = face toward camera
    c["face_frontal"] = float(vis[LEYE] and vis[REYE] and vis[LEAR] and vis[REAR])
    if vis[LEYE] and vis[REYE] and vis[NOSE]:
        dl, dr = abs(kp[NOSE,0]-kp[LEYE,0]), abs(kp[NOSE,0]-kp[REYE,0])
        c["face_symmetry"] = float(1 - abs(dl-dr)/(dl+dr+1e-6))      # ~1 = looking straight ahead
        eye_y = 0.5*(kp[LEYE,1]+kp[REYE,1])
        c["head_pitch"] = float((kp[NOSE,1]-eye_y)/scale)            # >0 nose below eyes = looking down
    else:
        c["face_symmetry"] = np.nan; c["head_pitch"] = np.nan
    # pointing / reaching: extended arm (wrist far from shoulder, arm straightened)
    reach = []
    for sh, el, wr in [(LSHO, LELB, LWRI), (RSHO, RELB, RWRI)]:
        if vis[sh] and vis[wr]:
            reach.append(d(sh, wr)/scale)
    c["arm_reach"] = float(max(reach)) if reach else np.nan
    # wrist height above hips (showing/holding up)
    hip_y = np.nanmean([kp[LHIP,1] if vis[LHIP] else np.nan, kp[RHIP,1] if vis[RHIP] else np.nan])
    wr_y = [kp[i,1] for i in (LWRI, RWRI) if vis[i]]
    c["wrist_above_hip"] = float((hip_y-min(wr_y))/scale) if wr_y and hip_y==hip_y else np.nan
    # manipulation detail: visible hand keypoints
    c["hand_kp"] = float(sum(vis[i] for i in LHAND+RHAND))
    # hands near view center (manipulated object centered in child's view)
    wr_pts = [kp[i] for i in (LWRI, RWRI) if vis[i]]
    if wr_pts:
        cen = np.mean(wr_pts, axis=0)
        c["hand_centrality"] = float(1 - np.hypot((cen[0]-FW/2)/FW, (cen[1]-FH/2)/FH))
    else:
        c["hand_centrality"] = np.nan
    c["n_person"] = len(pose["bboxes"])
    return c

def main(cap=45000):
    man = pd.read_parquet(f"{W}/manifests/boot_across_140000.parquet")
    xw = pd.read_csv(f"{W}/metadata/videos.csv", dtype=str, low_memory=False)
    xw.columns = [c.lstrip("﻿") for c in xw.columns]
    rec2gcp = dict(zip(xw.unique_video_id, xw.superseded_gcp_name_feb25))
    posedirs = set(os.listdir(POSE_ROOT))
    man["gcp"] = man.video_id.map(lambda v: rec2gcp.get(recid(v)))
    man["dir"] = man.gcp.map(lambda g: f"{g}_processed" if isinstance(g, str) else None)
    cov = man[man.dir.map(lambda dd: isinstance(dd, str) and dd in posedirs)].copy()
    print(f"pose-covered pairs: {len(cov)}/{len(man)} ({len(cov)/len(man):.1%})", flush=True)
    if len(cov) > cap:
        cov = cov.sample(cap, random_state=0)
        print(f"sampling {cap} for the audit", flush=True)

    rows = []; t0 = time.time()
    for j, (dd, g) in enumerate(cov.groupby("dir")):
        for r in g.itertuples(index=False):
            pp = f"{POSE_ROOT}/{dd}/{int(r.frame_idx):05d}.pkl"
            if not os.path.exists(pp):
                continue
            try:
                c = cues_for_frame(load_pose(pp))
            except Exception:
                c = None
            if c:
                c["clip_score_max"] = r.clip_score_max; rows.append(c)
        if j % 400 == 0:
            print(f"  {j} videos, {len(rows)} frames, {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows); df.to_parquet(f"{W}/runs/pose_cue_feats.parquet")
    clip = df.clip_score_max.to_numpy()
    print(f"\n=== POSE DIRECTIONAL CUE AUDIT ({len(df)} frames w/ a usable person) ===")
    print(f"aligned base rate (clip>0.24) = {(clip>0.24).mean():.1%}\n")
    def sp(x):
        m = ~(np.isnan(x)|np.isnan(clip))
        if m.sum() < 50: return np.nan, int(m.sum())
        return float(np.corrcoef(pd.Series(x[m]).rank(), pd.Series(clip[m]).rank())[0,1]), int(m.sum())
    cues = ["face_score","face_frontal","face_symmetry","head_pitch","arm_reach",
            "wrist_above_hip","hand_kp","hand_centrality","person_area","n_person"]
    print(f"{'cue':16s} {'rho':>7s} {'n':>7s}")
    for c in cues:
        if c in df:
            rho, n = sp(df[c].to_numpy(float)); print(f"{c:16s} {rho:7.3f} {n:7d}")
    # combined logistic (reuse simple CV)
    from cue_audit import logistic_cv, precision_at_k
    X = df[cues].to_numpy(float); y = (clip>0.24).astype(float)
    auc, sc = logistic_cv(X, y)
    print(f"\ncombined pose logistic: CV AUC={auc:.3f}  rho={sp(sc)[0]:.3f}  prec@10%={precision_at_k(sc,clip,0.1):.3f}")
    print(f"(reference: box cues combined rho=0.136 AUC=0.539; ignition bar rho~0.3)")

if __name__ == "__main__":
    main()
