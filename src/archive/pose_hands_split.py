"""Sharper cue v1: split hands into CHILD's-own (orphan hands at frame bottom, no attached
face/torso) vs ADULT (hands on a person with a visible face/torso), and audit each SEPARATELY
vs CLIP. The box audit lumped them (rho~0.005); the hypothesis is they carry opposite-signed
signal. Cheap screen: rho vs clip_score_max. A promising cue graduates to the real 4AFC gate."""
import os, re, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "/data2/mcfrank/vlm-headcam/src")
from pose_lib import (load_pose, POSE_ROOT, NOSE, LEYE, REYE, LEAR, REAR, LSHO, RSHO,
                      LELB, RELB, LWRI, RWRI, LHIP, RHIP, LHAND, RHAND)
W = "/data2/mcfrank/vlm-headcam"; FW, FH = 512.0, 910.0; TH = 0.3

def recid(v):
    m = re.search(r"rec[A-Za-z0-9]+", str(v)); return m.group(0) if m else None

def classify(P, b):
    kp, sc = P["kp"], P["score"]; vis = sc > TH
    face = float(np.mean(sc[[NOSE, LEYE, REYE, LEAR, REAR]]))
    torso = float(np.mean(sc[[LSHO, RSHO, LHIP, RHIP]]))
    hand = float(max(np.mean(sc[LHAND]), np.mean(sc[RHAND])))
    wr = [kp[i] for i in (LWRI, RWRI) if vis[i]]
    if hand < 0.3 and not wr:
        return None
    wy = np.mean([p[1] for p in wr]) / FH if wr else np.nan
    kind = "child" if (face < 0.15 and torso < 0.2 and (wy != wy or wy > 0.6) and b[3] > 0.9*FH) \
        else ("adult" if (face > 0.3 or torso > 0.3) else "other")
    scale = abs(kp[LSHO,0]-kp[RSHO,0]) if vis[LSHO] and vis[RSHO] else (b[2]-b[0])
    reach = max([np.hypot(*(kp[wri]-kp[sh])) / max(scale,1e-3)
                 for sh, wri in [(LSHO,LWRI),(RSHO,RWRI)] if vis[sh] and vis[wri]], default=np.nan)
    top_wy = min([p[1] for p in wr]) / FH if wr else np.nan   # highest wrist (small = high up)
    return dict(kind=kind, face=face, reach=reach, top_wy=top_wy, wy=wy, hand=hand)

def frame_cues(pose):
    ppl = []
    for P in pose["persons"]:
        i = P["id"]
        if isinstance(i, int) and i < len(pose["bboxes"]) and P["kp"].shape[0] == 133:
            c = classify(P, pose["bboxes"][i])
            if c: ppl.append(c)
    ch = [p for p in ppl if p["kind"] == "child"]; ad = [p for p in ppl if p["kind"] == "adult"]
    return dict(
        child_hand=int(len(ch) > 0),
        child_reach_up=float(np.nanmax([1-p["top_wy"] for p in ch])) if ch else 0.0,  # child lifting hands up
        adult_hand=int(len(ad) > 0),
        adult_reach=float(np.nanmax([p["reach"] for p in ad])) if ad else np.nan,     # adult arm extension
        adult_hand_up=float(np.nanmax([1-p["top_wy"] for p in ad])) if ad else np.nan,
        adult_face=int(any(p["face"] > 0.3 for p in ad)),
        any_hand=int(len(ppl) > 0),
    )

def main(cap=45000):
    man = pd.read_parquet(f"{W}/manifests/boot_across_140000.parquet")
    xw = pd.read_csv(f"{W}/metadata/videos.csv", dtype=str, low_memory=False)
    xw.columns = [c.lstrip("﻿") for c in xw.columns]
    rec2gcp = dict(zip(xw.unique_video_id, xw.superseded_gcp_name_feb25))
    posedirs = set(os.listdir(POSE_ROOT))
    man["dir"] = man.video_id.map(lambda v: (lambda g: f"{g}_processed" if isinstance(g, str) else None)(rec2gcp.get(recid(v))))
    cov = man[man.dir.map(lambda d: isinstance(d, str) and d in posedirs)].copy()
    if len(cov) > cap:
        cov = cov.sample(cap, random_state=0)
    rows = []; t0 = time.time()
    for j, (dd, g) in enumerate(cov.groupby("dir")):
        for r in g.itertuples(index=False):
            pp = f"{POSE_ROOT}/{dd}/{int(r.frame_idx):05d}.pkl"
            if not os.path.exists(pp): continue
            try: c = frame_cues(load_pose(pp))
            except Exception: continue
            c["clip"] = r.clip_score_max; rows.append(c)
        if j % 500 == 0: print(f"  {j} vids {len(rows)} frames {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows); df.to_parquet(f"{W}/runs/pose_hands_split.parquet")
    clip = df.clip.to_numpy()
    print(f"\n=== CHILD vs ADULT HAND AUDIT ({len(df)} frames) ===")
    print(f"child_hand present: {df.child_hand.mean():.1%}  adult_hand: {df.adult_hand.mean():.1%}  base rate clip>0.24: {(clip>0.24).mean():.1%}\n")
    def sp(x):
        m = ~(np.isnan(x)|np.isnan(clip))
        return (float(np.corrcoef(pd.Series(x[m]).rank(), pd.Series(clip[m]).rank())[0,1]), int(m.sum())) if m.sum()>50 else (np.nan,0)
    for c in ["child_hand","child_reach_up","adult_hand","adult_reach","adult_hand_up","adult_face","any_hand"]:
        rho,n = sp(df[c].to_numpy(float)); print(f"  {c:16s} rho={rho:+.3f}  n={n}")
    # conditional means: does clip differ by child-hand presence?
    print(f"\n  mean clip | child_hand=1: {clip[df.child_hand==1].mean():.3f}  =0: {clip[df.child_hand==0].mean():.3f}")
    print(f"  mean clip | adult_hand=1: {clip[df.adult_hand==1].mean():.3f}  =0: {clip[df.adult_hand==0].mean():.3f}")
    print("(box audit lumped all hands -> rho 0.005; ignition bar rho~0.3; CLIP is a coarse yardstick)")

if __name__ == "__main__":
    main()
