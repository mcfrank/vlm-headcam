"""Verify pose<->2025.2 frame indexing WITHOUT a shared detector (CDI and pose are on
disjoint video sets). Each pose dir holds one .pkl per sampled second; our emb_full holds
one row per 1fps frame. If the two pulls share a timebase, per video the second-ranges and
counts should coincide (and match crosswalk duration_sec). A start offset would show as a
min-second mismatch; different trimming as a max/count mismatch."""
import os, re, numpy as np, pandas as pd

POSE = "/ccn2/dataset/babyview/outputs_20250312/pose/4M_frames_old"
W = "/data2/mcfrank/vlm-headcam"
xw = pd.read_csv(f"{W}/metadata/videos.csv", dtype=str, low_memory=False)
xw.columns = [c.lstrip("﻿") for c in xw.columns]
rec2gcp = dict(zip(xw.unique_video_id, xw.superseded_gcp_name_feb25))
rec2dur = dict(zip(xw.unique_video_id, pd.to_numeric(xw.duration_sec, errors="coerce")))

man = pd.read_parquet(f"{W}/manifests/boot_across_140000.parquet")
idx = pd.read_parquet(f"{W}/emb_full/index.parquet")
ours = {v: (g.frame_idx.min(), g.frame_idx.max(), len(g)) for v, g in idx.groupby("video_id")}

def recid(v):
    m = re.search(r"rec[A-Za-z0-9]+", str(v)); return m.group(0) if m else None

posedirs = set(os.listdir(POSE))
vids = man.drop_duplicates("video_id").video_id.tolist()
cand = []
for v in vids:
    r = recid(v); g = rec2gcp.get(r) if r else None
    if isinstance(g, str) and f"{g}_processed" in posedirs and v in ours:
        cand.append((v, r, g))
print(f"manifest videos with pose-on-disk AND our frames: {len(cand)}")

rng = np.random.default_rng(1)
sample = [cand[i] for i in rng.choice(len(cand), min(10, len(cand)), replace=False)]
print(f"\n{'video':32s} {'pose[min,max,n]':>20s} {'ours[min,max,n]':>20s} {'dur':>6s}  match")
dmin, dmax = [], []
for v, r, g in sample:
    pk = [f for f in os.listdir(f"{POSE}/{g}_processed") if f.endswith(".pkl")]
    psec = sorted(int(f[:-4]) for f in pk if f[:-4].isdigit())
    if not psec:
        continue
    pmin, pmax, pn = psec[0], psec[-1], len(psec)
    omin, omax, on = ours[v]
    dur = rec2dur.get(r)
    dmin.append(pmin - omin); dmax.append(pmax - omax)
    ok = "OK" if abs(pmax - omax) <= 2 and abs(pmin - omin) <= 2 else "MISMATCH"
    print(f"{v[:32]:32s} {str([pmin,pmax,pn]):>20s} {str([omin,omax,on]):>20s} {str(dur)[:6]:>6s}  {ok}")
print(f"\nmin-second offset (pose-ours): median={np.median(dmin):.0f}  range=[{min(dmin)},{max(dmin)}]")
print(f"max-second offset (pose-ours): median={np.median(dmax):.0f}  range=[{min(dmax)},{max(dmax)}]")
print("interpretation: offsets ~0 => second N in pose == frame_idx N in ours (join is valid).")
