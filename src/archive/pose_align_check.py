"""Verify the pose<->2025.2 temporal join. The pose pull (feb25) and our CDI detections
(2025.2) both index by second. If the timebases agree, a person's bounding box at second N
in pose should match the person box at frame_idx N in the CDI detections. We compare the
largest-person box across matched seconds, at temporal offsets -3..+3, and check that IoU is
maximised at offset 0 (aligned) rather than shifted (lag)."""
import os, re, ast
import numpy as np, pandas as pd

POSE_CSV = "/ccn2/dataset/babyview/outputs_20250312/pose/4M_frames_old/4M_with_NA_bbox_limbs.csv"
CDI = "/ccn2a/dataset/babyview/2025.2/outputs/object_detections/cdi"
W = "/data2/mcfrank/vlm-headcam"

xw = pd.read_csv(f"{W}/metadata/videos.csv", dtype=str, low_memory=False)
xw.columns = [c.lstrip("﻿") for c in xw.columns]
rec2gcp = dict(zip(xw.unique_video_id, xw.superseded_gcp_name_feb25))

def recid(v):
    m = re.search(r"rec[A-Za-z0-9]+", str(v)); return m.group(0) if m else None

# pick CDI videos that map to a gcp name (candidates)
cdi_vids = os.listdir(CDI)
cand = []
for v in cdi_vids:
    r = recid(v); g = rec2gcp.get(r) if r else None
    if isinstance(g, str) and g:
        cand.append((v, g))
print(f"CDI videos mapping to a gcp name: {len(cand)}")

# load pose person boxes (only needed cols)
pose = pd.read_csv(POSE_CSV, usecols=["superseded_gcp_name_feb25", "time_in_extended_iso",
                                      "person_detected", "person_bounding_box_xyxy"], low_memory=False)
def iso2sec(s):
    p = str(s).split(":")
    try: return int(p[-3]) * 3600 + int(p[-2]) * 60 + int(float(p[-1])) if len(p) >= 3 else int(float(p[-1]))
    except Exception: return None
pose["sec"] = pose.time_in_extended_iso.map(iso2sec)
pose_g = {g: sub for g, sub in pose[pose.person_detected.astype(str) == "1"].groupby("superseded_gcp_name_feb25")}
gcp_have = set(pose_g)
cand = [(v, g) for v, g in cand if g in gcp_have]
print(f"...of which present in pose CSV: {len(cand)}")

def parse_box(s):
    try:
        b = ast.literal_eval(str(s)); return [float(x) for x in b] if len(b) == 4 else None
    except Exception: return None

def area(b): return max(0, b[2]-b[0]) * max(0, b[3]-b[1])
def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1-ix0) * max(0, iy1-iy0); u = area(a) + area(b) - inter
    return inter / u if u > 0 else 0.0

def largest_person_cdi(vid):
    d = pd.read_csv(f"{CDI}/{vid}/bounding_box_predictions.csv")
    d = d[(d.class_name.astype(str) == "person")].dropna(subset=["xmin", "ymin", "xmax", "ymax"])
    out = {}
    for fn, g in d.groupby("frame_number"):
        boxes = g[["xmin", "ymin", "xmax", "ymax"]].to_numpy()
        out[int(fn)] = max((list(b) for b in boxes), key=area)
    return out

def largest_person_pose(sub):
    out = {}
    for sec, g in sub.groupby("sec"):
        boxes = [parse_box(x) for x in g.person_bounding_box_xyxy]; boxes = [b for b in boxes if b]
        if boxes: out[int(sec)] = max(boxes, key=area)
    return out

rng = np.random.default_rng(0)
sample = [cand[i] for i in rng.choice(len(cand), min(6, len(cand)), replace=False)]
offset_iou = {o: [] for o in range(-3, 4)}
for vid, g in sample:
    cdi_b = largest_person_cdi(vid); pose_b = largest_person_pose(pose_g[g])
    common0 = set(cdi_b) & set(pose_b)
    print(f"\n{vid[:40]}... gcp={g[:34]}  cdi_secs={len(cdi_b)} pose_secs={len(pose_b)} overlap@0={len(common0)}")
    for o in offset_iou:
        ious = [iou(cdi_b[s], pose_b[s + o]) for s in cdi_b if (s + o) in pose_b]
        if ious: offset_iou[o].append(np.mean(ious))
print("\n=== mean person-box IoU by temporal offset (pose sec = cdi frame_idx + offset) ===")
for o in sorted(offset_iou):
    v = offset_iou[o]
    print(f"  offset {o:+d}: IoU={np.mean(v):.3f}  (over {len(v)} videos)" if v else f"  offset {o:+d}: (none)")
