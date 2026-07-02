"""Pose sanity check: (1) numeric validation of keypoint extraction (bounds, face-in-bbox,
plausible limb geometry, score coverage) — safe to inspect without viewing faces; (2)
generate skeleton-overlay JPEGs for a HUMAN to eyeball before we trust the reader or rerun.
Overlays are written to ccn2 only (faces = human subjects; not pulled locally)."""
import os, sys, random
import numpy as np
sys.path.insert(0, "/data2/mcfrank/vlm-headcam/src")
from pose_lib import (load_pose, frame_path, pose_path, POSE_ROOT, FRAME_ROOT, draw_overlay,
                      FACE, LSHO, RSHO, NOSE)

OUT = "/data2/mcfrank/vlm-headcam/runs/pose_sanity"
os.makedirs(OUT, exist_ok=True)
rng = random.Random(0)
dirs = [d for d in os.listdir(POSE_ROOT) if d.endswith("_processed")]
rng.shuffle(dirs)

def inbox(pt, b):
    return b[0] - 5 <= pt[0] <= b[2] + 5 and b[1] - 5 <= pt[1] <= b[3] + 5

n_overlay, n_person, checks = 0, 0, dict(face_in_bbox=[], shoulder_frac=[], kp_cover=[],
                                         x_in=[], y_in=[])
frame_wh = []
for d in dirs:
    if n_overlay >= 10:
        break
    pkls = [f for f in os.listdir(f"{POSE_ROOT}/{d}") if f.endswith(".pkl")]
    rng.shuffle(pkls)
    fr_ok = os.path.isdir(f"{FRAME_ROOT}/{d}")
    for pf in pkls[:40]:
        sec = pf[:-4]
        pose = load_pose(f"{POSE_ROOT}/{d}/{pf}")
        withkp = [P for P in pose["persons"] if P["kp"].shape[0] == 133 and (P["score"] > 0.3).sum() > 20]
        if not withkp or len(pose["bboxes"]) == 0:
            continue
        # frame size (from actual frame if present, else from bbox extent)
        fp = frame_path(d, sec)
        W = H = None
        if fr_ok and os.path.exists(fp):
            try:
                from PIL import Image
                W, H = Image.open(fp).size
            except Exception:
                pass
        for P, b in zip(withkp, pose["bboxes"][:len(withkp)]):
            kp, sc = P["kp"], P["score"]
            vis = sc > 0.3
            n_person += 1
            checks["kp_cover"].append(float(vis.mean()))
            face_pts = kp[[i for i in FACE if vis[i]]]
            if len(face_pts):
                checks["face_in_bbox"].append(float(np.mean([inbox(p, b) for p in face_pts])))
            if vis[LSHO] and vis[RSHO]:
                sw = abs(kp[LSHO, 0] - kp[RSHO, 0]); bw = (b[2] - b[0]) + 1e-6
                checks["shoulder_frac"].append(float(sw / bw))
            if W:
                checks["x_in"].append(float(np.mean((kp[vis, 0] >= -10) & (kp[vis, 0] <= W + 10))))
                checks["y_in"].append(float(np.mean((kp[vis, 1] >= -10) & (kp[vis, 1] <= H + 10))))
        if W:
            frame_wh.append((W, H))
        # write one overlay per selected frame
        out = f"{OUT}/{d[:24]}_{sec}.jpg"
        draw_overlay(fp if (fr_ok and os.path.exists(fp)) else "", pose, out)
        n_overlay += 1
        break

print(f"overlays written: {n_overlay} -> {OUT}")
print(f"persons checked: {n_person}")
print(f"frame sizes seen: {set(frame_wh)}")
def summ(k):
    v = checks[k]; return f"{np.mean(v):.3f} (n={len(v)})" if v else "n/a"
print("--- numeric validation (want ~1.0 for fractions) ---")
print(f"  keypoints visible (score>0.3), mean fraction : {summ('kp_cover')}")
print(f"  face landmarks inside person bbox            : {summ('face_in_bbox')}")
print(f"  keypoint x within frame                      : {summ('x_in')}")
print(f"  keypoint y within frame                      : {summ('y_in')}")
print(f"  shoulder-width / bbox-width (plausible ~0.3-0.9): {summ('shoulder_frac')}")
