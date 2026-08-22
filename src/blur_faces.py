"""Blur faces in BabyView frames, using the pose annotations we already have.

Every frame in 2025.2 has a pose pickle with 133 COCO-WholeBody keypoints per detected person,
including 68 dense face landmarks — a far better face locator than a generic detector, because it
finds people at odd angles and partial occlusion, and because a head can be localized from
nose/eyes/ears even when the dense face points are low-confidence.

Strategy per person:
  1. dense face landmarks above `kpt_thr`  -> tight face hull, padded
  2. else nose/eyes/ears                   -> head disc sized from the inter-ear/eye span
  3. else the top of the person bbox       -> conservative head band (a person was detected, so
                                              something head-like is there)
Blur is a heavy Gaussian composited through a feathered elliptical mask, so it degrades
gracefully and cannot be undone.

ALWAYS eyeball the output before any frame leaves the cluster: `--report` prints how many faces
were blurred per frame and flags frames where a person was detected but no head could be located.

usage:
  python src/blur_faces.py --video S00220001_..._rec... --frames 435,512 --out scratch/blurred
  python src/blur_faces.py --manifest scratch/gemini_examples/pos.csv --out scratch/blurred
"""
import argparse
import os

import numpy as np
import cv2
import pandas as pd

from common import frame_path
import pose_lib
from pose_lib import FACE, NOSE, LEYE, REYE, LEAR, REAR

POSE_2025_2 = "/ccn2/dataset/babyview/2025.2/outputs/pose_1fps"


def _head_ellipse(kp, sc, thr, bbox):
    """-> (cx, cy, rx, ry) or None. Tries dense face, then head keypoints, then bbox top."""
    f = [i for i in FACE if sc[i] >= thr]
    if len(f) >= 8:
        pts = kp[f]
        (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return cx, cy, (x1 - x0) * 0.78, (y1 - y0) * 0.85
    h = [i for i in (NOSE, LEYE, REYE, LEAR, REAR) if sc[i] >= thr]
    if len(h) >= 2:
        pts = kp[h]
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1.0)
        r = max(span * 1.7, 22.0)
        return cx, cy, r, r * 1.2
    if bbox is not None:                      # person detected, head not localized
        x0, y0, x1, y1 = bbox
        w = x1 - x0
        return (x0 + x1) / 2, y0 + 0.16 * (y1 - y0), w * 0.42, w * 0.46
    return None


def blur_frame(img, pose, kpt_thr=0.3, strength=0.14):
    """Return (blurred image, n_faces, n_fallback). `strength` is blur sigma as a frame fraction."""
    H, W = img.shape[:2]
    k = int(max(H, W) * strength) | 1
    heavy = cv2.GaussianBlur(img, (k, k), 0)
    heavy = cv2.GaussianBlur(heavy, (k, k), 0)          # twice: unrecoverable
    mask = np.zeros((H, W), np.float32)
    n, fallback = 0, 0
    persons = pose.get("persons", []) if pose else []
    bboxes = pose.get("bboxes", []) if pose else []
    for i, p in enumerate(persons):
        bb = bboxes[i] if i < len(bboxes) else None
        e = _head_ellipse(np.asarray(p["kp"]), np.asarray(p["score"]), kpt_thr, bb)
        if e is None:
            continue
        cx, cy, rx, ry = e
        used_bbox = not any(np.asarray(p["score"])[j] >= kpt_thr for j in (NOSE, LEYE, REYE, LEAR, REAR))
        fallback += int(used_bbox)
        cv2.ellipse(mask, (int(cx), int(cy)), (int(max(rx, 12)), int(max(ry, 12))),
                    0, 0, 360, 1.0, -1)
        n += 1
    if n:
        f = int(max(H, W) * 0.03) | 1
        mask = cv2.GaussianBlur(mask, (f, f), 0)        # feather so the edge is not a hard oval
        m3 = np.repeat(mask[:, :, None], 3, axis=2)
        img = (img * (1 - m3) + heavy * m3).astype(np.uint8)
    return img, n, fallback


def load_pose_for(video_id, frame_idx, root=POSE_2025_2):
    p = f"{root}/{video_id}/{int(frame_idx):05d}.pkl"
    if not os.path.exists(p):
        return None
    try:
        return pose_lib.load_pose(p)
    except Exception as e:
        print(f"  pose read failed {video_id}/{frame_idx}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video"); ap.add_argument("--frames")
    ap.add_argument("--manifest", help="csv with video_id, frame_idx columns")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pose-root", default=POSE_2025_2)
    ap.add_argument("--kpt-thr", type=float, default=0.3)
    ap.add_argument("--strength", type=float, default=0.14)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.manifest:
        rows = pd.read_csv(a.manifest)[["video_id", "frame_idx"]].values.tolist()
    else:
        rows = [(a.video, int(f)) for f in a.frames.split(",")]

    print(f"{'video/frame':52s} {'persons':>7s} {'blurred':>7s} {'bbox-fallback':>13s}")
    for vid, fi in rows:
        img = cv2.imread(str(frame_path(vid, fi)))
        if img is None:
            print(f"  MISSING FRAME {vid}/{fi}"); continue
        pose = load_pose_for(vid, fi, a.pose_root)
        np_ = len(pose.get("persons", [])) if pose else 0
        out, n, fb = blur_frame(img, pose, a.kpt_thr, a.strength)
        dst = f"{a.out}/{vid}_{int(fi):05d}.jpg"
        cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        flag = "  <-- CHECK (person found, head guessed)" if fb else ("" if n or np_ == 0 else "  <-- CHECK (person, no head)")
        print(f"{vid[:40]}/{fi:<6d}{'':6s} {np_:7d} {n:7d} {fb:13d}{flag}")
    print(f"\nwrote -> {a.out}\nALWAYS eyeball these before anything leaves the cluster.")


if __name__ == "__main__":
    main()
