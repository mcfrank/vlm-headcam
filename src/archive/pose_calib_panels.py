"""Calibration panels: same 2025.2 frame + same YOLO12x boxes, drawn two ways —
LEFT the mmpose rtmw-x pkl ([0,1] scores, thr 0.3), RIGHT the rtmlib rtmw-dw-x-l
output ([0,3] scores, thr 1.0). Isolates the pose-model calibration on identical boxes."""
import sys, os, cv2, numpy as np
sys.path.insert(0, "/data2/mcfrank/vlm-headcam/src")
import pose_lib
from pose_compare import draw, label
from pose_infer import make_model, infer

MMOUT = "/data2/mcfrank/pose_calib_out"
OUT = "/data2/mcfrank/vlm-headcam/pose_calib_panels"
os.makedirs(OUT, exist_ok=True)
paths = [l.strip() for l in open("/data2/mcfrank/calib_frames.txt") if l.strip()]
m = make_model(detector="yolo", yolo_conf=0.25, yolo_model="/data2/mcfrank/babyview-pose/yolo12x.pt")

n = 0
for p in paths:
    vid, fn = p.split("/")[-2], p.split("/")[-1]
    pkl = f"{MMOUT}/{vid}/{fn.replace('.jpg', '.pkl')}"
    img = cv2.imread(p)
    if img is None or not os.path.exists(pkl):
        continue
    o = pose_lib.load_pose(pkl)
    mm = [(P["kp"], P["score"]) for P in o["persons"]]
    kps, scs = infer(m, img)
    rl = list(zip(np.asarray(kps, np.float32), np.asarray(scs, np.float32)))
    if len(mm) == 0 and len(rl) == 0:
        continue                      # skip frames with no detections either way
    left = label(draw(img.copy(), mm, 0.3), f"mmpose rtmw-x (calibrated)  n={len(mm)}")
    right = label(draw(img.copy(), rl, 1.0), f"rtmlib dw-x-l  n={len(rl)}")
    cv2.imwrite(f"{OUT}/{vid}_{fn}", np.hstack([left, right]))
    n += 1
print("wrote", n, "panels ->", OUT)
