"""Does a stock YOLO11 person detector avoid the humanart-YOLOX false positives?
Report person boxes + confidences on the FP frames and on benchmark frames labeled
by whether the OLD pipeline saw a person."""
import sys, cv2, numpy as np
sys.path.insert(0, "/data2/mcfrank/vlm-headcam/src")
import pose_lib
from ultralytics import YOLO

det = YOLO("yolo11m.pt")

tests = [("00590001_2024-09-17_1_d1d576f25e_processed", 435, "FP hands"),
         ("01520001_2024-07-19_2_3b3541510d_processed", 178, "FP phantom")]
bg = "00220001_2024-02-05_1_57f12f7764_processed"
for s in [30, 60, 120, 300, 450, 600]:
    try:
        r = pose_lib.load_pose(f"{pose_lib.POSE_ROOT}/{bg}/{s:05d}.pkl")
        has = any((P["score"][:17] > 0.3).sum() >= 8 for P in r["persons"])
    except Exception:
        has = "?"
    tests.append((bg, s, f"bench old_person={has}"))

for g, s, tag in tests:
    im = cv2.imread(f"{pose_lib.FRAME_ROOT}/{g}/{s:05d}.jpg")
    if im is None:
        print(tag, s, "NO IMG"); continue
    r = det(im, classes=[0], conf=0.25, verbose=False)[0]
    confs = sorted(np.round(r.boxes.conf.cpu().numpy(), 2).tolist(), reverse=True)
    print(f"{tag:22s} sec={s}: yolo_persons={len(r.boxes)}  confs={confs}")
