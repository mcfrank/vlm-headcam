"""Throwaway: (1) confirm the detector score_thr actually changes detection counts,
(2) per-group keypoint-score percentiles to calibrate kpt_thr, (3) an overlay grid
across (det_thr, kpt_thr) for human review. Not committed."""
import glob, cv2, numpy as np
from pathlib import Path
from rtmlib import Wholebody, draw_skeleton

FR = "/ccn2a/dataset/babyview/2025.2/extracted_frames_1fps/S00220001_2024-02-05_1_recfcw2yqs02gdskT"
paths = sorted(glob.glob(FR + "/*.jpg"))[:200]
m = Wholebody(mode="performance", backend="onnxruntime", device="cuda")

# (1) does score_thr change counts?
print("=== detector threshold sweep (persons/frame) ===")
for dt in [0.3, 0.5, 0.7, 0.9, 0.97]:
    m.det_model.score_thr = dt
    tot = nf = 0
    for p in paths:
        k, s = m(cv2.imread(p)); tot += len(k); nf += 1
    print(f"  det_thr={dt}: {tot/nf:.2f} persons/frame")

# (2) per-group score percentiles (at det_thr 0.7)
m.det_model.score_thr = 0.7
B, Fc, H = [], [], []
for p in paths:
    k, s = m(cv2.imread(p))
    for j in range(len(k)):
        B.append(s[j][:17]); Fc.append(s[j][23:91]); H.append(s[j][91:133])
print("=== score percentiles 10/50/90 by group (det_thr 0.7) ===")
for name, g in [("body", B), ("face", Fc), ("hand", H)]:
    a = np.concatenate(g) if g else np.array([0.])
    print(f"  {name}: {np.round(np.percentile(a,[10,50,90]),2)}")

# (3) overlay grid for review
outroot = Path("/data2/mcfrank/vlm-headcam/pose_grid")
for dt in [0.7, 0.9]:
    m.det_model.score_thr = dt
    for kt in [0.6, 1.2, 1.8]:
        od = outroot / f"det{dt}_kpt{kt}"; od.mkdir(parents=True, exist_ok=True)
        saved = i = 0
        while saved < 6 and i < len(paths):
            im = cv2.imread(paths[i]); k, s = m(im)
            if len(k):
                cv2.imwrite(str(od / Path(paths[i]).name),
                            draw_skeleton(im.copy(), k, s, kpt_thr=kt))
                saved += 1
            i += 1
print("grid ->", outroot)
