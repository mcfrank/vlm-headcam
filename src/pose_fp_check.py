"""Diagnose false-positive skeletons on specific frames: per-person score profile by
body/face/hand, to see if hallucinated bodies (from detector FP boxes) are separable."""
import sys, cv2, numpy as np
sys.path.insert(0, "/data2/mcfrank/vlm-headcam/src")
from pose_infer import make_model, infer
import pose_lib

FRAMES = [("00590001_2024-09-17_1_d1d576f25e_processed", 435),   # skeleton on top of hands
          ("01520001_2024-07-19_2_3b3541510d_processed", 178),   # over-connected phantom
          ("00220001_2024-02-05_1_57f12f7764_processed", 1)]     # a known real person (contrast)

m = make_model("performance")
for gcp, sec in FRAMES:
    img = cv2.imread(f"{pose_lib.FRAME_ROOT}/{gcp}/{sec:05d}.jpg")
    if img is None:
        print(gcp, sec, "NO IMG"); continue
    H, W = img.shape[:2]
    boxes = m.det_model(img)
    kps, scs = infer(m, img)
    print(f"\n{gcp} sec={sec}  imgHW={H}x{W}  det_boxes={len(boxes)}")
    for j in range(len(kps)):
        sc, kp = scs[j], kps[j]
        body, face = sc[:17], sc[23:91]
        hands = np.r_[sc[91:112], sc[112:133]]
        torso = sc[[5, 6, 11, 12]]                       # shoulders + hips
        print(f"  p{j}: body_med={np.median(body):.2f} body>1={ (body>1).sum() }/17  "
              f"torso(sho+hip)>1={ (torso>1).sum() }/4  "
              f"hands_med={np.median(hands):.2f} hands>1={ (hands>1).sum() }/42  "
              f"kpY={kp[:,1].min():.0f}-{kp[:,1].max():.0f}(H={H})")
