"""Expose YOLOX person-box confidences (rtmlib drops them) to decide whether FP boxes are
low-confidence-and-filterable or genuinely high-confidence (needs a different detector)."""
import sys, cv2, numpy as np
sys.path.insert(0, "/data2/mcfrank/vlm-headcam/src")
from rtmlib import Wholebody
import rtmlib.tools.object_detection.yolox as yx
import pose_lib

cap = []
_orig = yx.multiclass_nms
def _wrap(*a, **k):
    r = _orig(*a, **k)
    cap.append(r[0])          # dets: [:,4] = score
    return r
yx.multiclass_nms = _wrap

m = Wholebody(mode="performance", backend="onnxruntime", device="cuda")
m.det_model.score_thr = 0.05  # low, to reveal all candidate boxes + their scores

FR = [("00590001_2024-09-17_1_d1d576f25e_processed", 435, "FP: skel on hands"),
      ("01520001_2024-07-19_2_3b3541510d_processed", 178, "FP: phantom"),
      ("00220001_2024-02-05_1_57f12f7764_processed", 300, "empty"),
      ("00220001_2024-02-05_1_57f12f7764_processed", 60, "?")]
for g, s, tag in FR:
    cap.clear()
    im = cv2.imread(f"{pose_lib.FRAME_ROOT}/{g}/{s:05d}.jpg")
    if im is None:
        print(g[:12], s, "NO IMG"); continue
    boxes = m.det_model(im)
    dets = cap[-1] if (cap and cap[-1] is not None and len(cap[-1])) else None
    scores = sorted((dets[:, 4]).tolist(), reverse=True)[:5] if dets is not None else []
    print(f"{tag:20s} {g[:10]} sec={s}: kept_boxes={len(boxes)}  "
          f"top_box_scores={[round(x,3) for x in scores]}  (nms_path={'yes' if cap else 'no/baked'})")
