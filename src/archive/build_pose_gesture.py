"""Sharper pose cue: caregiver 'teaching gesture' (extended-arm pointing / raised-hand showing),
from the full pose CSV bboxes. Does it predict Gemini alignment better than mere hand presence
(Condition 0)? Prerequisite for pose-as-utterance-filter. Prints correlation + selection AUC."""
import ast
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

POSE = "/ccn2/dataset/babyview/2025.2/outputs/pose_1fps_bbox_limbs.csv"


def ctr(s):
    try:
        b = ast.literal_eval(s); return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0, (b[2] - b[0]), (b[3] - b[1])
    except Exception:
        return None


def iso_sec(t):
    p = str(t).split(":")
    return int(p[0]) * 3600 + int(p[1]) * 60 + int(float(p[2])) if len(p) == 3 else np.nan


cols = ["superseded_gcp_name_feb25", "time_in_extended_iso", "body_bounding_box_size",
        "body_bounding_box_xyxy", "left_hand_bounding_box_xyxy", "right_hand_bounding_box_xyxy",
        "left_hand_score", "right_hand_score"]
df = pd.read_csv(POSE, usecols=cols)
df["video_id"] = df.superseded_gcp_name_feb25
df["frame_idx"] = df.time_in_extended_iso.map(iso_sec)
df["_area"] = pd.to_numeric(df.body_bounding_box_size, errors="coerce")
df = df.dropna(subset=["frame_idx", "_area"]); df["frame_idx"] = df.frame_idx.astype(int)
careg = df.loc[df.groupby(["video_id", "frame_idx"])._area.idxmax()].copy()


def gesture(r):
    body = ctr(r.body_bounding_box_xyxy)
    if not body:
        return 0.0
    bx, by, bw, bh = body
    best = 0.0
    for hb, sc in [(r.left_hand_bounding_box_xyxy, r.left_hand_score),
                   (r.right_hand_bounding_box_xyxy, r.right_hand_score)]:
        if sc is None or sc < 0.3:
            continue
        h = ctr(hb)
        if not h:
            continue
        extend = abs(h[0] - bx) / (bw + 1e-6)             # horizontal reach from body (pointing)
        raise_ = max(0.0, (by - h[1]) / (bh + 1e-6))      # hand above body center (showing)
        best = max(best, float(sc) * (extend + raise_))
    return best


careg["gest"] = careg.apply(gesture, axis=1)
pose = careg[["video_id", "frame_idx", "gest"]]

G = pd.read_parquet("scored/gemini_full.parquet")[["video_id", "frame_idx", "alignment"]]
m = G.merge(pose, on=["video_id", "frame_idx"], how="inner").dropna(subset=["gest", "alignment"])
print(f"matched {len(m)} pairs")
rho = spearmanr(m.gest, m.alignment).correlation
r = rankdata(m.gest); pos = (m.alignment >= 50).values
auc = (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())
print(f"gesture vs Gemini alignment: spearman {rho:.3f} | AUC(select >=50) {auc:.3f}")
q = pd.qcut(m.gest, 4, labels=False, duplicates="drop")
print("mean alignment by gesture quartile:", m.groupby(q).alignment.mean().round(1).tolist())
pose.to_parquet("manifests/pose_gesture.parquet", index=False)
