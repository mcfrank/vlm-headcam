"""From the full pose CSV (per-person, per-frame body-part bboxes), derive a per-frame target
region cell for pose cues: the caregiver's hand (pointing/holding) and face (head/gaze proxy).
Caregiver = largest detected body per frame (egocentric assumption). Region index in the 17-cell
grid (0=CLS, 1..16 = row-major 4x4), matching embed_regions."""
import ast
import numpy as np
import pandas as pd

POSE = "/ccn2/dataset/babyview/2025.2/outputs/pose_1fps_bbox_limbs.csv"
FW, FH = 512.0, 910.0


def cell(cx, cy):
    col = min(3, max(0, int(cx / FW * 4)))
    row = min(3, max(0, int(cy / FH * 4)))
    return 1 + row * 4 + col


def center(s):
    try:
        b = ast.literal_eval(s)
        return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    except Exception:
        return None


def iso_sec(t):
    p = str(t).split(":")
    return int(p[0]) * 3600 + int(p[1]) * 60 + int(float(p[2])) if len(p) == 3 else np.nan


cols = ["superseded_gcp_name_feb25", "time_in_extended_iso", "body_bounding_box_size",
        "left_hand_bounding_box_xyxy", "right_hand_bounding_box_xyxy", "face_bounding_box_xyxy",
        "left_hand_score", "right_hand_score"]
df = pd.read_csv(POSE, usecols=cols)
df["video_id"] = df.superseded_gcp_name_feb25
df["frame_idx"] = df.time_in_extended_iso.map(iso_sec)
df["_area"] = pd.to_numeric(df.body_bounding_box_size, errors="coerce")
df = df.dropna(subset=["frame_idx", "_area"])
df["frame_idx"] = df.frame_idx.astype(int)

# caregiver = largest body per frame
careg = df.loc[df.groupby(["video_id", "frame_idx"])._area.idxmax()].copy()


def hand_cell(r):
    lh = center(r.left_hand_bounding_box_xyxy) if r.left_hand_score > 0.3 else None
    rh = center(r.right_hand_bounding_box_xyxy) if r.right_hand_score > 0.3 else None
    c = rh if (rh and (lh is None or r.right_hand_score >= r.left_hand_score)) else lh
    return cell(*c) if c else -1


careg["hand_cell"] = careg.apply(hand_cell, axis=1)
careg["face_cell"] = careg.face_bounding_box_xyxy.map(lambda s: cell(*center(s)) if center(s) else -1)
out = careg[["video_id", "frame_idx", "hand_cell", "face_cell"]]
out.to_parquet("manifests/pose_targets.parquet", index=False)
print(f"pose targets: {len(out)} frames | hand present {int((out.hand_cell>0).sum())} "
      f"({(out.hand_cell>0).mean()*100:.0f}%) | face present {(out.face_cell>0).mean()*100:.0f}%")
print("  hand-cell distribution:", out[out.hand_cell>0].hand_cell.value_counts().sort_index().to_dict())
