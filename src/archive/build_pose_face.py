"""Face / person-presence cue vs Gemini alignment (from the pose CSV). Does a visible caregiver
face — a present, engaged partner — predict referential alignment? (Condition-0, presence axis.)"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

POSE = "/ccn2/dataset/babyview/2025.2/outputs/pose_1fps_bbox_limbs.csv"


def iso_sec(t):
    p = str(t).split(":")
    return int(p[0]) * 3600 + int(p[1]) * 60 + int(float(p[2])) if len(p) == 3 else np.nan


df = pd.read_csv(POSE, usecols=["superseded_gcp_name_feb25", "time_in_extended_iso",
                                "body_bounding_box_size", "face_score", "face_in_image"])
df["video_id"] = df.superseded_gcp_name_feb25
df["frame_idx"] = df.time_in_extended_iso.map(iso_sec)
df["_area"] = pd.to_numeric(df.body_bounding_box_size, errors="coerce")
df = df.dropna(subset=["frame_idx", "_area"]); df["frame_idx"] = df.frame_idx.astype(int)
careg = df.loc[df.groupby(["video_id", "frame_idx"])._area.idxmax()].copy()
careg["face_score"] = pd.to_numeric(careg.face_score, errors="coerce").fillna(0)
pf = careg.groupby(["video_id", "frame_idx"]).agg(face_score=("face_score", "max")).reset_index()

G = pd.read_parquet("scored/gemini_full.parquet")[["video_id", "frame_idx", "alignment"]]
vids = set(pf.video_id.unique())
G = G[G.video_id.isin(vids)]                                    # pose-processed videos only
m = G.merge(pf, on=["video_id", "frame_idx"], how="left")
m["person_present"] = m.face_score.notna().astype(int)          # frame has a detected person
m["face_present"] = (m.face_score.fillna(0) > 0.3).astype(int)  # a visible face
print(f"pose-video pairs: {len(m)}  | person present {m.person_present.mean():.2f} | face present {m.face_present.mean():.2f}")
for cue in ["person_present", "face_present"]:
    rho = spearmanr(m[cue], m.alignment).correlation
    r = rankdata(m[cue]); pos = (m.alignment >= 50).values
    auc = (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())
    m1 = m.alignment[m[cue] == 1].mean(); m0 = m.alignment[m[cue] == 0].mean()
    print(f"  {cue:16s} spearman {rho:+.3f} | AUC {auc:.3f} | mean-align {m1:.1f}(1) vs {m0:.1f}(0)")
