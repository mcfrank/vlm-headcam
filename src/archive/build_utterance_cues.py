"""Unify the utterance-weight cues onto the across-140k boot pairs and add an UNSUPERVISED
combined score (mean of rank-normalised cues — no CLIP, so it's a legitimate bootstrap prior,
unlike the CLIP-trained combined used for the earlier filter screen)."""
import numpy as np, pandas as pd
W = "/data2/mcfrank/vlm-headcam"
man = pd.read_parquet(f"{W}/manifests/boot_across_140000.parquet")[["video_id", "frame_idx", "clip_score_max"]]
def col(path, c):
    d = pd.read_parquet(f"{W}/manifests/{path}")
    return d[["video_id", "frame_idx", c]].drop_duplicates(["video_id", "frame_idx"])
df = man
for path, c in [("prosody_pairs.parquet", "rms_range"), ("speaker_pairs.parquet", "is_caregiver"),
                ("cg_close_pairs.parquet", "cg_close"), ("pose_cues_full.parquet", "child_hand"),
                ("discourse_pairs.parquet", "cont_share")]:
    df = df.merge(col(path, c), on=["video_id", "frame_idx"], how="left")
CUES = ["rms_range", "is_caregiver", "cg_close", "child_hand", "cont_share"]
def rank01(x):
    x = x.astype(float); m = np.nanmedian(x); x = np.where(np.isnan(x), m, x)
    r = pd.Series(x).rank().to_numpy(); return (r - 1) / max(1, len(r) - 1)
R = np.column_stack([rank01(df[c].to_numpy()) for c in CUES])
df["combined"] = R.mean(1)
df.to_parquet(f"{W}/manifests/boot_utterance_cues.parquet")
clip = df.clip_score_max.to_numpy()
print(f"unified utterance cues: {len(df)} pairs")
print("coverage:", {c: f"{df[c].notna().mean():.0%}" for c in CUES})
for c in CUES + ["combined"]:
    x = rank01(df[c].to_numpy()) if c != "combined" else df[c].to_numpy()
    print(f"  rho({c:14s}, clip) = {np.corrcoef(pd.Series(x).rank(), pd.Series(clip).rank())[0,1]:+.3f}")
