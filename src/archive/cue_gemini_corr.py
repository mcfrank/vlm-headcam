"""Condition 0: does any child-accessible cue correlate with alignment, using the
cleaner Gemini gold? Reports each cue's Spearman rho vs Gemini alignment AND vs CLIP
(side by side, to see signal CLIP masked), plus AUC for selecting Gemini-aligned pairs."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

G = pd.read_parquet("scored/gemini_full.parquet")
G = G[G.alignment.notna()][["video_id", "frame_idx", "text", "alignment", "clip_score_max"]]
G3 = G.copy()
G2 = (G.sort_values("alignment").drop_duplicates(["video_id", "frame_idx"], keep="last")
      [["video_id", "frame_idx", "alignment", "clip_score_max"]])


def auc(score, pos):
    r = rankdata(score); n1 = pos.sum(); n0 = (~pos).sum()
    return np.nan if n1 == 0 or n0 == 0 else (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# (file, [cue cols], has_text)
SPECS = [
    ("manifests/boot_utterance_cues.parquet", ["rms_range", "combined"], False),        # prosody, combined
    ("manifests/discourse_pairs.parquet", ["cont_share", "cont_jacc"], True),           # discourse
    ("manifests/speaker_pairs.parquet", ["is_caregiver"], True),                        # caregiver
    ("manifests/pose_cues_full.parquet", ["child_hand", "adult_hand", "adult_reach",
                                          "adult_face", "any_hand"], True),             # pose presence
    ("manifests/pose_pointing.parquet", ["reach", "straight", "index"], True),          # pose pointing
]

print(f"{'cue':16s} {'source':16s} {'n':>8s} {'rho_Gemini':>11s} {'rho_CLIP':>9s} {'AUC>=50':>8s}  aligned@1/@0")
for f, cols, has_text in SPECS:
    d = pd.read_parquet(f)
    keys = ["video_id", "frame_idx", "text"] if has_text else ["video_id", "frame_idx"]
    m = d[keys + cols].merge(G3 if has_text else G2, on=keys, how="inner")
    src = f.split("/")[-1].replace(".parquet", "").replace("_pairs", "").replace("_cues_full", "")
    for c in cols:
        s = m[[c, "alignment", "clip_score_max"]].dropna()
        if s[c].nunique() < 2:
            continue
        rg = spearmanr(s[c], s.alignment).correlation
        rc = spearmanr(s[c], s.clip_score_max).correlation
        a = auc(s[c].values, (s.alignment >= 50).values)
        binary = set(s[c].unique()) <= {0, 1, 0.0, 1.0, True, False}
        extra = ""
        if binary:
            m1 = s.alignment[s[c] == 1].mean(); m0 = s.alignment[s[c] == 0].mean()
            extra = f"  {m1:5.1f} / {m0:4.1f}"
        print(f"{c:16s} {src:16s} {len(s):8d} {rg:11.3f} {rc:9.3f} {a:8.3f}{extra}")
