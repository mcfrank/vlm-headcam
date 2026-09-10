"""Build the baseline/topline grid manifests off the Gemini-scored full stream, with an
80/20 video-level split (develop on the 80%, hold the 20% untouched for a final check)."""
import re
import numpy as np
import pandas as pd

G = pd.read_parquet("scored/gemini_full.parquet")
G = G[G.alignment.notna()].copy()

# 80/20 split at the VIDEO level (all pairs of a video stay together)
vids = sorted(G.video_id.unique())
rng = np.random.default_rng(0)
rng.shuffle(vids)
ntr = int(0.8 * len(vids))
train_vids, test_vids = set(vids[:ntr]), set(vids[ntr:])
pd.Series(sorted(test_vids)).to_csv("manifests/grid_heldout_videos.txt", index=False, header=False)
tr = G[G.video_id.isin(train_vids)]
print(f"videos {len(vids)} -> train {len(train_vids)} / heldout {len(test_vids)} | train pairs {len(tr)}")

cols = ["video_id", "frame_idx", "text"]

# Baseline: all train pairs, natural utterance
tr[cols].to_parquet("manifests/grid_baseline_train.parquet", index=False)
print(f"baseline: {len(tr)} pairs")

# Topline 1 (oracle filter): Gemini alignment >= threshold
for thr in [50, 70, 80, 90, 100]:
    s = tr[tr.alignment >= thr]
    s[cols].to_parquet(f"manifests/grid_t1_ge{thr}_train.parquet", index=False)
    print(f"topline1 >= {thr}: {len(s)} pairs")


# Topline 2 (oracle label): singularized Gemini referent as the text
def singular(w):
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if re.search(r"(ses|xes|zes|ches|shes)$", w):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


t2 = tr[tr.referent.fillna("").str.len() > 0].copy()
t2["text"] = t2.referent.map(lambda r: singular(str(r).strip().lower()))
t2[cols].to_parquet("manifests/grid_t2_labels_train.parquet", index=False)
print(f"topline2 labels: {len(t2)} pairs, {t2.text.nunique()} distinct label words")
