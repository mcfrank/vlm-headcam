"""Phase-2 setup: window-frame embedding cost, Topline-1.5 decomposition manifests, and a
CLS-only region cache (so region-MIL degenerates to whole-frame for the -MIL/+EM cell)."""
import os
import re
import shutil
import numpy as np
import pandas as pd
from common import tokenize

tr = pd.read_parquet("manifests/grid_baseline_train.parquet")           # video_id, frame_idx, text
reg = pd.read_parquet("emb_reg/index.parquet")[["video_id", "frame_idx"]]
have = reg.assign(_h=1)

print("=== window-frame embedding cost (train set, 911k pairs) ===")
for w in [2, 3, 5]:
    parts = []
    for d in range(-w, w + 1):
        x = tr[["video_id"]].copy()
        x["frame_idx"] = np.maximum(tr.frame_idx.values + d, 0)
        parts.append(x)
    u = pd.concat(parts, ignore_index=True).drop_duplicates()
    need = u.merge(have, on=["video_id", "frame_idx"], how="left")
    need = int(need._h.isna().sum())
    print(f"  +/-{w}s: {len(u):>9d} unique window frames | {need:>9d} new to embed "
          f"| ~{need * 17 * 768 * 2 / 1e9:.0f} GB region cache, ~{need/1e6*3:.1f}h @ ~100 fps")


def sing(w):
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if re.search(r"(ses|xes|zes|ches|shes)$", w):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


# Topline-1.5 decomposition: referent-bearing pairs in the train split
G = pd.read_parquet("scored/gemini_full.parquet")
G = G[G.alignment.notna()]
trvids = set(tr.video_id.unique())
ref = G[G.video_id.isin(trvids) & (G.referent.fillna("").str.len() > 0)].copy()
cols = ["video_id", "frame_idx", "text"]
# (i) filtered-natural: same pairs, natural utterance
ref[cols].to_parquet("manifests/grid_t15_filtnat_train.parquet", index=False)


# (ii) Topline-1.5: referent word where it is spoken, natural utterance otherwise
def t15_text(row):
    r = sing(str(row.referent).strip().lower())
    toks = set(tokenize(row.text))
    toks |= {sing(t) for t in toks}
    return r if r in toks else row.text


ref15 = ref.copy()
ref15["text"] = ref15.apply(t15_text, axis=1)
spoken = (ref15.text != ref.text).mean()
ref15[cols].to_parquet("manifests/grid_t15_train.parquet", index=False)
print(f"\ntopline-1.5: {len(ref)} referent pairs, {spoken*100:.0f}% relabeled (referent spoken)")

# CLS-only region cache -> region-MIL becomes whole-frame (for the -MIL/+EM EM cell)
os.makedirs("emb_cls1", exist_ok=True)
e = np.load("emb_reg/emb.f16.npy", mmap_mode="r")
np.save("emb_cls1/emb.f16.npy", np.ascontiguousarray(e[:, 0:1, :]))
shutil.copy("emb_reg/index.parquet", "emb_cls1/index.parquet")
print(f"built emb_cls1 {e[:, 0:1, :].shape} (CLS-only, for -MIL/+EM)")
