"""Per-word prosodic weight = RMS energy over the word's audio span (the stressed word is
referent-likely). Loads each video's mp3 once, computes RMS per transcript token, matches each
pair-word to the nearest-in-time token of the same string, normalizes within the utterance.
Sharded by video (audio decode is the cost): --shard i --nshard N writes pros_shard_i.parquet."""
import argparse
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import librosa
from common import PARSED, BV, tokenize

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
a = ap.parse_args()

cue = pd.read_parquet("manifests/cue_filtnat_train.parquet")
vids = sorted(cue.video_id.unique())
mine = set(vids[a.shard::a.nshard])
cue = cue[cue.video_id.isin(mine)]

tok = pd.read_csv(PARSED, usecols=["video_id", "token", "token_start_time", "token_end_time"])
tok["w"] = tok.token.str.lower().str.strip()
tok = tok[tok.w.str.match(r"^[a-z]+$", na=False) & tok.video_id.isin(mine)].dropna(subset=["token_start_time"])
tok["mid"] = (tok.token_start_time + tok.token_end_time) / 2


def token_rms(vid, tdf):
    path = os.path.join(BV, "mp3", vid.split("_")[0], vid + ".mp3")
    if not os.path.exists(path):
        return None
    try:
        y, sr = librosa.load(path, sr=16000, mono=True)
    except Exception:
        return None
    out = []
    for s, e in zip(tdf.token_start_time.values, tdf.token_end_time.values):
        i, j = int(s * sr), max(int(s * sr) + 1, int(e * sr))
        seg = y[i:j]
        out.append(float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0)
    return out


# per (video, word) -> list of (mid, rms)
wr = defaultdict(list)
for vid, tdf in tok.groupby("video_id"):
    r = token_rms(vid, tdf)
    if r is None:
        continue
    for w, mid, rms in zip(tdf.w.values, tdf.mid.values, r):
        wr[(vid, w)].append((mid, rms))


def pair_pros(row):
    ws = str(row.words).split(); fi = float(row.frame_idx)
    vals = []
    for w in ws:
        cand = wr.get((row.video_id, w), [])
        if cand:
            mid, rms = min(cand, key=lambda x: abs(x[0] - fi))
            vals.append(rms if abs(mid - fi) < 20 else np.nan)
        else:
            vals.append(np.nan)
    v = np.array(vals, float)
    if np.all(np.isnan(v)) or np.nanmax(v) <= 0:
        return " ".join("1.000" for _ in ws)
    v = np.where(np.isnan(v), np.nanmean(v), v)
    v = np.clip(v / np.max(v), 0.1, 1.0)          # normalize: loudest word -> 1
    return " ".join(f"{x:.3f}" for x in v)


cue = cue.copy()
cue["w_pros"] = cue.apply(pair_pros, axis=1)
out = f"manifests/pros_shard_{a.shard}.parquet"
cue[["video_id", "frame_idx", "w_pros"]].to_parquet(out, index=False)
print(f"shard {a.shard}/{a.nshard}: {len(cue)} pairs, {len(mine)} videos -> {out}", flush=True)
