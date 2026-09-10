"""Compute prosody features per training pair (with keys) so prosody can be tested as a
TRAINING FILTER vs downstream 4AFC (the decisive test the pose cues failed). Mirror the pose
filter design: high-emphasis pairs vs random, matched N, clip>0.24 positive control."""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "/data2/mcfrank/vlm-headcam/src")
from cue_audit import load_wav_ffmpeg, prosody_window
BV = "/ccn2a/dataset/babyview/2025.2"; W = "/data2/mcfrank/vlm-headcam"; HELD = "S00360001"

fc = pd.read_csv(f"{BV}/outputs/full_clip_results.csv", usecols=[
    "child_id", "video_name", "utterance", "utterance_start_time", "utterance_end_time", "clip_score_max"]).rename(
    columns={"video_name": "video_id", "utterance": "text"})
fc = fc.dropna(subset=["clip_score_max", "utterance_start_time", "utterance_end_time"])
fc = fc[fc.child_id.astype(str) != HELD]
fc["frame_idx"] = ((fc.utterance_start_time + fc.utterance_end_time) / 2).astype(int)
# keep only pairs with a region embedding (trainable)
ridx = pd.read_parquet(f"{W}/emb_reg/index.parquet")
have = set(zip(ridx.video_id, ridx.frame_idx.astype(int)))
fc = fc[[(v, int(f)) in have for v, f in zip(fc.video_id, fc.frame_idx)]]
print(f"trainable pairs (excl held, in emb_reg): {len(fc)} across {fc.video_id.nunique()} videos", flush=True)

rng = np.random.default_rng(0)
vids = rng.permutation(fc.video_id.unique())
rows = []; t0 = time.time(); sr = 16000
for j, vid in enumerate(vids):
    if len(rows) > 55000:
        break
    child = vid.split("_")[0]
    mp = f"{BV}/mp3/{child}/{vid}.mp3"
    if not os.path.exists(mp):
        continue
    x = load_wav_ffmpeg(mp)
    if x is None:
        continue
    for r in fc[fc.video_id == vid].itertuples(index=False):
        pf = prosody_window(x, sr, r.utterance_start_time, r.utterance_end_time)
        if pf:
            pf.update(video_id=vid, frame_idx=int(r.frame_idx), text=r.text,
                      clip_score_max=r.clip_score_max, child_id=r.child_id)
            rows.append(pf)
    if j % 200 == 0:
        print(f"  {j} vids, {len(rows)} pairs, {time.time()-t0:.0f}s", flush=True)
df = pd.DataFrame(rows).drop_duplicates(["video_id", "frame_idx"])
df.to_parquet(f"{W}/manifests/prosody_pairs.parquet")
# emphasis = combined pitch+energy dynamics (z-scored, per the audit's best cues)
for c in ["f0_range", "rms_range", "rms_cv"]:
    z = (df[c] - df[c].mean()) / (df[c].std() + 1e-9)
    df[c + "_z"] = z
print(f"wrote prosody_pairs: {len(df)} pairs", flush=True)
