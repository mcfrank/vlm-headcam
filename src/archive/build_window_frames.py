"""Build, for each utterance, the LIST of 1fps frames spanning its window (Vong-style temporal
sampling) instead of a single midpoint. Outputs a frame list to embed and a multiframe manifest
(frames = space-joined frame_idxs) that the trainer samples one-per-epoch from."""
import sys
import numpy as np, pandas as pd
BV = "/ccn2a/dataset/babyview/2025.2/outputs"; W = "/data2/mcfrank/vlm-headcam"
CH = sys.argv[1] if len(sys.argv) > 1 else "S00510002"; K = 8

train_vids = set(pd.read_parquet(f"{W}/manifests/{CH}_train.parquet").video_id.unique())
fc = pd.read_csv(f"{BV}/full_clip_results.csv", usecols=[
    "child_id", "video_name", "utterance", "utterance_start_time", "utterance_end_time", "clip_score_max"]).dropna(
    subset=["clip_score_max", "utterance_start_time", "utterance_end_time"]).rename(columns={"video_name": "video_id", "utterance": "text"})
fc = fc[(fc.child_id == CH) & (fc.video_id.isin(train_vids))].reset_index(drop=True)

def window(a, b):
    fr = list(range(int(np.floor(a)), int(np.ceil(b)) + 1))
    if len(fr) > K:
        fr = [fr[i] for i in np.linspace(0, len(fr) - 1, K).round().astype(int)]
    return sorted(set(fr))

frames_set = set(); rows = []
for r in fc.itertuples(index=False):
    fr = window(r.utterance_start_time, r.utterance_end_time)
    for f in fr:
        frames_set.add((r.video_id, f))
    rows.append(dict(video_id=r.video_id, text=r.text, clip_score_max=r.clip_score_max,
                     child_id=r.child_id, frames=" ".join(map(str, fr))))
mf = pd.DataFrame(rows)
mf.to_parquet(f"{W}/manifests/{CH}_multiframe.parquet")
pd.DataFrame([{"video_id": v, "frame_idx": f} for v, f in frames_set]).to_parquet(f"{W}/manifests/{CH}_windowframes.parquet")
print(f"{CH}: {len(mf)} utterances, {len(frames_set)} unique window frames")
print(f"  wrote {CH}_multiframe.parquet + {CH}_windowframes.parquet")
