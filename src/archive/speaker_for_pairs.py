"""Assign a speaker (LENA role) to each training pair by TIME-OVERLAP join (utterance_id
doesn't align across the two transcript files). Caregiver = FEM/MAL; child = KCHI/OCH. Lets us
soft-WEIGHT toward caregiver (child-directed) speech in the bootstrap without hard-filtering
(which would break comparability). Also screens: are caregiver utterances more CLIP-aligned?"""
import numpy as np, pandas as pd
BV = "/ccn2a/dataset/babyview/2025.2/outputs"; W = "/data2/mcfrank/vlm-headcam"; HELD = "S00360001"

mt = pd.read_csv(f"{BV}/merged_transcripts_parsed.csv",
                 usecols=["video_id", "utterance_id", "speaker", "token_start_time", "token_end_time"],
                 low_memory=False)
utt = mt.groupby(["video_id", "utterance_id"]).agg(
    s=("token_start_time", "min"), e=("token_end_time", "max"),
    spk=("speaker", lambda x: x.mode().iat[0] if len(x.mode()) else "unknown")).reset_index()
utt = utt.dropna(subset=["s", "e"])
by_vid = {v: g[["s", "e", "spk"]].to_numpy() for v, g in utt.groupby("video_id")}

fc = pd.read_csv(f"{BV}/full_clip_results.csv", usecols=[
    "child_id", "video_name", "utterance", "utterance_start_time", "utterance_end_time", "clip_score_max"]).rename(
    columns={"video_name": "video_id"})
fc = fc.dropna(subset=["clip_score_max", "utterance_start_time", "utterance_end_time"])
fc = fc[fc.child_id.astype(str) != HELD]
fc["frame_idx"] = ((fc.utterance_start_time + fc.utterance_end_time) / 2).astype(int)
ridx = pd.read_parquet(f"{W}/emb_reg/index.parquet")
have = set(zip(ridx.video_id, ridx.frame_idx.astype(int)))

rows = []
for r in fc.itertuples(index=False):
    if (r.video_id, int(r.frame_idx)) not in have:
        continue
    cand = by_vid.get(r.video_id)
    spk = "unknown"
    if cand is not None:
        ov = np.minimum(r.utterance_end_time, cand[:, 1].astype(float)) - np.maximum(r.utterance_start_time, cand[:, 0].astype(float))
        k = int(np.argmax(ov))
        if ov[k] > 0:
            spk = cand[k, 2]
    rows.append(dict(video_id=r.video_id, frame_idx=int(r.frame_idx), text=r.utterance,
                     clip_score_max=r.clip_score_max, child_id=r.child_id, speaker=spk,
                     is_caregiver=int(spk in ("FEM", "MAL")), is_child=int(spk in ("KCHI", "OCH"))))
df = pd.DataFrame(rows).drop_duplicates(["video_id", "frame_idx"])
df.to_parquet(f"{W}/manifests/speaker_pairs.parquet")
clip = df.clip_score_max.to_numpy()
print(f"speaker pairs: {len(df)}")
print("speaker dist:", df.speaker.value_counts(dropna=False).to_dict())
print(f"caregiver (FEM/MAL) {df.is_caregiver.mean():.1%}  child (KCHI/OCH) {df.is_child.mean():.1%}")
print(f"mean clip | caregiver: {clip[df.is_caregiver==1].mean():.3f}  child: {clip[df.is_child==1].mean():.3f}  overall: {clip.mean():.3f}")
print(f"aligned% | caregiver: {(clip[df.is_caregiver==1]>0.24).mean():.1%}  child: {(clip[df.is_child==1]>0.24).mean():.1%}")
def sp(x):
    return float(np.corrcoef(pd.Series(x).rank(), pd.Series(clip).rank())[0, 1])
print(f"rho(is_caregiver, clip) = {sp(df.is_caregiver.to_numpy(float)):+.3f}")
