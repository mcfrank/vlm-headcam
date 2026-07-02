"""Discourse-continuity cue (FTF-2013's best-F cue, .53), transcript-only so no 1fps gesture
problem. For each utterance, does it CONTINUE the topic of the recent prior utterances
(shared content words)? A pair inside a 'bout' about a present object is more likely aligned.
Test as a training filter vs downstream 4AFC (same decisive design as pose/prosody)."""
import re
import numpy as np, pandas as pd
BV = "/ccn2a/dataset/babyview/2025.2"; W = "/data2/mcfrank/vlm-headcam"; HELD = "S00360001"
STOP = set("a an the this that these those it its is are was were be been being do does did "
           "you your yours i me my we our he she they them his her their to of in on at for "
           "and or but so no not yes yeah ok okay oh um uh hmm well now here there what who "
           "with up down out off go going get got want like look see come came put let s t re "
           "ll ve d m can will just really very too all some any one two do n't don have has".split())
_tok = re.compile(r"[a-z]+")

def content(t):
    return {w for w in _tok.findall(str(t).lower()) if len(w) > 2 and w not in STOP}

fc = pd.read_csv(f"{BV}/outputs/full_clip_results.csv", usecols=[
    "child_id", "video_name", "utterance", "utterance_start_time", "utterance_end_time", "clip_score_max"]).rename(
    columns={"video_name": "video_id", "utterance": "text"})
fc = fc.dropna(subset=["clip_score_max", "utterance_start_time", "utterance_end_time"])
fc = fc[fc.child_id.astype(str) != HELD]
fc["frame_idx"] = ((fc.utterance_start_time + fc.utterance_end_time) / 2).astype(int)
ridx = pd.read_parquet(f"{W}/emb_reg/index.parquet")
have = set(zip(ridx.video_id, ridx.frame_idx.astype(int)))

rows = []
for vid, g in fc.sort_values(["video_id", "utterance_start_time"]).groupby("video_id"):
    g = g.reset_index(drop=True)
    csets = [content(t) for t in g.text]
    starts = g.utterance_start_time.to_numpy()
    for i, r in enumerate(g.itertuples(index=False)):
        if (r.video_id, int(r.frame_idx)) not in have:
            continue
        cur = csets[i]
        # prior utterances within 20s
        prev = set()
        j = i - 1
        while j >= 0 and starts[i] - starts[j] <= 20:
            prev |= csets[j]; j -= 1
        share = len(cur & prev)
        rows.append(dict(video_id=r.video_id, frame_idx=int(r.frame_idx), text=r.text,
                         clip_score_max=r.clip_score_max, child_id=r.child_id,
                         cont_share=share, cont_bin=int(share > 0),
                         n_content=len(cur), cont_jacc=share / (len(cur | prev) + 1e-9)))
df = pd.DataFrame(rows)
df.to_parquet(f"{W}/manifests/discourse_pairs.parquet")
clip = df.clip_score_max.to_numpy()
def sp(x):
    m = ~(np.isnan(x)); return float(np.corrcoef(pd.Series(x[m]).rank(), pd.Series(clip[m]).rank())[0, 1])
print(f"discourse pairs: {len(df)}  cont_bin rate={df.cont_bin.mean():.1%}")
for c in ["cont_share", "cont_bin", "cont_jacc"]:
    print(f"  rho({c}, clip) = {sp(df[c].to_numpy(float)):+.3f}")
print(f"  mean clip | continue=1: {clip[df.cont_bin==1].mean():.3f}  =0: {clip[df.cont_bin==0].mean():.3f}")
