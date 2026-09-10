"""FTF-2013's core claim: no single cue determines reference, but cues COMBINE probabilistically.
Merge every real cue we have (prosody, discourse continuity, box hands/stability, pose), fit a
cross-validated predictor of alignment, and test the COMBINED filter vs downstream 4AFC with a
clip>0.24 positive control. Also report each cue's rho on the common pool."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/data2/mcfrank/vlm-headcam/src")
from cue_audit import logistic_cv
W = "/data2/mcfrank/vlm-headcam"

pro = pd.read_parquet(f"{W}/manifests/prosody_pairs.parquet")
dis = pd.read_parquet(f"{W}/manifests/discourse_pairs.parquet")[["video_id","frame_idx","cont_share","cont_bin","cont_jacc"]]
box = pd.read_parquet(f"{W}/runs/cue_feats.parquet")[["video_id","frame_idx","hand_present","hand_obj_contact","person_area","n_obj","stability"]]
pose = pd.read_parquet(f"{W}/manifests/pose_cues_full.parquet")[["video_id","frame_idx","child_hand","adult_face","adult_reach","child_reach_up"]]
point = pd.read_parquet(f"{W}/manifests/pose_pointing.parquet")[["video_id","frame_idx","point_score","reach","straight"]]
ctr = pd.read_parquet(f"{W}/manifests/frame_center.parquet")[["video_id","frame_idx","center_distinct","center_norm"]]

df = pro.merge(dis, on=["video_id","frame_idx"], how="left")
for extra in (box, pose, point, ctr):
    df = df.merge(extra, on=["video_id","frame_idx"], how="left")
df = df.drop_duplicates(["video_id","frame_idx"]).reset_index(drop=True)
clip = df.clip_score_max.to_numpy()
print(f"common pool: {len(df)} pairs  aligned%={ (clip>0.24).mean():.1%}")
print(f"coverage: discourse={df.cont_share.notna().mean():.0%} box={df.hand_present.notna().mean():.0%} pose={df.child_hand.notna().mean():.0%} center={df.center_distinct.notna().mean():.0%}")

FEATS = ["f0_range","rms_range","rms_cv","cent_std",                 # prosody
         "cont_share","cont_jacc",                                    # discourse
         "center_distinct","center_norm",                            # frame-center (child eyes)
         "point_score","reach","straight",                           # pointing
         "hand_present","hand_obj_contact","person_area","stability", # box
         "child_hand","adult_face","adult_reach","child_reach_up"]   # pose
FEATS = [c for c in FEATS if c in df]
def sp(x):
    m = ~(np.isnan(x)|np.isnan(clip))
    return float(np.corrcoef(pd.Series(x[m]).rank(), pd.Series(clip[m]).rank())[0,1]) if m.sum()>50 else np.nan
print("\nper-cue rho vs clip on common pool:")
for c in FEATS:
    print(f"  {c:16s} {sp(df[c].to_numpy(float)):+.3f}")

X = df[FEATS].to_numpy(float); y = (clip>0.24).astype(float)
auc, oof = logistic_cv(X, y, folds=5, seed=0)
df["combined"] = oof
print(f"\nCOMBINED (CV logistic of all cues): AUC={auc:.3f}  rho(combined,clip)={sp(oof):+.3f}")
print(f"  (best single ~0.15; ignition bar rho~0.3)")

# filter manifests: top-N by each selector vs random vs clip control
clippos = df[clip>0.24]
N = min(len(clippos), 6000)
def topN(col): return df.nlargest(N, col)
sets = {"combined": topN("combined"), "discourse": topN("cont_share"),
        "prosody": topN("rms_range"), "clippos": clippos.sample(N, random_state=0),
        "random": df.sample(N, random_state=0)}
cols = ["video_id","frame_idx","text","clip_score_max","child_id"]
for k,v in sets.items():
    v[cols].reset_index(drop=True).to_parquet(f"{W}/manifests/CB_{k}.parquet")
    print(f"  CB_{k}: N={len(v)} aligned%={(v.clip_score_max>0.24).mean():.1%}")
