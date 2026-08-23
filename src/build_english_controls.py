"""Controls that separate WHY the English-filtered arm underperforms.

Observed: English >=80% keeps 786,288 of 904,812 pairs (28 of 36 children) and scores 67.5,
which is 3.3 pts BELOW the random-subsample scaling curve at the same pair count. The prediction
was the opposite — non-English speech should be worth LESS on an English benchmark, so removing
it should have left us on or above the curve.

Two confounds are baked into that comparison:
  (a) QUANTITY   — 13% fewer pairs
  (b) DIVERSITY  — 8 fewer children, and ch6 says diversity is worth ~+7 at matched count

So build matched-count controls:
  ctl_qty    786,288 pairs drawn at random from all 36 children   -> isolates (a)
  ctl_kids   786,288 pairs after dropping 8 RANDOM children       -> isolates (a)+(b)
Then:  english ~= ctl_kids  -> the loss is diversity, not language
       english <  ctl_kids  -> non-English pairs were genuinely useful (surprising)
       english >  ctl_kids  -> removing non-English helped, as predicted

usage: python src/build_english_controls.py [--seeds 0,1,2]
"""
import argparse
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", default="manifests/grid_baseline_train.parquet")
ap.add_argument("--english", default="manifests/grid_baseline_train_en.parquet")
ap.add_argument("--seeds", default="0,1,2", help="one child-drop draw per seed")
a = ap.parse_args()

full = pd.read_parquet(a.pairs)
eng = pd.read_parquet(a.english)
full["child"] = full.video_id.str.split("_").str[0]
N, K = len(eng), eng.video_id.str.split("_").str[0].nunique()
kids = sorted(full.child.unique())
n_drop = len(kids) - K
print(f"full {len(full):,} pairs / {len(kids)} children")
print(f"english target: {N:,} pairs / {K} children  (drop {n_drop} children)")

# how much of the REMOVED data comes from majority-English children?
removed = full[~full.index.isin(eng.index)] if full.index.equals(eng.index.union(full.index)) else None
d = pd.read_csv("data/BV-Main Demographics-Grid view.csv", low_memory=False)
d["pe"] = pd.to_numeric(d.percent_english.astype(str).str.rstrip("%"), errors="coerce")
excl = d[d.pe < 80][["subject_id", "pe"]]
sub = full[full.child.isin(set(excl.subject_id))]
if len(sub):
    per = sub.groupby("child").size().to_frame("pairs").join(excl.set_index("subject_id"))
    per["english_pairs_est"] = (per.pairs * per.pe / 100).round().astype(int)
    print(f"\nexcluded children contribute {len(sub):,} pairs, of which "
          f"~{per.english_pairs_est.sum():,} are estimated ENGLISH:")
    print(per.sort_values("pairs", ascending=False).to_string())

# ---- ctl_qty: random pairs, all children ---------------------------------------
full.sample(N, random_state=0)[["video_id", "frame_idx", "text"]].to_parquet(
    "manifests/ctl_qty.parquet", index=False)
print(f"\nwrote manifests/ctl_qty.parquet  {N:,} pairs / {len(kids)} children")

# ---- ctl_kids: drop n_drop random children, then match the pair count ----------
for s in [int(x) for x in a.seeds.split(",")]:
    rng = np.random.default_rng(100 + s)
    for _ in range(200):                       # find a drop set that leaves >= N pairs
        drop = set(rng.choice(kids, n_drop, replace=False))
        keep = full[~full.child.isin(drop)]
        if len(keep) >= N:
            break
    out = keep.sample(N, random_state=s)
    out[["video_id", "frame_idx", "text"]].to_parquet(f"manifests/ctl_kids_s{s}.parquet", index=False)
    print(f"wrote manifests/ctl_kids_s{s}.parquet  {len(out):,} pairs / "
          f"{out.video_id.str.split('_').str[0].nunique()} children  (dropped {sorted(drop)})")

# ---- follow-ups, added after the first controls came back -----------------------
# ctl_qty 72.0 and ctl_kids 71.8 vs english 67.5: neither pair count nor child count
# explains the gap, so it is WHICH children the language filter removes. The prime suspect
# is S00240001 — 61,002 pairs (half of everything removed) and 68% English, i.e. mostly
# usable data that the >=80% threshold discards.
if __name__ == "__main__" and True:
    N = len(eng)
    # (1) a >=50% threshold keeps S00240001 (68%) and S00680001 (50%)
    keep50 = set(d.loc[d.pe >= 50, "subject_id"])
    e50 = full[full.child.isin(keep50)]
    print(f"\n>=50% English: {len(e50):,} pairs / {e50.child.nunique()} children")
    e50[["video_id", "frame_idx", "text"]].to_parquet("manifests/en50.parquet", index=False)
    # matched-count version so it is comparable to the others
    if len(e50) > N:
        e50.sample(N, random_state=0)[["video_id", "frame_idx", "text"]].to_parquet(
            "manifests/en50_matched.parquet", index=False)
        print(f"  matched to {N:,} -> manifests/en50_matched.parquet")
    # (2) drop ONLY the big 68%-English child, matched count
    big = full[full.child != "S00240001"]
    if len(big) >= N:
        big.sample(N, random_state=0)[["video_id", "frame_idx", "text"]].to_parquet(
            "manifests/ctl_drop_big.parquet", index=False)
        print(f"drop S00240001 only, matched to {N:,} / {big.child.nunique()} children "
              f"-> manifests/ctl_drop_big.parquet")
