"""Second round of alignment-selection controls (post-audit). Derived from the EXISTING draws
in build_aligned_controls.py so the matched sets are byte-identical to the ones already run:
  minusmatch_s{s}   full corpus minus matchrand_s{s}   -> does removing the noun/length-matched
                    UNALIGNED 10% do what removing the aligned 10% does? (the control the text
                    describes; 'minusrand' only removed a uniform random draw)
  rand172k_s{s}     plain random |A| draw              -> the neutral size-matched reference that
                    'matchrand' (unaligned by construction) is not
usage: python src/build_aligned_controls2.py"""
import json
import pandas as pd

COLS = ["video_id", "frame_idx", "text"]
n = json.load(open("results/aligned_control_diagnostics.json"))["n_aligned"]
en = pd.read_parquet("manifests/bv26_pairs_en_audio.parquet")[COLS]
assert len(en) == 1_686_105, len(en)
# the pair key repeats for a few rows (same text twice within one second); the matched draws
# were row samples, so remove ONE corpus row per matched row (rows sharing a key are identical)
en["k"] = en.groupby(COLS).cumcount()
print(f"corpus {len(en):,} | |A| = {n:,} | rows with a repeated key {(en.k > 0).sum():,}")
for s in range(3):
    m = pd.read_parquet(f"manifests/bv26a_matchrand_s{s}.parquet")[COLS]
    # matchrand draws are 54 short of |A|: aligned utterances with zero letter tokens have no
    # length bin and were silently absent from the matching targets (0.03%; not rebuilt)
    assert abs(len(m) - n) <= 100, (len(m), n)
    m["k"] = m.groupby(COLS).cumcount()
    keep = en.merge(m, on=COLS + ["k"], how="left", indicator=True)
    keep = keep[keep._merge == "left_only"][COLS]
    assert len(keep) == len(en) - len(m), f"seed {s}: removed {len(en) - len(keep):,}, expected {len(m):,}"
    keep.to_parquet(f"manifests/bv26a_minusmatch_s{s}.parquet", index=False)
    r = en.sample(n, random_state=7000 + s)[COLS]
    r.to_parquet(f"manifests/bv26a_rand172k_s{s}.parquet", index=False)
    print(f"  s{s}: minusmatch {len(keep):,} | rand172k {len(r):,}")
