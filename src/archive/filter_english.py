"""Filter a pair manifest to children whose home input is predominantly English.

Why this matters: the evaluation is English (Konkle category words, LEVANTE English vocabulary),
but BabyView includes children whose home language is largely or entirely something else. In the
2025.2 training split, 6 of 36 children have median percent_english < 50 (three at 0%), and 749
videos list no English at all. Their utterances enter the bag-of-words vocabulary and the training
signal but can never align to an English eval — so leaving them in silently understates the
learner and muddies "frequency in the model's own input".

Source of truth is the per-CHILD demographics export (`data/BV-Main Demographics-Grid view.csv`,
55 subjects: `subject_id`, `percent_english`, `languages`), which joins straight to our `child_id`
— no rec-id indirection. Falls back to the per-video Airtable export if that file is absent.

Nothing here is a judgement about the children; it is about matching the corpus to an English
eval, and the unfiltered arm is reported alongside as a robustness check.

usage: python src/filter_english.py --in manifests/X.parquet --out manifests/X_en.parquet [--min 80]
"""
import argparse
import os

import pandas as pd

REC = r"(rec[A-Za-z0-9]{10,})"
DEMOG = "data/BV-Main Demographics-Grid view.csv"


def _pct(s):
    return pd.to_numeric(s.astype(str).str.rstrip("%"), errors="coerce")


def english_children(demog_csv, min_pct):
    """-> (set of child_ids to keep, table) from the per-child demographics export."""
    d = pd.read_csv(demog_csv, low_memory=False)
    d["pe"] = _pct(d.percent_english)
    keep = set(d.loc[d.pe >= min_pct, "subject_id"])
    return keep, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--demog-csv", default=DEMOG)
    ap.add_argument("--videos-csv", default="metadata/videos.csv")
    ap.add_argument("--min", type=float, default=80.0, help="minimum percent_english to keep")
    a = ap.parse_args()

    m = pd.read_parquet(a.src)
    kid = m.video_id.str.split("_").str[0]
    if os.path.exists(a.demog_csv):
        keep, _ = english_children(a.demog_csv, a.min)
        sel = kid.isin(keep); how = "per-child demographics"
    else:                                        # fallback: per-video Airtable export
        v = pd.read_csv(a.videos_csv, low_memory=False)
        vk = set(v.loc[_pct(v.percent_english) >= a.min, "unique_video_id"])
        sel = m.video_id.str.extract(REC, expand=False).isin(vk); how = "per-video Airtable"
    out = m[sel].reset_index(drop=True)
    print(f"{a.src}: {len(m):,} pairs, {m.video_id.nunique():,} videos, {kid.nunique()} children")
    print(f"  >= {a.min:g}% English ({how}) -> {len(out):,} pairs "
          f"({100*len(out)/max(len(m),1):.1f}%), {out.video_id.nunique():,} videos, "
          f"{out.video_id.str.split('_').str[0].nunique()} children")
    out.to_parquet(a.out, index=False)
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
