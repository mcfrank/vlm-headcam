"""Per-utterance language ID over the whole corpus (fastText lid.176).

Why: the Airtable `percent_english` is a HOUSEHOLD SURVEY, not a description of the recording.
On tape, the eight children the >=80% filter removes are ~93% Latin-script, and survey-vs-tape
correlate only 0.72 — so the filter drops ~120k mostly-English pairs. A script heuristic cannot
help because Spanish/Portuguese/French are Latin too. This labels every utterance so we can
filter at the UTTERANCE level and keep all 36 children.

Caveat carried in the output: 43% of utterances are 1-2 words, where any language ID is weak, so
we record `nwords` and the model's confidence and let downstream apply thresholds.
"""
import os, sys
import numpy as np, pandas as pd, fasttext

MODEL = "/data2/mcfrank/tmp/lid.176.ftz"
OUT = "scored/utt_langid.parquet"
m = fasttext.load_model(MODEL)

G = pd.read_parquet("scored/gemini_full.parquet", columns=["video_id","frame_idx","text","child_id"])
txt = G.text.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
G["nwords"] = txt.str.split().str.len()
clean = txt.str.replace("\n", " ").tolist()
print(f"{len(clean):,} utterances", flush=True)

labs, confs = [], []
B = 100_000
for i in range(0, len(clean), B):
    lab, prob = m.predict(clean[i:i+B], k=1)
    labs += [l[0].replace("__label__", "") for l in lab]
    confs += [float(p[0]) for p in prob]
    if i % 500_000 == 0: print(f"  {i:,}", flush=True)
G["lang"] = labs; G["lang_conf"] = np.round(confs, 4)
G[["video_id","frame_idx","text","child_id","nwords","lang","lang_conf"]].to_parquet(OUT, index=False)
print(f"wrote {OUT}")

print("\n=== corpus-wide ===")
print(G.lang.value_counts(normalize=True).head(8).mul(100).round(2).to_string())
conf = G.lang_conf >= 0.70
print(f"\nconfident (>=0.70): {100*conf.mean():.1f}%   | of those, English: "
      f"{100*(G.loc[conf,'lang']=='en').mean():.1f}%")
print(f"among utterances with >=4 words: confident {100*(G[G.nwords>=4].lang_conf>=0.7).mean():.1f}%, "
      f"English {100*(G[(G.nwords>=4)&conf].lang=='en').mean():.1f}%")

d = pd.read_csv("data/BV-Main Demographics-Grid view.csv", low_memory=False)
d["pe"] = pd.to_numeric(d.percent_english.astype(str).str.rstrip("%"), errors="coerce")
pe = dict(zip(d.subject_id, d.pe))
G["pe"] = G.child_id.map(pe)
# measured English share, restricted to utterances we can actually judge
j = G[(G.nwords >= 4) & (G.lang_conf >= 0.70)]
r = (j.groupby("child_id").agg(pe=("pe","first"), judged=("lang","size"),
                               en=("lang", lambda s: 100*(s=="en").mean())).reset_index())
r = r[r.pe.notna()].sort_values("pe")
print(f"\n{'child':11s} {'survey%':>7s} {'judged':>7s} {'MEASURED %English':>17s}")
for x in r.head(9).itertuples():
    print(f"{x.child_id:11s} {x.pe:7.0f} {x.judged:7,d} {x.en:17.1f}")
print(f"\ncorr(survey, measured) = {r[['pe','en']].corr().iloc[0,1]:.2f}")
lowc = r[r.pe < 80]
print(f"children dropped by >=80%: measured English {lowc.en.mean():.1f}% "
      f"(kept children: {r[r.pe>=80].en.mean():.1f}%)")
