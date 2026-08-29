"""Children's LEVANTE vocabulary accuracy by age, macro-averaged over the same items the
model is scored on -> results/levante_child_by_age.csv.

Reads trial-level LEVANTE responses from the local levante-bench checkout (machine-specific
path); item-level accuracy per age-year, macro-averaged, with a bootstrap-over-children CI.
Ages with < 20 children are dropped (age 13 has 9).
"""
import numpy as np, pandas as pd
from pathlib import Path

TRIALS = Path.home() / "Projects/levante-bench/data/responses/v2_0/tasks/vocab_trials.csv"
R = Path(__file__).resolve().parent.parent / "results"

t = pd.read_csv(TRIALS, low_memory=False)
words = set(pd.read_csv(R / "lev_scaling.csv").word.unique())
t["word"] = t.item_uid.str.replace("vocab_word_", "", regex=False)
t = t[t.word.isin(words) & t.age.notna() & t.correct.notna()].copy()
t["age_yr"] = np.floor(t.age.astype(float))

rng = np.random.default_rng(0)
rows = []
for age, d in t.groupby("age_yr"):
    kids = d.user_id.unique()
    if len(kids) < 20:
        continue
    macro = d.groupby("word").correct.mean().mean()
    boots = []
    for _ in range(500):
        samp = d[d.user_id.isin(rng.choice(kids, len(kids), replace=True))]
        boots.append(samp.groupby("word").correct.mean().mean())
    lo, hi = np.percentile(boots, [10, 90])
    rows.append(dict(age_yr=int(age), acc_macro=macro, lo=lo, hi=hi,
                     n_children=len(kids), n_items=d.word.nunique()))
out = pd.DataFrame(rows)
out.to_csv(R / "levante_child_by_age.csv", index=False)
print(out.round(3).to_string(index=False))
