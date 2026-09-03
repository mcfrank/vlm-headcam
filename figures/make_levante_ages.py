"""Children's LEVANTE vocabulary accuracy by age -> results/levante_child_by_age.csv

The vocab task is ADAPTIVE: children see a subset of items chosen online near their ability,
so a raw proportion correct over administered trials is not a full-scale accuracy and is not
comparable across ages. Following the LEVANTE-bench paper we impute expected performance on
ALL items from the fitted IRT model: for child ability theta and item intercept d,

    P(correct) = g + (1 - g) * logistic(theta + d),   g = 0.25 (4AFC chance, fixed in the fit)

NB the exported item parameter is mirt's INTERCEPT d, where higher = EASIER (levante-bench
ships scripts/analysis/validate_sign_conventions.py asserting this), not a b-parameter.

Uses the ENGLISH (en-US) language-specific calibration, since the models are trained on
English; run make_levante_ages.R first to refresh the inputs from Redivis. Language/site
differences are substantial -- the Spanish-speaking Colombian sample sits ~8-10 points lower
at every age -- so this choice matters and belongs in the caption.
"""
import numpy as np
import pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
GUESS, MIN_KIDS = 0.25, 8

ip = pd.read_csv(R / "levante_en_item_d.csv")
sc = pd.read_csv(R / "levante_en_scores.csv").dropna(subset=["score", "age"])
ip["w"] = ip.item.str.replace("vocab_word_", "", regex=False)
ours = {w.replace("vocab__", "") for w in pd.read_csv(R / "lev_scaling_final.csv").item.unique()}
d = ip[ip.w.isin(ours)].d.values
print(f"  en-US calibration: {len(ip)} items, {len(d)} matched to the {len(ours)}-item model eval")

sc["age_yr"] = np.floor(sc.age)
sc["imputed"] = [(GUESS + (1 - GUESS) / (1 + np.exp(-(t + d)))).mean() for t in sc.score]
rng = np.random.default_rng(0)
rows = []
for age, g in sc.groupby("age_yr"):
    if len(g) < MIN_KIDS:
        continue
    bs = [g.imputed.sample(len(g), replace=True, random_state=int(s)).mean()
          for s in rng.integers(0, 10 ** 6, 1000)]
    rows.append(dict(age_yr=int(age), language="en-US", acc_macro=g.imputed.mean(),
                     lo=np.percentile(bs, 10), hi=np.percentile(bs, 90),
                     n_children=g.user_id.nunique(), n_items=len(d)))
out = pd.DataFrame(rows)
out.to_csv(R / "levante_child_by_age.csv", index=False)
print(out.round(3).to_string(index=False))
print(f"  total {sc.user_id.nunique()} English-speaking children, ages "
      f"{out.age_yr.min()}-{out.age_yr.max()}")
