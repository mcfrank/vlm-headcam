"""Children's LEVANTE vocabulary accuracy by age -> results/levante_child_by_age.csv

The LEVANTE vocabulary task is ADAPTIVE: children see a subset of items chosen online near
their estimated ability (median 97 of 164 here), so a raw proportion correct over
administered trials is not a full-scale accuracy and is not comparable across ages -- higher
ability children are given harder items, which compresses the age trend badly. Following the
LEVANTE-bench paper, we instead impute expected performance on ALL items from the fitted IRT
model: for child ability theta and item intercept d,

    P(correct) = g + (1 - g) * logistic(theta + d),   g = 0.25 (4AFC chance)

The fitted model is a Rasch (a = 1) with the per-item guessing lower bound fixed at chance;
NB the exported `difficulty` column is mirt's INTERCEPT d, where higher = easier (it
correlates +0.83 with observed item accuracy), not a b-parameter. Expected accuracy is
averaged over the items our models are scored on, then over children within a year of age.
Both the imputed and the (inappropriate) observed macro-average are written, so the
difference is auditable.

IMPORTANT SAMPLE CAVEAT. The vocab task was run at three sites -- MPI-EVA (Germany, n=285),
Uniandes (Colombia, n=195) and Western (Canada, n=127) -- so LEVANTE children are NOT
English-speaking, and the task is administered in each site's language. Ability scores exist
only for the Colombian sample, so the imputed curve describes Spanish-speaking children in
Colombia. Site differences are large (observed macro-average at 12 y: DE 88, CA 79, CO 69),
so this is not a neutral choice; it is simply the only site for which the released IRT
outputs allow a full-scale estimate. Any figure using this must say so.

Machine-specific path: reads the local levante-bench checkout.
"""
import numpy as np
import pandas as pd
from pathlib import Path

B = Path.home() / "Projects/levante-bench/data/responses/v2_0"
R = Path(__file__).resolve().parent.parent / "results"
GUESS, MIN_KIDS = 0.25, 8

ip = pd.read_csv(B / "irt_models/vocab_item_params.csv")
ab = pd.read_csv(B / "irt_models/vocab_ability_scores.csv")
tr = pd.read_csv(B / "tasks/vocab_trials.csv", low_memory=False).drop(columns=["difficulty"])
for f in (ip, tr):
    f["w"] = f.item_uid.str.replace("vocab_word_", "", regex=False)

ours = {w.replace("vocab__", "") for w in pd.read_csv(R / "lev_scaling_final.csv").item.unique()}
d = ip[ip.w.isin(ours)].difficulty.values
P = lambda th: GUESS + (1 - GUESS) / (1 + np.exp(-(th + d)))

# calibration check: does the model reproduce the trials children actually saw?
v = (tr[tr.correct.notna()].merge(ab[["run_id", "ability"]], on="run_id")
       .merge(ip[["w", "difficulty"]], on="w"))
v["pred"] = GUESS + (1 - GUESS) / (1 + np.exp(-(v.ability.values + v.difficulty.values)))
print(f"  calibration on {len(v):,} administered trials: observed {v.correct.mean():.3f} "
      f"vs predicted {v.pred.mean():.3f}")

m = ab.merge(tr.groupby("run_id").age.median(), on="run_id")
m["age_yr"] = np.floor(m.age)
m["imputed"] = [P(t).mean() for t in m.ability]
obs = tr[tr.w.isin(ours) & tr.correct.notna()].copy()
obs["age_yr"] = np.floor(obs.age)

rng = np.random.default_rng(0)
rows = []
for age, g in m.groupby("age_yr"):
    if len(g) < MIN_KIDS:
        continue
    bs = [g.imputed.sample(len(g), replace=True, random_state=int(s)).mean()
          for s in rng.integers(0, 10 ** 6, 1000)]
    o = obs[obs.age_yr == age]
    rows.append(dict(age_yr=int(age), site="pilot_uniandes_co", acc_macro=g.imputed.mean(),
                     lo=np.percentile(bs, 10), hi=np.percentile(bs, 90),
                     n_children=len(g), n_items=len(d),
                     acc_observed_administered=o.groupby("w").correct.mean().mean(),
                     n_children_observed=o.user_id.nunique()))
out = pd.DataFrame(rows)
out.to_csv(R / "levante_child_by_age.csv", index=False)
print(out.round(3).to_string(index=False))
print(f"\n  imputed slope {(100*(out.acc_macro.iloc[-1]-out.acc_macro.iloc[0]))/np.log10(out.age_yr.iloc[-1]/out.age_yr.iloc[0]):.0f}"
      f" vs observed-administered {(100*(out.acc_observed_administered.iloc[-1]-out.acc_observed_administered.iloc[0]))/np.log10(out.age_yr.iloc[-1]/out.age_yr.iloc[0]):.0f} pts/decade")
