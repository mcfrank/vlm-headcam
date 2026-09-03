"""Put models and children on the SAME 40 Konkle words for fig4A (audit fix).

The CDI-matched words are a strict subset of the 60 test categories (40 on WG, 46 on WS, and
the 40 are inside the 46); the 20 unmatched words (guitar, umbrella, trumpet, rug, ...) are
markedly harder for the models too, so scoring children on 40/46 and models on 60 biased the
comparison against the models by 2-6 points at mid scales.

Writes
  results/wordbank_anchors_40.csv    both CDI forms re-aggregated over the 40 WG-matched words
  results/konkle_wg40_per_seed.csv   per-seed model accuracy over those 40 words, random-
                                     subsample family + full corpus, every encoder (from the
                                     item-level evaluation, results/item_eval_final.csv)
usage: python figures/make_wordbank_40.py"""
import pandas as pd

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
it = pd.read_csv(R / "wordbank_anchors_items.csv")
w40 = sorted(set(it[it.form == "WG"].word))
assert len(w40) == 40 and set(w40) <= set(it[it.form == "WS"].word)

anc = (it[it.word.isin(w40)].groupby(["form", "measure", "age"])
         .agg(n_items=("word", "size"), n_children=("n_admin", "max"), mean_p_know=("p", "mean"),
              pred_4afc=("p", lambda p: 100 * (p + (1 - p) / 4).mean())).reset_index())
assert (anc.n_items == 40).all()
anc.to_csv(R / "wordbank_anchors_40.csv", index=False)

d = pd.read_csv(R / "item_eval_final.csv")
d = d[(d.set == "test60") & d.category.isin(w40)]
# random subsamples (<= 1M) + the full-corpus base runs (ladder family, rung 'base', 'full')
d = d[(d.family == "rand") | ((d.family == "lad") & (d.rung == "base") & (d.scale == "full"))]
assert d.category.nunique() == 40 and (d.scale == "full").any()
d["N"] = d.scale.replace("full", 1686105).astype(int)
per = d.groupby(["encoder", "N", "seed"]).acc.mean().round(3).reset_index()
per.to_csv(R / "konkle_wg40_per_seed.csv", index=False)
print(anc.round(1).to_string(index=False))
print(per.groupby(["encoder", "N"]).acc.agg(["mean", "std", "size"]).round(1).to_string())
