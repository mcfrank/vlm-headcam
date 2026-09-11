"""Aggregate the human check responses against Gemini -> SI numbers.

Inputs: sample.parquet (from sample.py; has Gemini's score/referent + sampling weights) and the
responses dir (one JSON per rater x item, from the app; pull with gcs_sync.py if deployed on
Cloud Run). Outputs, all aggregate (no frames, no ids, no rater names):
    results/gemini_human_check.csv              long table: metric, value, n, ci_lo, ci_hi
    results/gemini_human_check_calibration.csv  per Gemini-score stratum
    results/gemini_human_check_threshold.csv    precision/recall/F1 of Gemini >= t vs the human majority
A rater table (names, counts, median RT) is printed and written NEXT TO the responses dir, not
into the repo.

Definitions. Ratings are binary since 2026-09-11 (no=0 / yes=100, at Gemini's 50-point anchor:
object visible even if small/partial/one of many); the earlier three-level ratings map
none=0, partial=50, clear=100. cant_tell excluded. Item human score = mean over raters; item
human-aligned = majority of raters said yes (exact ties -> unresolved, excluded from binary
metrics). Gemini-aligned = score >= 50 (the paper's rule); a threshold sweep (>= t for t in
50..100) is written too, since the human judgment is binary and the boundary is the paper's
choice. Weighted precision/recall reweight items by n_pop/n_sample of their stratum, so they
estimate the corpus-level precision/recall. CIs: 1000 item bootstraps.

Usage:
    python human_check/analyze.py --sample /data2/mcfrank/gemini_check/sample.parquet \
        --responses /data2/mcfrank/gemini_check/data/responses --out results
"""
import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

STRATA = ["0_noun", "0_other", "1-49", "50-60", "70-75", "80", "90-95", "100"]


def load_responses(d):
    rows = [json.loads(p.read_text()) for p in Path(d).glob("*/*.json")]
    r = pd.DataFrame(rows)
    return r[["rater", "item_id", "answer", "score", "referent", "rt_ms", "server_ts"]]


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cats = sorted(set(a) | set(b))
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def kripp_alpha(units):
    """Krippendorff's alpha, nominal, units = list of lists of labels (>=2 labels each)."""
    units = [u for u in units if len(u) >= 2]
    cats = sorted({v for u in units for v in u})
    k = {c: i for i, c in enumerate(cats)}
    o = np.zeros((len(cats), len(cats)))
    for u in units:
        m = len(u)
        for i, a in enumerate(u):
            for j, b in enumerate(u):
                if i != j:
                    o[k[a], k[b]] += 1 / (m - 1)
    n = o.sum()
    nc = o.sum(1)
    do = 1 - np.trace(o) / n
    de = 1 - (nc * (nc - 1)).sum() / (n * (n - 1))
    return 1 - do / de if de > 0 else np.nan


def pairwise_kappa(r, col):
    """Mean Cohen's kappa over rater pairs with >= 20 shared items."""
    w = r.pivot_table(index="item_id", columns="rater", values=col, aggfunc="first")
    ks = []
    for a, b in combinations(w.columns, 2):
        both = w[[a, b]].dropna()
        if len(both) >= 20:
            ks.append(cohen_kappa(both[a], both[b]))
    return float(np.nanmean(ks)) if ks else np.nan, len(ks)


def singular(t):
    return t[:-3] + "y" if t.endswith("ies") else t[:-2] if t.endswith("es") else t[:-1] if t.endswith("s") else t


def synonyms(w):
    try:
        from nltk.corpus import wordnet as wn
        return {l.name().lower().replace("_", " ") for s in wn.synsets(w, pos="n") for l in s.lemmas()}
    except Exception:
        return set()


EQUIV = [{"mom", "mommy", "mama", "mother"}, {"dad", "daddy", "papa", "father"}, {"person", "man", "woman", "adult", "dad", "mom", "mommy", "daddy"},
         {"kid", "child", "baby", "girl", "boy", "toddler"}, {"dog", "doggy", "doggie", "puppy"}, {"cat", "kitty", "kitten"},
         {"bunny", "rabbit"}, {"bird", "birdie", "birdies"}, {"tv", "screen", "television", "ipad", "tablet", "phone"},
         {"book", "page"}, {"food", "snack", "meal"}, {"drink", "water", "milk", "juice"}, {"toy", "toys"}]


def ref_match(g, h):
    g, h = g.strip().lower(), h.strip().lower()
    if not g or not h:
        return None
    if g == h:
        return "exact"
    if any(g in e and h in e for e in EQUIV):
        return "lemma"
    gl, hl = singular(g), singular(h)
    if gl == hl or gl in h.split() or hl in g.split():
        return "lemma"
    if h in synonyms(g) or g in synonyms(h) or hl in synonyms(gl) or gl in synonyms(hl):
        return "synonym"
    return "no"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--raters", help="comma-separated rater ids to keep (default all); order defines R1, R2, ...")
    ap.add_argument("--suffix", default="", help="appended to output file names, e.g. _all")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    s = pd.read_parquet(args.sample)
    r = load_responses(args.responses)
    r = r[r.item_id.isin(s.item_id)]
    keep = args.raters.split(",") if args.raters else sorted(r.rater.unique())
    r = r[r.rater.isin(keep)].copy()
    label = {name: f"R{i + 1}" for i, name in enumerate(keep)}
    print("rater labels:", label)
    out = []

    def add(metric, value, n=None, ci=(np.nan, np.nan)):
        out.append(dict(metric=metric, value=round(float(value), 4) if pd.notna(value) else np.nan,
                        n=n, ci_lo=round(float(ci[0]), 4) if pd.notna(ci[0]) else np.nan,
                        ci_hi=round(float(ci[1]), 4) if pd.notna(ci[1]) else np.nan))

    # ---- rater table (not committed) ----
    rt = r.groupby("rater").agg(n=("item_id", "size"), cant_tell=("answer", lambda a: (a == "cant_tell").mean()),
                                median_rt_s=("rt_ms", lambda x: np.nanmedian(x) / 1000))
    print(rt.round(3).to_string())
    rt.to_csv(Path(args.responses).parent / "rater_stats.csv")
    add("n_raters", len(rt)); add("n_ratings", len(r)); add("cant_tell_rate", (r.answer == "cant_tell").mean(), len(r))
    add("median_rt_s", np.nanmedian(r.rt_ms) / 1000, len(r))

    r = r[r.answer != "cant_tell"].copy()
    r["aligned"] = (r.score >= 50).astype(int)

    # ---- inter-rater ----
    r["answer"] = r.answer.replace({"none": "no", "partial": "yes", "clear": "yes"})  # old format -> binary
    units3 = r.groupby("item_id").answer.apply(list).tolist()
    units2 = r.groupby("item_id").aligned.apply(list).tolist()
    n_multi = sum(len(u) >= 2 for u in units3)
    add("kripp_alpha_3level", kripp_alpha(units3), n_multi)
    add("kripp_alpha_binary", kripp_alpha(units2), n_multi)
    k3, np3 = pairwise_kappa(r, "answer"); add("pairwise_kappa_3level", k3, np3)
    k2, np2 = pairwise_kappa(r, "aligned"); add("pairwise_kappa_binary", k2, np2)
    wide = r.pivot_table(index="item_id", columns="rater", values="aligned", aggfunc="first")
    pairs = []
    for a_, b_ in combinations(wide.columns, 2):
        both = wide[[a_, b_]].dropna()
        if len(both) >= 20:
            pairs.append(dict(pair=f"{label[a_]}-{label[b_]}", n_items=len(both), agree=np.mean(both[a_] == both[b_]),
                              kappa=cohen_kappa(both[a_], both[b_]), yes_rate_a=both[a_].mean(), yes_rate_b=both[b_].mean()))
    pairs = pd.DataFrame(pairs).round(4)
    if len(pairs):
        pairs.to_csv(Path(args.out) / f"gemini_human_check_pairs{args.suffix}.csv", index=False)
        print(pairs.to_string(index=False))

    # ---- per item human summary ----
    it = r.groupby("item_id").agg(n_raters=("rater", "size"), human_score=("score", "mean"),
                                  p_aligned=("aligned", "mean")).reset_index()
    it["human_aligned"] = np.where(it.p_aligned > 0.5, 1.0, np.where(it.p_aligned < 0.5, 0.0, np.nan))
    d = s.merge(it, on="item_id")
    d["gemini_aligned"] = (d.gemini_alignment >= 50).astype(int)
    add("n_items_rated", len(d)); add("n_items_ge2_raters", (d.n_raters >= 2).sum())
    add("n_items_tie_excluded", d.human_aligned.isna().sum())
    add("spearman_gemini_vs_human", d[["gemini_alignment", "human_score"]].corr(method="spearman").iloc[0, 1], len(d))

    b = d.dropna(subset=["human_aligned"]).copy()
    b["human_aligned"] = b.human_aligned.astype(int)

    def binary_stats(x):
        g, h, w = x.gemini_aligned.values, x.human_aligned.values, x.weight.values
        tp = (g & h).astype(float)
        p_, r_ = tp.sum() / max(g.sum(), 1), tp.sum() / max(h.sum(), 1)
        wp, wr = (w * tp).sum() / max((w * g).sum(), 1e-9), (w * tp).sum() / max((w * h).sum(), 1e-9)
        return dict(kappa=cohen_kappa(g, h), agree=np.mean(g == h),
                    precision=p_, recall=r_, f1=2 * p_ * r_ / max(p_ + r_, 1e-9),
                    w_precision=wp, w_recall=wr, w_f1=2 * wp * wr / max(wp + wr, 1e-9),
                    w_human_aligned_rate=(w * h).sum() / w.sum(),
                    w_gemini_aligned_rate=(w * g).sum() / w.sum())
    point = binary_stats(b)
    boots = pd.DataFrame([binary_stats(b.sample(frac=1, replace=True, random_state=int(rng.integers(2**31))))
                          for _ in range(args.boot)])
    for k, v in point.items():
        add(f"binary_{k}", v, len(b), (boots[k].quantile(0.025), boots[k].quantile(0.975)))

    # ---- per rater vs Gemini >= 50 ----
    rs = r.merge(s[["item_id", "gemini_alignment", "weight"]], on="item_id")
    rs["gemini_aligned"] = (rs.gemini_alignment >= 50).astype(int)
    for name, x in rs.groupby("rater"):
        x = x.rename(columns={"aligned": "human_aligned"})
        pt = binary_stats(x)
        bs = pd.DataFrame([binary_stats(x.sample(frac=1, replace=True, random_state=int(rng.integers(2**31)))) for _ in range(args.boot)])
        for k in ["precision", "recall", "f1", "w_precision", "w_recall", "w_f1", "agree", "kappa"]:
            add(f"{label[name]}_binary_{k}", pt[k], len(x), (bs[k].quantile(0.025), bs[k].quantile(0.975)))

    # ---- threshold sweep: Gemini >= t vs human majority (and per rater), with item bootstrap CIs ----
    def sweep(x, who):
        rows = []
        for t in [50, 60, 70, 80, 90, 100]:
            def stats(y):
                g, h, w = (y.gemini_alignment >= t).astype(int).values, y.human_aligned.values.astype(int), y.weight.values
                tp = (g & h).astype(float)
                p_, r_ = tp.sum() / max(g.sum(), 1), tp.sum() / max(h.sum(), 1)
                wp, wr = (w * tp).sum() / max((w * g).sum(), 1e-9), (w * tp).sum() / max((w * h).sum(), 1e-9)
                return dict(precision=p_, recall=r_, f1=2 * p_ * r_ / max(p_ + r_, 1e-9),
                            w_precision=wp, w_recall=wr, w_f1=2 * wp * wr / max(wp + wr, 1e-9))
            pt = stats(x)
            bs = pd.DataFrame([stats(x.sample(frac=1, replace=True, random_state=int(rng.integers(2**31)))) for _ in range(min(args.boot, 300))])
            row = dict(who=who, threshold=t, n_items=len(x), n_gemini_pos=int((x.gemini_alignment >= t).sum()),
                       corpus_share=float(s.weight[s.gemini_alignment >= t].sum() / s.weight.sum()))
            for k, v in pt.items():
                row[k], row[k + "_lo"], row[k + "_hi"] = v, bs[k].quantile(0.025), bs[k].quantile(0.975)
            rows.append(row)
        return rows
    rows = sweep(b, "consensus")
    for name, x in rs.groupby("rater"):
        rows += sweep(x.rename(columns={"aligned": "human_aligned"}), label[name])
    thr = pd.DataFrame(rows).round(4)
    thr.to_csv(Path(args.out) / f"gemini_human_check_threshold{args.suffix}.csv", index=False)
    print(thr[thr.who == "consensus"][["threshold", "n_gemini_pos", "precision", "recall", "f1", "w_precision", "w_recall", "w_f1", "corpus_share"]].to_string(index=False))

    # ---- calibration by stratum ----
    cal = (d.groupby("stratum").agg(n_items=("item_id", "size"), gemini_mean=("gemini_alignment", "mean"),
                                    human_mean=("human_score", "mean"), human_aligned=("human_aligned", "mean"),
                                    n_pop=("n_pop", "first"))
           .reindex(STRATA))
    cal["agree_binary"] = b.groupby("stratum").apply(lambda x: np.mean(x.gemini_aligned == x.human_aligned)).reindex(STRATA)
    # human yes-rate per rating (all raters pooled) with Wilson 95% CI, plus per-rater yes-rates
    rs2 = rs.merge(s[["item_id", "stratum"]], on="item_id")
    k_, n_ = rs2.groupby("stratum").aligned.sum().reindex(STRATA), rs2.groupby("stratum").aligned.size().reindex(STRATA)
    z = 1.96; ph = k_ / n_
    cal["n_ratings"], cal["yes_rate"] = n_, ph
    cal["yes_lo"] = (ph + z**2 / (2 * n_) - z * np.sqrt(ph * (1 - ph) / n_ + z**2 / (4 * n_**2))) / (1 + z**2 / n_)
    cal["yes_hi"] = (ph + z**2 / (2 * n_) + z * np.sqrt(ph * (1 - ph) / n_ + z**2 / (4 * n_**2))) / (1 + z**2 / n_)
    for name in keep:
        cal[f"yes_rate_{label[name]}"] = rs2[rs2.rater == name].groupby("stratum").aligned.mean().reindex(STRATA)
    cal.round(4).to_csv(Path(args.out) / f"gemini_human_check_calibration{args.suffix}.csv")
    print(cal.round(3).to_string())
    for st in STRATA:
        if st in cal.index and pd.notna(cal.loc[st, "human_mean"]):
            add(f"human_mean_{st}", cal.loc[st, "human_mean"], int(cal.loc[st, "n_items"]))
            add(f"human_aligned_{st}", cal.loc[st, "human_aligned"], int(cal.loc[st, "n_items"]))
    # false-negative rate among Gemini-0 items with a concrete noun vs without
    for st in ["0_noun", "0_other"]:
        x = b[b.stratum == st]
        if len(x):
            add(f"fn_rate_{st}", x.human_aligned.mean(), len(x))

    # ---- referent accuracy among items both call aligned ----
    rr = r.merge(s[["item_id", "gemini_referent"]], on="item_id").merge(b[["item_id", "human_aligned"]], on="item_id")
    rr = rr[(rr.aligned == 1) & (rr.human_aligned == 1) & (rr.referent.fillna("").str.len() > 0)].copy()  # old 3-level ratings only
    rr["match"] = [ref_match(g, h) for g, h in zip(rr.gemini_referent, rr.referent)]
    add("referent_n_ratings", len(rr))
    for lvl, ok in [("exact", {"exact"}), ("lemma", {"exact", "lemma"}), ("synonym", {"exact", "lemma", "synonym"})]:
        add(f"referent_match_{lvl}", rr.match.isin(ok).mean() if len(rr) else np.nan, len(rr))
    # item-level: majority of raters match at lemma level
    im = rr.assign(ok=rr.match.isin({"exact", "lemma"})).groupby("item_id").ok.mean()
    add("referent_item_majority_lemma", (im > 0.5).mean() if len(im) else np.nan, len(im))
    mism = rr[rr.match == "no"].groupby(["gemini_referent", "referent"]).size().sort_values(ascending=False).head(15)
    print("top referent mismatches (gemini, human):\n", mism.to_string())

    # ---- per-child spread (no ids written) ----
    pc = b.groupby("child_id").apply(lambda x: np.mean(x.gemini_aligned == x.human_aligned))
    pc = pc[b.child_id.value_counts().reindex(pc.index) >= 10]
    add("per_child_agree_n_children", len(pc)); add("per_child_agree_min", pc.min()); add("per_child_agree_max", pc.max())
    add("per_child_agree_sd", pc.std()); add("per_child_agree_below_0.7", (pc < 0.7).sum())

    res = pd.DataFrame(out)
    Path(args.out).mkdir(exist_ok=True, parents=True)
    res.to_csv(Path(args.out) / f"gemini_human_check{args.suffix}.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
