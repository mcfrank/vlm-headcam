"""The 3x2 encoder grid (S/B/L x OTS/BV) in one table: Konkle scaling points, aligned, ladder
ceiling, alignment-removal, no-MIL, window, LEVANTE (fair), in-domain word learning.
Sources: results/runs.parquet (best_acc = dev-selected test accuracy, what the figures use),
results/lev_scaling_final.csv, results/indomain_eval.csv.
Writes results/encoder_grid.csv and prints a markdown table (paper labels by parameter count).
usage: python src/make_encoder_grid.py"""
import numpy as np
import pandas as pd

R = "results"
ENC = [("dinov3s", "OTS", "22M", "S-OTS"), ("dinov3b", "OTS", "86M", "B-OTS"), ("dinov3l", "OTS", "304M", "L-OTS"),
       ("vits_bv", "BV", "22M", "S-BV"), ("vitb_bv", "BV", "86M", "B-BV"), ("vitl_bv", "BV", "304M", "L-BV")]
ROWS = [("Konkle 30k", "rand_30000"), ("Konkle 100k", "rand_100000"), ("Konkle 300k", "rand_300000"),
        ("Konkle 1M", "rand_1000000"), ("Konkle full", "base"), ("aligned 170k", "align_170000"),
        ("ladder: clean label", "lad_t2"), ("aligned-only 172k", "alignedonly"), ("full − aligned", "minusaligned"),
        ("full − matched", "minusmatch"), ("no-MIL full", "wffull"), ("window full", "win5_full")]
runs = pd.read_parquet(f"{R}/runs.parquet")
lev = pd.read_csv(f"{R}/lev_scaling_final.csv")
ind = pd.read_csv(f"{R}/indomain_eval.csv")


def cell(vals):
    vals = [v for v in vals if pd.notna(v)]
    if not vals:
        return np.nan, np.nan, 0
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, len(vals)


out = []
for tag, regime, params, lbl in ENC:
    for name, cond in ROWS:
        m, s, n = cell(runs[runs.family == f"F_{tag}_{cond}"].best_acc.tolist())
        out.append(dict(encoder=tag, label=lbl, regime=regime, params=params, measure=name, mean=m, sd=s, n=n))
    L = lev[(lev.encoder == lbl) & (lev.N == 1686105)].copy()
    if len(L):
        L["fair"] = np.where(L.playable, L.correct, 0.25)
        per = L.groupby("seed").fair.mean().mul(100)
        m, s, n = cell(per.tolist())
    else:
        m, s, n = np.nan, np.nan, 0
    out.append(dict(encoder=tag, label=lbl, regime=regime, params=params, measure="LEVANTE full (fair)", mean=m, sd=s, n=n))
    m, s, n = cell(ind[(ind.encoder == tag) & (ind.N == 1686105)].acc.tolist())
    out.append(dict(encoder=tag, label=lbl, regime=regime, params=params, measure="in-domain full", mean=m, sd=s, n=n))
G = pd.DataFrame(out)
G.to_csv(f"{R}/encoder_grid.csv", index=False)

order = [f"{r}-{p}" for _, r, p, _ in ENC]
G["col"] = G.regime + "-" + G.params
T = G.pivot(index="measure", columns="col", values="mean").reindex([r for r, _ in ROWS] + ["LEVANTE full (fair)", "in-domain full"])[order]
S = G.pivot(index="measure", columns="col", values="sd").reindex(T.index)[order]
N = G.pivot(index="measure", columns="col", values="n").reindex(T.index)[order]
print("| measure | " + " | ".join(order) + " |")
print("|---|" + "---|" * len(order))
for meas in T.index:
    cells = []
    for c in order:
        m, s, n = T.loc[meas, c], S.loc[meas, c], N.loc[meas, c]
        cells.append("—" if pd.isna(m) or n == 0 else f"{m:.1f} ± {s:.1f}")
    print(f"| {meas} | " + " | ".join(cells) + " |")
print(f"\nwrote {R}/encoder_grid.csv ({len(G)} rows)")
