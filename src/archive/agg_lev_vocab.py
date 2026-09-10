"""Aggregate the LEVANTE-vocab eval across seeds: overall accuracy, tercile accuracy (mean+/-sd),
and item-difficulty correlations. Writes a seed-mean per-item table for the figure."""
import numpy as np
import pandas as pd

W = "/data2/mcfrank/vlm-headcam"
S = [pd.read_parquet(f"{W}/book_figs/lev_vocab_G_base_mil_full_s{s}.parquet") for s in range(3)]

base = S[0][["item_uid", "target_word", "d"]].copy()
base["correct"] = np.mean([s.correct.values for s in S], 0)
base["p_correct"] = np.mean([s.p_correct.values for s in S], 0)
base.to_parquet(f"{W}/book_figs/lev_vocab_seedmean.parquet", index=False)

p = base[base.correct.notna() & base.d.notna()].copy()
print("playable+d items:", len(p))

accs = [s[s.correct.notna()].correct.mean() * 100 for s in S]
print(f"overall 4AFC: {np.mean(accs):.1f} +/- {np.std(accs):.1f}  (seeds {[round(a, 1) for a in accs]})")

p["bin"] = pd.qcut(p.d, 3, labels=["easy", "med", "hard"])
uid2bin = dict(zip(p.item_uid, p.bin))
rows = {b: [] for b in ["easy", "med", "hard"]}
for s in S:
    ss = s[s.item_uid.isin(p.item_uid)].copy()
    ss["bin"] = ss.item_uid.map(uid2bin)
    g = ss.groupby("bin", observed=True).correct.mean() * 100
    for b in rows:
        rows[b].append(g[b])
print("tercile accuracy (mean +/- sd across seeds):")
for b in ["easy", "med", "hard"]:
    n = int((p.bin == b).sum())
    print(f"  child-{b:4s}: {np.mean(rows[b]):.1f} +/- {np.std(rows[b]):.1f}  (n={n})")

rhos = [s[s.correct.notna() & s.d.notna()][["p_correct", "d"]].corr("spearman").iloc[0, 1] for s in S]
rho_pc = p[["p_correct", "d"]].corr("spearman").iloc[0, 1]
rho_c = p[["correct", "d"]].corr("spearman").iloc[0, 1]
print(f"rho(p_correct, d) per seed: {[round(r, 3) for r in rhos]}  mean {np.mean(rhos):+.3f}")
print(f"rho(seed-avg p_correct, d) = {rho_pc:+.3f}")
print(f"rho(seed-avg correct, d)   = {rho_c:+.3f}")
