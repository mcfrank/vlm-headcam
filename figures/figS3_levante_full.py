"""Display item 11 — the same scaling experiment under two evals: Konkle and LEVANTE.

Same encoders, same training draws, both 4AFC (chance 25). LEVANTE (159 vocabulary items,
difficulty-calibrated on real children) is scored two ways because vocabulary coverage grows
with the corpus (1% -> 70% of items in-vocab): solid = all 159 items with chance credited for
out-of-vocab items (the fair floor); dotted = in-vocab items only (upper bound; item set grows
with scale, so composition shifts). Konkle items are all in-vocab from 30k on, so the two
conventions coincide there. """
import sys, re, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
lev = pd.read_csv(R / "lev_scaling_final.csv")
lev = lev[lev.N <= 1_686_105]                     # final-corpus rows only (drop preview runs)

ENC = [("L-OTS", "DINOv3-L\noff-the-shelf", r"F_dinov3l_rand_(\d+)", "F_dinov3l_base", T.OTHER),
       ("B-OTS", "DINOv3-B\noff-the-shelf", r"F_dinov3b_rand_(\d+)", "F_dinov3b_base", T.FREE),
       ("B-BV", "ViT-B\nBabyView-trained", r"F_vitb_bv_rand_(\d+)", "F_vitb_bv_base", T.INDOM),
       ("S-BV", "ViT-S\nBabyView-trained", r"F_vits_bv_rand_(\d+)", "F_vits_bv_base", "#d99aa7")]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(T.W2, 2.6))

# ---- A: Konkle -------------------------------------------------------------------
for key, lab, rx, full, col in ENC:
    fams = sorted((int(m.group(1)), f) for f in D.runs.family.unique()
                  if (m := re.fullmatch(rx, str(f)))) + [(None, full)]
    fam = [D.family(f) for _, f in fams]
    x = np.array([fm["n_pairs"] for fm in fam])
    ax.errorbar(x, [fm["mean"] for fm in fam], yerr=[fm["sd"] for fm in fam], fmt="-o",
                color=col, ms=2.8, lw=1.0, elinewidth=0.7, capsize=1.5, zorder=3)
    va = {"B-OTS": "top", "L-OTS": "bottom"}.get(key, "center")
    dy = {"top": -1.0, "bottom": 1.0}.get(va, 0)
    ax.text(x[-1] * 1.3, fam[-1]["mean"] + dy, lab, fontsize=5.4, color=col, va=va,
            linespacing=1.25)

# ---- B: LEVANTE ------------------------------------------------------------------
lev["fair"] = np.where(lev.playable, lev.correct, 0.25)
per_seed = (lev.groupby(["encoder", "N", "seed"])
               .agg(fair=("fair", "mean"), cov=("playable", "mean")).reset_index())
inv = (lev[lev.playable].groupby(["encoder", "N", "seed"]).correct.mean()
          .rename("inv").reset_index())
per_seed = per_seed.merge(inv, on=["encoder", "N", "seed"])
S = (per_seed.groupby(["encoder", "N"])
        .agg(fair_m=("fair", "mean"), fair_sd=("fair", "std"),
             inv_m=("inv", "mean"), inv_sd=("inv", "std"), coverage=("cov", "mean")).reset_index())
N_ITEMS = lev.item.nunique()
for key, lab, rx, full, col in ENC:
    d = S[S.encoder == key].sort_values("N")
    bx.errorbar(d.N, 100 * d.fair_m, yerr=100 * d.fair_sd, fmt="-o", color=col, ms=2.8,
                lw=1.0, elinewidth=0.7, capsize=1.5, zorder=3)
    di = d[d.coverage * N_ITEMS >= 15]                 # in-vocab-only needs enough items to mean much
    bx.plot(di.N, 100 * di.inv_m, ls=(0, (1.5, 1.5)), lw=0.9, color=col, zorder=2, alpha=0.8)
    yl = 30.8 if key == "L-BV" else 100 * d.fair_m.iloc[-1]
    bx.text(d.N.max() * 1.3, yl, lab, fontsize=5.4, color=col, va="center", linespacing=1.25)
# vocabulary coverage along the bottom (identical for all encoders: a corpus property)
cov = S[S.encoder == "B-OTS"].sort_values("N")
for n, c in zip(cov.N, cov.coverage):
    bx.text(n, 19.8, f"{100*c:.0f}", fontsize=4.6, color=T.SUB, ha="center", va="bottom")
bx.text(2.6e7, 19.8, "% of words in vocab", fontsize=4.6, color=T.SUB, ha="right", va="bottom")
from matplotlib.lines import Line2D
bx.legend(handles=[Line2D([], [], color=T.SUB, lw=1.0, marker="o", ms=2.8,
                          label="all 159 items (chance if out-of-vocab)"),
                   Line2D([], [], color=T.SUB, lw=0.9, ls=(0, (1.5, 1.5)),
                          label="in-vocab items only")],
          loc="upper left", fontsize=5.0, handlelength=1.8, borderpad=0.3, labelspacing=0.4)

for a, ylab in [(ax, "Konkle 4AFC (%)"), (bx, "LEVANTE vocabulary 4AFC (%)")]:
    a.axhline(25, color=T.SUB, lw=0.6, ls=(0, (4, 3)))
    a.set_xscale("log"); a.set_xlim(2e3, 3e7); a.set_ylim(18, 95)
    a.set_xlabel("training pairs"); a.set_ylabel(ylab)
    T.clean(a)
ax.text(2.6e7, 22.2, "chance", fontsize=5.4, color=T.SUB, ha="right")
for a, l in zip((ax, bx), "AB"):
    T.panel(a, l)
T.save(fig, "figS3_levante_full")
