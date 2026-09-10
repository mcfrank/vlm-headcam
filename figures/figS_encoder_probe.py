"""encoder visual quality does not explain the lexical-learning gap.

A head-free prototype 4AFC probe (leave-one-out nearest-prototype, 3 foils) measures how
separable object categories already are in each frozen encoder, on both evaluation domains.

A: probe accuracy per encoder on each domain. Naturalistic frames are far harder than
   catalogue photos for every encoder, and the encoders differ by only a few points.
B: the dissociation. Probe quality against what the same encoder's two-tower actually
   learns, in domain. Coarse category separability spans ~6 points while lexical learning
   spans ~26, so separability of this kind does not predict lexical scaling.

CAVEAT, and it bounds the claim: the probe is near ceiling on Konkle (96.5-100), so it
compresses real differences there. B is therefore plotted in domain, where the probe has
room, and the SI text should say a harder probe (82-way kNN or a linear readout) is the
test that would settle how large the true quality gap is.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T
import matplotlib.pyplot as plt

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
pr = pd.read_csv(R / "encoder_probe_domains.csv")
ind = pd.read_csv(R / "indomain_eval.csv")
ENC = [(E["tag"], E["label"], E["color"]) for E in reversed(T.ENCODERS)]   # smallest first
MK = {E["tag"]: E["marker"] for E in T.ENCODERS}

fig, (ax, bx) = plt.subplots(1, 2, figsize=(T.W2, 2.6),
                             gridspec_kw=dict(width_ratios=[1, 1.1], wspace=0.34))

# ---- A: probe by domain ----------------------------------------------------------
xs = np.arange(len(ENC))
for i, (enc, key, col) in enumerate(ENC):
    d = pr[pr.encoder == enc].set_index("domain").proto
    ax.plot([i, i], [d["indomain"], d["konkle"]], color=col, lw=1.0, zorder=2)
    ax.scatter([i], [d["konkle"]], s=22, color=col, zorder=3)
    ax.scatter([i], [d["indomain"]], s=20, facecolors="white", edgecolors=col, lw=1.0, zorder=3)
ax.axhline(25, color=T.SUB, lw=0.6, ls=(0, (4, 3)))
ax.text(len(ENC) - 0.55, 27, "chance", fontsize=5.4, color=T.SUB, ha="right")
for yy, fc, lb in [(35, T.INK, "Konkle photos"), (30, "white", "held-out BabyView frames")]:
    ax.scatter([-0.32], [yy], s=20, facecolors=fc, edgecolors=T.INK, lw=1.0, zorder=4,
               clip_on=False)
    ax.text(-0.18, yy, lb, fontsize=5.4, color=T.INK, va="center")
ax.set_xticks(xs)
ax.set_xticklabels([e[1] for e in ENC], fontsize=5.2, rotation=30, ha="right",
                   rotation_mode="anchor")
ax.set_xlim(-0.5, len(ENC) - 0.4)
ax.set_ylim(20, 105); ax.set_ylabel("prototype 4AFC (%)\nfrozen encoder, no learned head")
T.clean(ax)

# ---- B: separability vs what the tower learns ------------------------------------
for enc, key, col in ENC:
    x = pr[(pr.encoder == enc) & (pr.domain == "indomain")].proto.iloc[0]
    d = ind[(ind.encoder == enc) & (ind.N == 1686105)].acc
    bx.errorbar([x], [d.mean()], yerr=[d.std()], fmt=MK[enc], color=col, ms=4.6,
                elinewidth=0.8, capsize=2, zorder=3)
xr = pr[pr.domain == "indomain"].proto
yr = [ind[(ind.encoder == e) & (ind.N == 1686105)].acc.mean() for e, _, _ in ENC]
bx.annotate("", xy=(xr.min(), 80), xytext=(xr.max(), 80),
            arrowprops=dict(arrowstyle="<->", color=T.SUB, lw=0.7))
bx.text((xr.min() + xr.max()) / 2, 81.5, f"{xr.max() - xr.min():.0f} pts of separability",
        fontsize=5.4, color=T.SUB, ha="center")
bx.annotate("", xy=(62.0, min(yr)), xytext=(62.0, max(yr)),
            arrowprops=dict(arrowstyle="<->", color=T.SUB, lw=0.7))
bx.text(61.6, (min(yr) + max(yr)) / 2, f"{max(yr) - min(yr):.0f} pts\nof word\nlearning",
        fontsize=5.4, color=T.SUB, va="center", ha="right", linespacing=1.3)
# encoder key, drawn the same way as the keys in figS_nomil / figS_window
for i, (enc, key, col) in enumerate(ENC):
    bx.scatter([53.3], [44 - i * 3.2], s=14, color=col, marker=MK[enc], zorder=4)
    bx.text(53.75, 44 - i * 3.2, key, fontsize=5.2, color=col, va="center")
bx.set_xlim(52.8, 62.6); bx.set_ylim(26, 86)
bx.set_xlabel("prototype 4AFC (%), in domain")
bx.set_ylabel("in-domain word-learning 4AFC (%)\nfull corpus")
T.clean(bx)
print("  NOTE figS_encoder_probe: probe by domain " + "; ".join(
    f"{k} {pr[(pr.encoder==e)&(pr.domain=='konkle')].proto.iloc[0]:.1f}/"
    f"{pr[(pr.encoder==e)&(pr.domain=='indomain')].proto.iloc[0]:.1f}" for e, k, _ in ENC)
    + f" (konkle/in-domain); separability spans {xr.max()-xr.min():.1f} pts vs "
      f"{max(yr)-min(yr):.1f} pts of word learning")
for a, l in zip((ax, bx), "AB"):
    T.panel(a, l, dx=-0.17)
T.save(fig, "figS_encoder_probe")
