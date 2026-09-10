import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="/data2/mcfrank/vlm-headcam/book_figs"
FREE,ORACLE,INK,SUB,GRID="#1d9e75","#185fa5","#2c2c2a","#6b6a66","#e1e0d9"
NULLC="#b0655a"
# (label, value, color)  DINOv2 emb_reg, Konkle test-60, 3 seeds
BARS=[("contrastive\nhard-max (MIL)",66.1,FREE),
      ("soft pool\n(t=0.05)",65.3,"#6fae97"),
      ("soft pool\n(t=0.15)",63.1,"#9ec9b8"),
      ("soft pool\n(t=0.50)",61.6,"#c9a24b"),
      ("learned\nattention",60.2,NULLC),
      ("generative\ncaptioner",62.1,ORACLE)]
fig,ax=plt.subplots(figsize=(9,4.7),dpi=150)
x=range(len(BARS))
ax.bar(x,[b[1] for b in BARS],width=0.62,color=[b[2] for b in BARS],zorder=3)
for i,b in enumerate(BARS): ax.text(i,b[1]+0.4,f"{b[1]:.1f}",ha="center",fontsize=9.5,color=INK)
ax.axhline(66.1,color=FREE,lw=1.1,ls=(0,(5,4)),zorder=2)
ax.text(5.4,66.4,"hard-max baseline",color=FREE,fontsize=8.5,ha="right",va="bottom")
ax.set_xticks(list(x)); ax.set_xticklabels([b[0] for b in BARS],fontsize=8.5)
ax.set_ylim(55,70); ax.set_ylabel("Konkle 4AFC",fontsize=10,color=SUB)
ax.set_title("Nothing beats plain contrastive hard-max: softer and generative alike",fontsize=11.5,color=INK,loc="left")
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=SUB,length=0); ax.yaxis.grid(True,color=GRID,lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_architectures.png",bbox_inches="tight"); print("wrote arch")
