"""Decompose the region-MIL oracle 4AFC (49.9) by category. For each eval category:
per-category 4AFC, eval-pool size, whether the word is in the trained vocab, and how often
the word is actually SPOKEN in the training utterances (all pairs + clip-aligned subset).
Tests: is the 'squish' driven by many words that are rarely named (text coverage), rather
than a uniform ~50% ceiling on every word?"""
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, "/data2/mcfrank/vlm-headcam/src")
from common import MANIFEST_DIR, frame_key
from train import encode
from train_region_mil import RegionMIL, load_region_cache
import json

W = "/data2/mcfrank/vlm-headcam"
run = f"{W}/runs/P1_oracle_across"
dev = "cuda" if torch.cuda.is_available() else "cpu"

vocab = json.load(open(f"{run}/vocab.json"))
emb, lut = load_region_cache(f"{W}/emb_reg")
ev = pd.read_parquet(f"{W}/manifests/eval_frames.parquet")
model = RegionMIL(len(vocab), 512).to(dev)
model.load_state_dict(torch.load(f"{run}/model.pt", map_location=dev)); model.eval()

# word freq in training utterances (all + clip-aligned)
man = pd.read_parquet(f"{W}/manifests/boot_across_140000.parquet")
def wcount(df):
    c = {}
    for t in df.text:
        for w in set(encode(str(t), vocab, 64)):   # token ids present
            c[w] = c.get(w, 0) + 1
    return c
all_c = wcount(man)
aligned_c = wcount(man[man.clip_score_max > 0.24])
n_all, n_al = len(man), int((man.clip_score_max > 0.24).sum())

rng = np.random.default_rng(0); n_trials = 200; max_len = 16
# build pools
pools, cat_ids = {}, {}
for cat, g in ev.groupby("category"):
    ids = encode(cat, vocab, max_len)
    rows = [lut[frame_key(v, f)] for v, f in zip(g.video_id, g.frame_idx) if frame_key(v, f) in lut]
    pools[cat] = rows; cat_ids[cat] = ids   # keep even if OOV (ids empty) to report

in_vocab_cats = [c for c in pools if cat_ids[c] and len(pools[c]) >= 4]
all_rows = sorted({r for c in in_vocab_cats for r in pools[c]})
V = torch.from_numpy(np.asarray(emb[all_rows], np.float32)).to(dev)
Rp = model.enc_regions(V); r2i = {r: i for i, r in enumerate(all_rows)}

rows_out = []
for cat in sorted(pools):
    ids = cat_ids[cat]; pool = pools[cat]
    wid = ids[0] if len(ids) == 1 else (ids[0] if ids else None)
    # word freq: use the (single) token id for the category word
    freq_all = sum(all_c.get(t, 0) for t in set(ids)) if ids else 0
    freq_al = sum(aligned_c.get(t, 0) for t in set(ids)) if ids else 0
    if not ids or len(pool) < 4:
        rows_out.append(dict(cat=cat, acc=np.nan, npool=len(pool), in_vocab=bool(ids),
                             said_all=freq_all, said_aligned=freq_al)); continue
    t = torch.zeros(1, max_len, dtype=torch.long, device=dev); t[0, :len(ids)] = torch.tensor(ids, device=dev)
    tv = model.enc_text(t, torch.tensor([len(ids)], device=dev))
    others = [c for c in in_vocab_cats if c != cat]; correct = 0
    for _ in range(n_trials):
        cand = [rng.choice(pool)] + [rng.choice(pools[c]) for c in rng.choice(others, 3, replace=False)]
        cr = Rp[[r2i[r] for r in cand]]
        sc = torch.einsum('brd,md->brm', cr, tv).max(1).values.squeeze(-1)
        correct += int(sc.argmax().item() == 0)
    rows_out.append(dict(cat=cat, acc=correct / n_trials, npool=len(pool), in_vocab=True,
                         said_all=freq_all, said_aligned=freq_al))

df = pd.DataFrame(rows_out)
scored = df[df.acc.notna()].copy()
print(f"categories total={len(df)}  scored(in vocab, >=4 frames)={len(scored)}  "
      f"OOV/too-few={len(df)-len(scored)}")
print(f"mean 4AFC over scored = {scored.acc.mean()*100:.1f}  (matches oracle ~49.9)")
print(f"aligned training pairs (clip>0.24) = {n_al}/{n_all}\n")
print("worst 12 categories (acc, pool, times-said-aligned):")
for r in scored.nsmallest(12, "acc").itertuples():
    print(f"  {r.cat:14s} acc={r.acc*100:4.0f}  npool={r.npool:3d}  said_aligned={r.said_aligned:4d}  said_all={r.said_all}")
print("\nbest 12 categories:")
for r in scored.nlargest(12, "acc").itertuples():
    print(f"  {r.cat:14s} acc={r.acc*100:4.0f}  npool={r.npool:3d}  said_aligned={r.said_aligned:4d}  said_all={r.said_all}")
# correlation: does per-category acc track how often the word is spoken (aligned)?
from numpy import log1p
rho = np.corrcoef(pd.Series(scored.acc).rank(), pd.Series(scored.said_aligned).rank())[0,1]
rho_pool = np.corrcoef(pd.Series(scored.acc).rank(), pd.Series(scored.npool).rank())[0,1]
print(f"\nSpearman(acc, times-said-aligned) = {rho:.3f}")
print(f"Spearman(acc, eval-pool-size)     = {rho_pool:.3f}")
print(f"categories never said in aligned data: {(scored.said_aligned==0).sum()}  "
      f"their mean acc = {scored[scored.said_aligned==0].acc.mean()*100:.1f}")
print(f"categories said >=50x aligned:        {(scored.said_aligned>=50).sum()}  "
      f"their mean acc = {scored[scored.said_aligned>=50].acc.mean()*100:.1f}")
df.to_csv(f"{W}/runs/oracle_per_category.csv", index=False)
