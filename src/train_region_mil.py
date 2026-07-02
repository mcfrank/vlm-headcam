"""Region-MIL: an utterance matches its BEST region (CLS + GxG grid) in a frame, not the
whole scene. Supports plain contrastive training and the self-bootstrapped EM (the max-
region cosine is the endogenous alignment signal). No detector, no CLIP in training.

--mode plain : standard region-MIL InfoNCE (for oracle / baselines)
--mode boot  : warmup + EM rounds reweighting by the model's own max-region alignment
"""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from common import MANIFEST_DIR, load_emb_cache, EMB_DIM, save_json, frame_key
from train import build_vocab, encode
from train_boot import gmm2_posterior, yardstick


def load_region_cache(d):
    idx = pd.read_parquet(Path(d) / "index.parquet")
    emb = np.load(Path(d) / "emb.f16.npy", mmap_mode="r")   # [N, R, 768]
    lut = {frame_key(v, f): int(r) for v, f, r in zip(idx.video_id, idx.frame_idx, idx.row)}
    return emb, lut


class RegionPairs(torch.utils.data.Dataset):
    def __init__(self, emb, lut, man, vocab, max_len=16):
        self.rows, self.ids, self.clip = [], [], []
        for r in man.itertuples(index=False):
            k = frame_key(r.video_id, r.frame_idx)
            if k not in lut:
                continue
            toks = encode(r.text, vocab, max_len)
            if not toks:
                continue
            self.rows.append(lut[k]); self.ids.append(toks)
            self.clip.append(float(getattr(r, "clip_score_max", np.nan)))
        self.emb = emb; self.max_len = max_len; self.clip = np.array(self.clip)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        v = torch.from_numpy(np.asarray(self.emb[self.rows[i]], dtype=np.float32))  # [R,768]
        t = torch.zeros(self.max_len, dtype=torch.long)
        t[:len(self.ids[i])] = torch.tensor(self.ids[i])
        return i, v, t, len(self.ids[i])


def collate(b):
    idx = torch.tensor([x[0] for x in b]); v = torch.stack([x[1] for x in b])
    t = torch.stack([x[2] for x in b]); n = torch.tensor([x[3] for x in b])
    return idx, v, t, n


class RegionMIL(nn.Module):
    def __init__(self, vocab, dim=512, drop=0.1):
        super().__init__()
        self.vproj = nn.Sequential(nn.LayerNorm(EMB_DIM), nn.Dropout(drop), nn.Linear(EMB_DIM, dim))
        self.word = nn.Embedding(vocab, dim, padding_idx=0)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07)))

    def enc_regions(self, v):           # v [B,R,768] -> [B,R,D] normalized
        return F.normalize(self.vproj(v), dim=-1)

    def enc_text(self, t, n):
        e = self.word(t); m = (t != 0).unsqueeze(-1).float()
        return F.normalize((e * m).sum(1) / n.clamp(min=1).unsqueeze(-1).float(), dim=-1)

    def sims(self, R, T):               # R [B,Rn,D], T [M,D] -> [B,M] max over regions
        return torch.einsum('brd,md->brm', R, T).max(1).values

    def forward_loss(self, v, t, n, w):
        R = self.enc_regions(v); T = self.enc_text(t, n)
        logits = self.logit_scale.clamp(max=np.log(100)).exp() * self.sims(R, T)
        lab = torch.arange(len(v), device=v.device)
        li = F.cross_entropy(logits, lab, reduction="none")
        lt = F.cross_entropy(logits.t(), lab, reduction="none")
        return (0.5 * w * (li + lt)).sum() / w.sum().clamp(min=1e-6)


@torch.no_grad()
def eval_4afc_region(model, emb, lut, ev, vocab, dev, n_trials=100, seed=0, max_len=16):
    rng = np.random.default_rng(seed)
    pools, cat_ids = {}, {}
    for cat, g in ev.groupby("category"):
        ids = encode(cat, vocab, max_len)
        rows = [lut[frame_key(v, f)] for v, f in zip(g.video_id, g.frame_idx)
                if frame_key(v, f) in lut]
        if ids and len(rows) >= 4:
            pools[cat] = rows; cat_ids[cat] = ids
    cats = sorted(pools)
    if len(cats) < 4:
        return float("nan")
    all_rows = sorted({r for rs in pools.values() for r in rs})
    V = torch.from_numpy(np.asarray(emb[all_rows], dtype=np.float32)).to(dev)  # [K,R,768]
    Rp = model.enc_regions(V)                                                  # [K,R,D]
    r2i = {r: i for i, r in enumerate(all_rows)}
    accs = []
    for cat in cats:
        toks = cat_ids[cat]
        t = torch.zeros(1, max_len, dtype=torch.long, device=dev); t[0, :len(toks)] = torch.tensor(toks, device=dev)
        tv = model.enc_text(t, torch.tensor([len(toks)], device=dev))          # [1,D]
        others = [c for c in cats if c != cat]; correct = 0
        for _ in range(n_trials):
            cand = [rng.choice(pools[cat])] + [rng.choice(pools[c]) for c in rng.choice(others, 3, replace=False)]
            cr = Rp[[r2i[r] for r in cand]]                                    # [4,R,D]
            sc = torch.einsum('brd,md->brm', cr, tv).max(1).values.squeeze(-1) # [4]
            if sc.argmax().item() == 0:
                correct += 1
        accs.append(correct / n_trials)
    return float(np.mean(accs))


def rank01(x):
    r = pd.Series(x).rank().to_numpy()
    return (r - 1) / max(1, len(r) - 1)


def language_prior(ds, s):
    """Bootstrapped 'nameability': a word is grounded if the model reliably finds a
    matching region when it's said (mean max-region score over its occurrences). An
    utterance's prior = its best-grounded word. Function words self-exclude (low score);
    no POS/external knowledge used. Returns (L per pair, g per word)."""
    gsum, gcnt = {}, {}
    for i, toks in enumerate(ds.ids):
        for t in set(toks):
            gsum[t] = gsum.get(t, 0.0) + s[i]; gcnt[t] = gcnt.get(t, 0) + 1
    g = {t: gsum[t] / gcnt[t] for t in gsum}
    L = np.array([max((g.get(t, 0.0) for t in toks), default=0.0) for toks in ds.ids])
    return L, g


@torch.no_grad()
def mean_text(model, ds, dev, bs=512):
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate)
    acc = torch.zeros(model.word.embedding_dim, device=dev); nn_ = 0
    for idx, v, t, n in dl:
        T = model.enc_text(t.to(dev), n.to(dev)); acc += T.sum(0); nn_ += len(T)
    return F.normalize(acc / nn_, dim=-1)


@torch.no_grad()
def pair_scores(model, ds, dev, bs=512, mode="max", tbg=None):
    """max: pair's best region-text cosine. distinct: best region AFTER subtracting that
    region's match to the average utterance (down-weights ever-present, generic regions)."""
    model.eval(); s = np.zeros(len(ds), np.float32)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate)
    for idx, v, t, n in dl:
        R = model.enc_regions(v.to(dev)); T = model.enc_text(t.to(dev), n.to(dev))
        sim = torch.einsum('brd,bd->br', R, T)                      # [B,Rn]
        if mode == "distinct":
            sim = sim - (R @ tbg)                                   # subtract generic salience
        s[idx.numpy()] = sim.max(1).values.cpu().numpy()
    return s


@torch.no_grad()
def region_argmax_emb(model, ds, dev, bs=512):
    """For each pair, the JOINT embedding of the region best-aligned to its own text (E),
    and that alignment score (s)."""
    model.eval(); D = model.word.embedding_dim
    E = np.zeros((len(ds), D), np.float32); s = np.zeros(len(ds), np.float32)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate)
    for idx, v, t, n in dl:
        R = model.enc_regions(v.to(dev)); T = model.enc_text(t.to(dev), n.to(dev))
        sim = torch.einsum('brd,bd->br', R, T)              # [B,Rn]
        mx, arg = sim.max(1)
        e = R[torch.arange(len(R)), arg]                    # [B,D]
        E[idx.numpy()] = e.cpu().numpy(); s[idx.numpy()] = mx.cpu().numpy()
    return E, s


def cross_situational_scores(ds, E, conf, vocab_size):
    """Cross-situational prototype step. proto[w] = confidence-weighted mean of the chosen
    region embeddings across all situations where w occurs. A pair's alignment = does its
    chosen region match the accumulated prototype of one of its words? (Stable signal from
    accumulation, vs a single noisy pair.) Returns (score per pair, prototype coherence per word)."""
    D = E.shape[1]
    psum = np.zeros((vocab_size, D)); pcnt = np.zeros(vocab_size)
    for i, toks in enumerate(ds.ids):
        c = conf[i]
        for w in set(toks):
            psum[w] += c * E[i]; pcnt[w] += c
    proto = psum / np.maximum(pcnt[:, None], 1e-6)
    nrm = np.linalg.norm(proto, axis=1, keepdims=True)
    proto = proto / np.maximum(nrm, 1e-6)
    coh = nrm.squeeze(-1)                                    # ||mean||: coherent word -> ~1
    s = np.zeros(len(ds), np.float32)
    for i, toks in enumerate(ds.ids):
        best = 0.0
        for w in set(toks):
            best = max(best, float(E[i] @ proto[w]))
        s[i] = best
    return s, coh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--region-cache", required=True)
    ap.add_argument("--eval-frames", default=str(MANIFEST_DIR / "eval_frames.parquet"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["plain", "boot"], default="plain")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)      # plain
    ap.add_argument("--warmup", type=int, default=5)       # boot
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--epochs-per-round", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-freq", type=int, default=5)
    ap.add_argument("--floor", type=float, default=0.05)
    ap.add_argument("--lang-prior", action="store_true",
                    help="fold bootstrapped word-groundedness prior into the E-step weight")
    ap.add_argument("--score-mode", choices=["max", "distinct"], default="max",
                    help="distinct = subtract each region's generic salience (base-rate correction)")
    ap.add_argument("--proto", action="store_true",
                    help="cross-situational prototype E-step (accumulate word->region prototypes)")
    ap.add_argument("--vocab-json", default=None, help="load a fixed shared vocab (for curriculum)")
    ap.add_argument("--init-from", default=None, help="init model from a prior run's model.pt (curriculum)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    emb, lut = load_region_cache(args.region_cache)
    man = pd.read_parquet(args.manifest); ev = pd.read_parquet(args.eval_frames)
    if args.vocab_json:
        import json as _json
        vocab = _json.load(open(args.vocab_json))
    else:
        vocab = build_vocab(man.text, args.min_freq)
    ds = RegionPairs(emb, lut, man, vocab)
    save_json(vocab, out / "vocab.json")
    print(f"pairs {len(ds)} | vocab {len(vocab)} | mode {args.mode}", flush=True)

    model = RegionMIL(len(vocab), args.dim).to(dev)
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location=dev))
        print(f"init from {args.init_from}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    W = torch.ones(len(ds), device=dev)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True, collate_fn=collate)
    log = []

    def epochs(k, phase, rnd):
        for _ in range(k):
            model.train()
            for idx, v, t, n in dl:
                loss = model.forward_loss(v.to(dev), t.to(dev), n.to(dev), W[idx.to(dev)])
                opt.zero_grad(); loss.backward(); opt.step()
        acc = eval_4afc_region(model, emb, lut, ev, vocab, dev, seed=args.seed)
        rec = {"phase": phase, "round": rnd, "acc_4afc": round(acc, 4), "eff": int(W.gt(0.5).sum().item())}
        log.append(rec); print(rec, flush=True); return acc

    if args.mode == "plain":
        for e in range(args.epochs):
            epochs(1, "plain", e)
    else:
        epochs(args.warmup, "warmup", -1)
        conf = np.ones(len(ds), np.float32)   # for proto accumulation
        for r in range(args.rounds):
            if args.proto:
                E, _ = region_argmax_emb(model, ds, dev)
                s, coh = cross_situational_scores(ds, E, conf, len(vocab))
            else:
                tbg = mean_text(model, ds, dev) if args.score_mode == "distinct" else None
                s = pair_scores(model, ds, dev, mode=args.score_mode, tbg=tbg)
            if args.lang_prior:
                L, g = language_prior(ds, s)
                combined = 0.5 * (rank01(s) + rank01(L))   # need BOTH a matching region and a grounded word
                w = np.clip(gmm2_posterior(combined), args.floor, 1.0)
            else:
                w = np.clip(gmm2_posterior(s), args.floor, 1.0)
            w = w.astype(np.float32)
            conf = w                                # refine prototype confidence next round
            W = torch.tensor(w, dtype=torch.float32, device=dev)
            epochs(args.epochs_per_round, "boot", r)
            y = yardstick(w, ds.clip); log[-1].update(y); print("  yardstick:", y, flush=True)
        np.savez(out / "weights.npz", w=w, clip=ds.clip, s=s)

    save_json(log, out / "log.json")
    torch.save(model.state_dict(), out / "model.pt")
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
