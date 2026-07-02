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
        self.rows, self.ids, self.clip, self.vf = [], [], [], []
        for r in man.itertuples(index=False):
            k = frame_key(r.video_id, r.frame_idx)
            if k not in lut:
                continue
            toks = encode(r.text, vocab, max_len)
            if not toks:
                continue
            self.rows.append(lut[k]); self.ids.append(toks)
            self.clip.append(float(getattr(r, "clip_score_max", np.nan)))
            self.vf.append((r.video_id, int(r.frame_idx)))
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
        self.region_prior = None        # optional [Rn] bias on which region wins the MIL max

    def enc_regions(self, v):           # v [B,R,768] -> [B,R,D] normalized
        return F.normalize(self.vproj(v), dim=-1)

    def enc_text(self, t, n):
        e = self.word(t); m = (t != 0).unsqueeze(-1).float()
        return F.normalize((e * m).sum(1) / n.clamp(min=1).unsqueeze(-1).float(), dim=-1)

    def sims(self, R, T):               # R [B,Rn,D], T [M,D] -> [B,M] max over regions
        s = torch.einsum('brd,md->brm', R, T)
        if self.region_prior is not None:
            s = s + self.region_prior.view(1, -1, 1)   # bias region SELECTION toward cued regions
        return s.max(1).values

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
            sc = torch.einsum('brd,md->brm', cr, tv)                           # [4,R,1]
            if model.region_prior is not None:
                sc = sc + model.region_prior.view(1, -1, 1)
            sc = sc.max(1).values.squeeze(-1)                                  # [4]
            if sc.argmax().item() == 0:
                correct += 1
        accs.append(correct / n_trials)
    return float(np.mean(accs))


def rank01(x):
    r = pd.Series(x).rank().to_numpy()
    return (r - 1) / max(1, len(r) - 1)


def _norm_ppf(u):
    """Inverse standard-normal CDF via torch.erfinv (avoids a scipy dependency)."""
    u = np.clip(u, 1e-6, 1 - 1e-6)
    return (np.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * u - 1.0)).numpy())


def make_cue(clip, rho, cov, rng, neutral=0.5):
    """Synthetic 'social cue' of controlled quality: a Gaussian-copula corruption of the
    held-out CLIP truth. `rho` sets the target Spearman with clip; `cov` is the fraction of
    pairs on which the cue is revealed (the rest get a neutral weight). Returns a per-pair
    soft weight in [0,1] aligned to ds order, plus the ACHIEVED Spearman on revealed pairs.
    This is the titration knob for 'how much aligned-ness information does the loop need'."""
    ok = ~np.isnan(clip)
    zt = _norm_ppf(rank01(clip[ok]))                      # latent gaussian of truth
    noise = rng.standard_normal(ok.sum())
    z = rho * zt + np.sqrt(max(0.0, 1 - rho * rho)) * noise
    c_ok = rank01(z)                                      # copula -> uniform, spearman≈rho
    cue = np.full(len(clip), neutral, np.float64)
    cue[ok] = c_ok
    hide = rng.random(len(cue)) > cov                     # coverage: hide (1-cov) of pairs
    cue[hide] = neutral
    shown = ok & ~hide
    ach = float(np.corrcoef(pd.Series(cue[shown]).rank(), pd.Series(clip[shown]).rank())[0, 1]) \
        if shown.sum() > 50 else float("nan")
    return cue.astype(np.float32), ach


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
        if model.region_prior is not None:
            sim = sim + model.region_prior.view(1, -1)
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
        if model.region_prior is not None:
            sim = sim + model.region_prior.view(1, -1)
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
    ap.add_argument("--titrate-rho", type=float, default=None,
                    help="TRACK B: inject a synthetic cue with this target Spearman vs CLIP truth")
    ap.add_argument("--titrate-cov", type=float, default=1.0,
                    help="TRACK B: fraction of pairs on which the synthetic cue is revealed")
    ap.add_argument("--prior-mode", choices=["fixed", "seed", "blend", "gate"], default="fixed",
                    help="fixed: soft cue IS the weight every round (graded titration curve); "
                         "gate: HARD-threshold the cue to keep the top --gate-frac as positives "
                         "(weight 1) and drop the rest (this mirrors the ch.3 CLIP filter); "
                         "seed: cue weights warmup only, then endogenous EM (amplification test); "
                         "blend: combine cue with the model's own score each round")
    ap.add_argument("--gate-frac", type=float, default=0.12,
                    help="gate mode: fraction of pairs kept as hard positives (default ~ base rate)")
    ap.add_argument("--ext-prior", default=None,
                    help="parquet (video_id,frame_idx,<col>) of an external per-pair cue folded "
                         "into the boot E-step weight (e.g. prosody/discourse). See --ext-col")
    ap.add_argument("--ext-col", default="cont_share", help="column in --ext-prior to use")
    ap.add_argument("--center-prior", type=float, default=0.0,
                    help="REGION prior (child-gaze): add this cosine bonus to the central 2x2 "
                         "grid cells so the MIL prefers central regions (0 = off)")
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

    ext_prior = None
    if args.ext_prior:
        ep = pd.read_parquet(args.ext_prior)
        d = {(v, int(f)): float(x) for v, f, x in zip(ep.video_id, ep.frame_idx, ep[args.ext_col])}
        ext_prior = np.array([d.get(vf, np.nan) for vf in ds.vf], np.float32)
        med = np.nanmedian(ext_prior); ext_prior = np.where(np.isnan(ext_prior), med, ext_prior)
        print(f"ext-prior {args.ext_col}: covered {np.isfinite([d.get(vf, np.nan) for vf in ds.vf]).mean():.0%} "
              f"| rho(prior,clip)={yardstick(rank01(ext_prior), ds.clip).get('spearman_w_clip')}", flush=True)

    model = RegionMIL(len(vocab), args.dim).to(dev)
    if args.center_prior > 0:
        R = emb.shape[1]                     # 1 CLS + G*G grid (17 for 4x4)
        pri = torch.zeros(R, device=dev)
        pri[[6, 7, 10, 11]] = args.center_prior   # central 2x2 of the 4x4 grid (CLS=idx0)
        model.region_prior = pri
        print(f"center-prior {args.center_prior} on region cells [6,7,10,11]", flush=True)
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

    cue = None
    if args.titrate_rho is not None:
        cue, ach = make_cue(ds.clip, args.titrate_rho, args.titrate_cov,
                            np.random.default_rng(args.seed))
        print(f"titrate: target_rho={args.titrate_rho} cov={args.titrate_cov} "
              f"achieved_rho={ach:.3f} mode={args.prior_mode}", flush=True)
        log.append({"phase": "titrate", "target_rho": args.titrate_rho, "cov": args.titrate_cov,
                    "achieved_rho": round(ach, 3), "prior_mode": args.prior_mode})

    if args.mode == "plain":
        for e in range(args.epochs):
            epochs(1, "plain", e)
    else:
        # a hard gate keeps only the top --gate-frac of the cue as positives (mirrors the filter)
        cue_gate = None
        if cue is not None and args.prior_mode == "gate":
            thr = np.quantile(cue, 1.0 - args.gate_frac)
            cue_gate = np.where(cue >= thr, 1.0, args.floor).astype(np.float32)
            print(f"gate: keep frac={float((cue_gate > args.floor).mean()):.3f} thr={thr:.3f}", flush=True)
        # cue weights the warm-up for fixed/seed/gate modes (a seeded toehold before EM)
        if cue is not None and args.prior_mode in ("fixed", "seed", "gate"):
            wfx = cue_gate if cue_gate is not None else np.clip(cue, args.floor, 1.0)
            W = torch.tensor(wfx, dtype=torch.float32, device=dev)
        epochs(args.warmup, "warmup", -1)
        conf = np.ones(len(ds), np.float32)   # for proto accumulation
        for r in range(args.rounds):
            if cue is not None and args.prior_mode in ("fixed", "gate"):
                w = (cue_gate if cue_gate is not None
                     else np.clip(cue, args.floor, 1.0)).astype(np.float32)   # cue IS the weight, no model update
            else:
                if args.proto:
                    E, _ = region_argmax_emb(model, ds, dev)
                    s, coh = cross_situational_scores(ds, E, conf, len(vocab))
                else:
                    tbg = mean_text(model, ds, dev) if args.score_mode == "distinct" else None
                    s = pair_scores(model, ds, dev, mode=args.score_mode, tbg=tbg)
                if ext_prior is not None:
                    combined = 0.5 * (rank01(s) + rank01(ext_prior))   # endogenous score + external cue
                    w = np.clip(gmm2_posterior(combined), args.floor, 1.0)
                elif cue is not None and args.prior_mode == "blend":
                    combined = 0.5 * (rank01(s) + rank01(cue))   # cue + endogenous score
                    w = np.clip(gmm2_posterior(combined), args.floor, 1.0)
                elif args.lang_prior:
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
        np.savez(out / "weights.npz", w=w, clip=ds.clip, cue=(cue if cue is not None else []))

    save_json(log, out / "log.json")
    torch.save(model.state_dict(), out / "model.pt")
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
