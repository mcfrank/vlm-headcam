"""Per-pair, per-word discourse-newness weights on the pairs' OWN text (so the referent word
is preserved and the uniform control matches filtnat). Newness: a word that hasn't appeared in
the last WINDOW words of this video's speech stream gets weight ~1; a recently-repeated word
(incl. function words) gets a low weight. Output aligned to the tokenized text."""
from collections import Counter, deque
import pandas as pd
from common import tokenize

WINDOW = 40  # rolling lookback in words

man = pd.read_parquet("manifests/grid_t15_filtnat_train.parquet").sort_values(["video_id", "frame_idx"])
rows = []
for vid, g in man.groupby("video_id"):
    dq, cnt = deque(), Counter()
    for r in g.itertuples(index=False):
        toks = tokenize(str(r.text))
        if not toks:
            continue
        ws = []
        for tk in toks:
            ws.append(1.0 / (1.0 + cnt[tk]))
            dq.append(tk); cnt[tk] += 1
            if len(dq) > WINDOW:
                cnt[dq.popleft()] -= 1
        rows.append((r.video_id, int(r.frame_idx), " ".join(toks),
                     " ".join(f"{x:.3f}" for x in ws)))

out = pd.DataFrame(rows, columns=["video_id", "frame_idx", "words", "w_disc"])
out.to_parquet("manifests/cue_filtnat_train.parquet", index=False)
print(f"cue manifest: {len(out)} pairs (own text + discourse-newness weights)")
print("  sample:", out.iloc[0].words, "|", out.iloc[0].w_disc)
