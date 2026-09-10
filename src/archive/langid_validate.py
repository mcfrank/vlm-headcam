"""Validate fastText language ID against Gemini on a stratified sample of utterances.

fastText says the corpus is 88.7% English, but 43% of utterances are 1-2 words where any
language ID is weak, and the corpus-wide tail (it 3.6%, eo 0.4%, la 0.3%) looks like classic
short-string false positives. Before we filter 1.1M pairs on this, measure how good it is HERE:
child-directed ASR, mostly very short, sometimes code-switched.

Text-only Gemini call (no image), so this is cheap. Stratified by predicted language, utterance
length and model confidence so the error rate is estimated where it matters, not just on the easy
majority.

usage: python src/langid_validate.py --n 1200 --out scored/langid_validation.parquet
"""
import argparse, json, os, re, threading, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np, pandas as pd
from google import genai
from google.genai import types

SYSTEM = ("You identify the language of short utterances transcribed from home audio recordings "
          "of families with young children. The text is ASR output: it may be fragmentary, "
          "misspelled, or code-switched.")
PROMPT = ('Return JSON: {"lang": "<ISO 639-1 code, e.g. en, es, pt, ja, ko, zh>", '
          '"mixed": true/false (does it contain more than one language?), '
          '"undecidable": true/false (too short or ambiguous to tell)}\nUtterance: ')

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="scored/utt_langid.parquet")
ap.add_argument("--out", default="scored/langid_validation.parquet")
ap.add_argument("--n", type=int, default=1200)
ap.add_argument("--workers", type=int, default=24)
ap.add_argument("--model", default="gemini-2.5-flash")
a = ap.parse_args()

d = pd.read_parquet(a.src)
d["lenbin"] = pd.cut(d.nwords, [0, 3, 7, 1000], labels=["1-3", "4-7", "8+"])
d["confbin"] = np.where(d.lang_conf >= 0.7, "conf", "unsure")
d["pred"] = np.where(d.lang == "en", "en", "non-en")
strata = d.groupby(["pred", "lenbin", "confbin"], observed=True)
per = max(1, a.n // max(len(strata), 1))
samp = strata.apply(lambda g: g.sample(min(per, len(g)), random_state=0)).reset_index(drop=True)
print(f"{len(samp)} utterances across {len(strata)} strata", flush=True)

client = genai.Client()
cfg = types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0.0,
                                  response_mime_type="application/json",
                                  thinking_config=types.ThinkingConfig(thinking_budget=0))
lock, out = threading.Lock(), []

def one(row):
    for attempt in range(4):
        try:
            r = client.models.generate_content(model=a.model, contents=PROMPT + str(row.text),
                                               config=cfg)
            j = json.loads(r.text)
            return dict(text=row.text, nwords=row.nwords, ft_lang=row.lang,
                        ft_conf=row.lang_conf, lenbin=str(row.lenbin), confbin=row.confbin,
                        g_lang=j.get("lang"), g_mixed=bool(j.get("mixed")),
                        g_undecidable=bool(j.get("undecidable")))
        except Exception as e:
            if attempt == 3:
                return dict(text=row.text, nwords=row.nwords, ft_lang=row.lang,
                            ft_conf=row.lang_conf, lenbin=str(row.lenbin), confbin=row.confbin,
                            g_lang=None, g_mixed=None, g_undecidable=None, error=str(e)[:80])
            time.sleep(2 ** attempt)

with ThreadPoolExecutor(max_workers=a.workers) as ex:
    for i, rec in enumerate(ex.map(one, samp.itertuples())):
        out.append(rec)
        if i % 200 == 0: print(f"  {i}/{len(samp)}", flush=True)

v = pd.DataFrame(out)
os.makedirs("scored", exist_ok=True)
v.to_parquet(a.out, index=False)
ok = v[v.g_lang.notna()]
print(f"\nscored {len(ok)}/{len(v)}")
dec = ok[~ok.g_undecidable.astype(bool)]
dec = dec.assign(ft_en=dec.ft_lang.eq("en"), g_en=dec.g_lang.eq("en"))
print(f"Gemini calls it undecidable on {100*ok.g_undecidable.mean():.0f}% "
      f"(1-3 words: {100*ok[ok.lenbin=='1-3'].g_undecidable.mean():.0f}%)")
print(f"\nEnglish/not agreement on decidable items: {100*(dec.ft_en==dec.g_en).mean():.1f}%")
print(pd.crosstab(dec.ft_en, dec.g_en, rownames=["fastText EN"], colnames=["Gemini EN"]).to_string())
print("\nby length:")
for L, g in dec.groupby("lenbin", observed=True):
    print(f"  {L:>4s}  n={len(g):4d}  agreement {100*(g.ft_en==g.g_en).mean():5.1f}%  "
          f"fastText-EN {100*g.ft_en.mean():5.1f}%  Gemini-EN {100*g.g_en.mean():5.1f}%")
fp = dec[(dec.ft_en == False) & (dec.g_en == True)]
print(f"\nfastText says non-English but Gemini says English: {len(fp)} "
      f"({100*len(fp)/max(len(dec),1):.1f}% of decidable) — these are the ones a filter would wrongly drop")
if len(fp):
    print(fp.ft_lang.value_counts().head(5).to_string())
