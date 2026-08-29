"""CDI (WS, English American) item -> semantic category, for coloring the lexicon maps.

Source: the Wordbank WS item dump vendored in lindazeng979/bilingual-babyLM
(BL_collection_data/wordbank_CDI_vocab/wordbank_item_data.csv, downloaded 2023-11-30) —
identical content to wordbankr::get_item_data(), used offline. Definitions normalized the
same way as figures/make_wordbank_anchors.R. -> results/cdi_categories.csv
"""
import re, sys
from pathlib import Path
import pandas as pd

src = Path(sys.argv[1])
d = pd.read_csv(src)[["item_definition", "category"]]
d["word"] = d.item_definition.astype(str).str.lower().str.replace(r" \(.*\)$", "", regex=True).str.strip()
d = d.drop_duplicates("word")[["word", "category"]]
d.to_csv(Path(__file__).resolve().parent.parent / "results" / "cdi_categories.csv", index=False)
print(len(d), "items;", d.category.nunique(), "categories:", ", ".join(sorted(d.category.unique())))
