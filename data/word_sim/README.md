# Word-relatedness / similarity evaluation sets

Copied verbatim from EvLab-MIT/LexiContrastiveGrd (`src/llm_devo/word_sim/data/`,
github.com/EvLab-MIT/LexiContrastiveGrd) — the eval used in Vong-style lexical-semantics
comparisons and, vocab-filtered, in lindazeng979/bilingual-babyLM (arXiv 2603.29552).
Format: `word1;word2;human_score` per line. We follow the bilingual-babyLM protocol:
score only pairs where BOTH words are in the model's vocabulary, report coverage.
Scored by src/lex_score.py -> results/lexicon_relatedness.csv.
