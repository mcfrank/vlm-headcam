# results/ — every number the paper uses, and what reads it

Generated 2026-09-10 by a scan of `figures/*.py` and `src/make_*.py` for file references.
Files under **paper-visible** are read by a figure or table generator; the rest are legacy
(book-era, preview-corpus, or superseded) and are kept only for the record.

## Paper-visible

| file | read by |
|---|---|
| `cdi_categories.csv` | fig3_lexicon.py, figS_lexicon_tsne.py, make_cdi_categories.py |
| `corpus.csv` | figS_alignment_scores.py |
| `encoder_grid.csv` | make_encoder_grid.py |
| `encoder_probe_domains.csv` | figS_encoder_probe.py |
| `evals.parquet` | check_provenance.py |
| `experiments_table.csv` | make_experiments_table.py |
| `experiments_table.tex` | make_experiments_table.py |
| `indomain_eval.csv` | figS_encoder_probe.py, figS_indomain.py, make_encoder_grid.py |
| `item_eval_final.csv` | figS_item_difficulty.py, make_wordbank_40.py |
| `konkle_wg40_per_seed.csv` | fig4_development.py, make_wordbank_40.py |
| `lev_scaling.csv` | fig4_development.py, figS_levante.py, make_encoder_grid.py, make_levante_ages.py, theme.py |
| `lev_scaling_final.csv` | fig4_development.py, figS_levante.py, make_encoder_grid.py, make_levante_ages.py, theme.py |
| `levante_child_by_age.csv` | fig4_development.py, make_levante_ages.py |
| `levante_en_item_d.csv` | make_levante_ages.py |
| `levante_en_scores.csv` | make_levante_ages.py |
| `lexicon_category_structure_F_dinov3b_base.csv` | fig3_lexicon.py, figS_lexicon_tsne.py, figS_lexicon_tsne.py/fig3_lexicon.py |
| `lexicon_category_structure_F_dinov3l_base.csv` | fig3_lexicon.py, figS_lexicon_tsne.py, figS_lexicon_tsne.py/fig3_lexicon.py |
| `lexicon_category_structure_F_dinov3s_base.csv` | fig3_lexicon.py, figS_lexicon_tsne.py, figS_lexicon_tsne.py/fig3_lexicon.py |
| `lexicon_category_structure_F_vitb_bv_base.csv` | fig3_lexicon.py, figS_lexicon_tsne.py, figS_lexicon_tsne.py/fig3_lexicon.py |
| `lexicon_category_structure_F_vitl_bv_base.csv` | fig3_lexicon.py, figS_lexicon_tsne.py, figS_lexicon_tsne.py/fig3_lexicon.py |
| `lexicon_category_structure_F_vits_bv_base.csv` | fig3_lexicon.py, figS_lexicon_tsne.py, figS_lexicon_tsne.py/fig3_lexicon.py |
| `lexicon_partial_F-dinov3b-aligned.csv` | fig3_lexicon.py, figS_lexicon_relatedness.py, figS_lexicon_relatedness.py/fig3_lexicon.py |
| `lexicon_partial_F-dinov3b.csv` | fig3_lexicon.py, figS_lexicon_relatedness.py, figS_lexicon_relatedness.py/fig3_lexicon.py |
| `lexicon_partial_F-dinov3l-aligned.csv` | fig3_lexicon.py, figS_lexicon_relatedness.py, figS_lexicon_relatedness.py/fig3_lexicon.py |
| `lexicon_partial_F-dinov3l.csv` | fig3_lexicon.py, figS_lexicon_relatedness.py, figS_lexicon_relatedness.py/fig3_lexicon.py |
| `lexicon_partial_F-dinov3s-aligned.csv` | fig3_lexicon.py, figS_lexicon_relatedness.py, figS_lexicon_relatedness.py/fig3_lexicon.py |
| `lexicon_partial_F-dinov3s.csv` | fig3_lexicon.py, figS_lexicon_relatedness.py, figS_lexicon_relatedness.py/fig3_lexicon.py |
| `lexicon_partial_F-vitb_bv-aligned.csv` | fig3_lexicon.py, figS_lexicon_relatedness.py, figS_lexicon_relatedness.py/fig3_lexicon.py |
| `lexicon_partial_F-vitb_bv.csv` | fig3_lexicon.py, figS_lexicon_relatedness.py, figS_lexicon_relatedness.py/fig3_lexicon.py |
| `lexicon_partial_F-vitl_bv-aligned.csv` | fig3_lexicon.py, figS_lexicon_relatedness.py, figS_lexicon_relatedness.py/fig3_lexicon.py |
| `lexicon_partial_F-vitl_bv.csv` | fig3_lexicon.py, figS_lexicon_relatedness.py, figS_lexicon_relatedness.py/fig3_lexicon.py |
| `lexicon_partial_F-vits_bv-aligned.csv` | fig3_lexicon.py, figS_lexicon_relatedness.py, figS_lexicon_relatedness.py/fig3_lexicon.py |
| `lexicon_partial_F-vits_bv.csv` | fig3_lexicon.py, figS_lexicon_relatedness.py, figS_lexicon_relatedness.py/fig3_lexicon.py |
| `lexicon_rungs.csv` | figS_lexicon_alignment.py |
| `lexicon_tsne_F_dinov3b_base_s0_NOUN.csv` | fig3_lexicon.py, figS_lexicon_tsne.py, figS_lexicon_tsne.py/fig3_lexicon.py |
| `lexicon_tsne_F_dinov3l_base_s0_NOUN.csv` | fig3_lexicon.py, figS_lexicon_tsne.py, figS_lexicon_tsne.py/fig3_lexicon.py |
| `lexicon_tsne_F_dinov3l_lad_filtnat_s0_NOUN.csv` | fig3_lexicon.py, figS_lexicon_tsne.py, figS_lexicon_tsne.py/fig3_lexicon.py |
| `lexicon_tsne_F_dinov3l_rand_100000_s0_NOUN.csv` | fig3_lexicon.py, figS_lexicon_tsne.py, figS_lexicon_tsne.py/fig3_lexicon.py |
| `lexicon_tsne_F_dinov3s_base_s0_NOUN.csv` | fig3_lexicon.py, figS_lexicon_tsne.py, figS_lexicon_tsne.py/fig3_lexicon.py |
| `lexicon_tsne_F_vitb_bv_base_s0_NOUN.csv` | fig3_lexicon.py, figS_lexicon_tsne.py, figS_lexicon_tsne.py/fig3_lexicon.py |
| `lexicon_tsne_F_vitl_bv_base_s0_NOUN.csv` | fig3_lexicon.py, figS_lexicon_tsne.py, figS_lexicon_tsne.py/fig3_lexicon.py |
| `lexicon_tsne_F_vits_bv_base_s0_NOUN.csv` | fig3_lexicon.py, figS_lexicon_tsne.py, figS_lexicon_tsne.py/fig3_lexicon.py |
| `lexicon_ws_scaling_F-dinov3b-aligned.csv` | fig3_lexicon.py, figS_lexicon_alignment.py, figS_lexicon_relatedness.py, figS_lexicon_relatedness.py/fig3_lexicon.py/figS_lexicon_alignment.py |
| `lexicon_ws_scaling_F-dinov3b.csv` | fig3_lexicon.py, figS_lexicon_alignment.py, figS_lexicon_relatedness.py, figS_lexicon_relatedness.py/fig3_lexicon.py/figS_lexicon_alignment.py |
| `lexicon_ws_scaling_F-dinov3l-aligned.csv` | fig3_lexicon.py, figS_lexicon_alignment.py, figS_lexicon_relatedness.py, figS_lexicon_relatedness.py/fig3_lexicon.py/figS_lexicon_alignment.py |
| `lexicon_ws_scaling_F-dinov3l.csv` | fig3_lexicon.py, figS_lexicon_alignment.py, figS_lexicon_relatedness.py, figS_lexicon_relatedness.py/fig3_lexicon.py/figS_lexicon_alignment.py |
| `lexicon_ws_scaling_F-dinov3s-aligned.csv` | fig3_lexicon.py, figS_lexicon_alignment.py, figS_lexicon_relatedness.py, figS_lexicon_relatedness.py/fig3_lexicon.py/figS_lexicon_alignment.py |
| `lexicon_ws_scaling_F-dinov3s.csv` | fig3_lexicon.py, figS_lexicon_alignment.py, figS_lexicon_relatedness.py, figS_lexicon_relatedness.py/fig3_lexicon.py/figS_lexicon_alignment.py |
| `lexicon_ws_scaling_F-vitb_bv-aligned.csv` | fig3_lexicon.py, figS_lexicon_alignment.py, figS_lexicon_relatedness.py, figS_lexicon_relatedness.py/fig3_lexicon.py/figS_lexicon_alignment.py |
| `lexicon_ws_scaling_F-vitb_bv.csv` | fig3_lexicon.py, figS_lexicon_alignment.py, figS_lexicon_relatedness.py, figS_lexicon_relatedness.py/fig3_lexicon.py/figS_lexicon_alignment.py |
| `lexicon_ws_scaling_F-vitl_bv-aligned.csv` | fig3_lexicon.py, figS_lexicon_alignment.py, figS_lexicon_relatedness.py, figS_lexicon_relatedness.py/fig3_lexicon.py/figS_lexicon_alignment.py |
| `lexicon_ws_scaling_F-vitl_bv.csv` | fig3_lexicon.py, figS_lexicon_alignment.py, figS_lexicon_relatedness.py, figS_lexicon_relatedness.py/fig3_lexicon.py/figS_lexicon_alignment.py |
| `lexicon_ws_scaling_F-vits_bv-aligned.csv` | fig3_lexicon.py, figS_lexicon_alignment.py, figS_lexicon_relatedness.py, figS_lexicon_relatedness.py/fig3_lexicon.py/figS_lexicon_alignment.py |
| `lexicon_ws_scaling_F-vits_bv.csv` | fig3_lexicon.py, figS_lexicon_alignment.py, figS_lexicon_relatedness.py, figS_lexicon_relatedness.py/fig3_lexicon.py/figS_lexicon_alignment.py |
| `literature.csv` | fig2_scaling.py |
| `methods_numbers.json` | make_methods_numbers.py |
| `methods_numbers.tex` | make_methods_numbers.py |
| `pipeline_counts.json` | fig1_pipeline.py, make_pipeline_counts.py |
| `provenance_report.csv` | check_provenance.py, data.py |
| `published.csv` | check_provenance.py, data.py |
| `reeval_corrected.csv` | check_provenance.py |
| `runs.parquet` | check_provenance.py, data.py, make_encoder_grid.py, make_experiments_table.py |
| `titration.csv` | data.py |
| `utterance_rate.csv` | fig4_development.py, figS_speech_density.py |
| `wordbank_anchors.csv` | fig4_development.py, make_wordbank_40.py |
| `wordbank_anchors_40.csv` | fig4_development.py, make_wordbank_40.py |
| `wordbank_anchors_items.csv` | make_wordbank_40.py |
| `wordbank_rasch.csv` | figS_item_difficulty.py |

## Legacy / not read by the paper

`cues.csv`, `item_eval_b26.csv`, `lev_vocab_seedmean.parquet`, `lev_vocab_seedmean_freq.parquet`, `lexicon_category_structure_C8_dinov3l_grid4x4_base.csv`, `lexicon_partial_B26.csv`, `lexicon_partial_L-OTS.csv`, `lexicon_relatedness.csv`, `lexicon_rsa.csv`, `lexicon_tsne_B26_lad_base_s0.csv`, `lexicon_tsne_B26_lad_base_s0_NOUN.csv`, `lexicon_tsne_C8_dinov3l_grid4x4_base_s0_NOUN.csv`, `lexicon_ws_scaling_B26.csv`, `lexicon_ws_scaling_L-OTS.csv`
