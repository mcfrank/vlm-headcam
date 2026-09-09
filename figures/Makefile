PY := ../.venv/bin/python
FIGS := fig1_pipeline fig2_scaling fig3_lexicon fig4_development \
        figS_diversity figS_indomain figS_alignment_encoders figS_lexicon_tsne \
        figS_lexicon_relatedness figS_levante figS_encoder_probe figS_item_difficulty \
        figS_lexicon_alignment figS_nomil figS_alignment_controls figS_kchi \
        figS_alignment_scores figS_corpus_effort figS_corpus_age figS_speech_density \
        figS_window

all: $(FIGS)

$(FIGS): %:
	@$(PY) $*.py 2>&1 | grep -vE "Glyph|post.stringData|timestamp|UserWarning|fig.savefig" || true

check:
	@cd .. && .venv/bin/python src/check_provenance.py

clean:
	rm -f out/*.pdf out/*.png

.PHONY: all check clean $(FIGS)
