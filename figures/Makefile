PY := ../.venv/bin/python
FIGS := fig1_pipeline fig2_scaling fig3_development fig4_lexicon \
        figS2_diversity figS3_indomain figS4_probe figS5_alignment_encoders \
        figS6_tsne_encoders figS7_lexicon_encoders figS8_levante figS9_items figS10_lexicon_alignment

all: $(FIGS)

$(FIGS): %:
	@$(PY) $*.py 2>&1 | grep -vE "Glyph|post.stringData|timestamp|UserWarning|fig.savefig" || true

check:
	@cd .. && .venv/bin/python src/check_provenance.py

clean:
	rm -f out/*.pdf out/*.png

.PHONY: all check clean $(FIGS)
