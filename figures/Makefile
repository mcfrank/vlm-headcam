PY := ../.venv/bin/python
FIGS := fig1_pipeline fig2_scaling fig3_development fig4_lexicon \
        figS1_alignment_encoders figS2_diversity figS3_levante_full figS4_items figS5_lexicon_encoders

all: $(FIGS)

$(FIGS): %:
	@$(PY) $*.py 2>&1 | grep -vE "Glyph|post.stringData|timestamp|UserWarning|fig.savefig" || true

check:
	@cd .. && .venv/bin/python src/check_provenance.py

clean:
	rm -f out/*.pdf out/*.png

.PHONY: all check clean $(FIGS)
