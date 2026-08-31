PY := ../.venv/bin/python
FIGS := fig1_pipeline fig2_scaling fig3_development fig4_lexicon \
        figS1_scaling_saycam figS2_representation figS3_levante figS4_items figS5_lev_scaling figS6_diversity

all: $(FIGS)

$(FIGS): %:
	@$(PY) $*.py 2>&1 | grep -vE "Glyph|post.stringData|timestamp|UserWarning|fig.savefig" || true

check:
	@cd .. && .venv/bin/python src/check_provenance.py

clean:
	rm -f out/*.pdf out/*.png

.PHONY: all check clean $(FIGS)
