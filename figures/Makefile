PY := ../.venv/bin/python
FIGS := fig1_pipeline fig2_encoders fig3_composite fig4_lev_scaling fig5_lexicon \
        figS1_scaling_saycam figS2_cues figS3_representation figS4_levante figS5_items figS6_development

all: $(FIGS)

$(FIGS): %:
	@$(PY) $*.py 2>&1 | grep -vE "Glyph|post.stringData|timestamp|UserWarning|fig.savefig" || true

check:
	@cd .. && .venv/bin/python src/check_provenance.py

clean:
	rm -f out/*.pdf out/*.png

.PHONY: all check clean $(FIGS)
