PY := ../.venv/bin/python
FIGS := fig1_pipeline fig2_scaling fig3_ladder fig4_cues fig5_representation fig6_levante fig7_diversity fig8_items fig9_development fig10_encoders fig11_lev_scaling

all: $(FIGS)

$(FIGS): %:
	@$(PY) $*.py 2>&1 | grep -vE "Glyph|post.stringData|timestamp|UserWarning|fig.savefig" || true

check:
	@cd .. && .venv/bin/python src/check_provenance.py

clean:
	rm -f out/*.pdf out/*.png

.PHONY: all check clean $(FIGS)
