.PHONY: install test lint format check tokenizer-check reasoning-data eval-hindi clean

PY ?= python

install:            ## editable install with dev tools
	$(PY) -m pip install -e ".[dev]"

test:               ## run the test suite
	$(PY) -m pytest

lint:               ## static checks
	ruff check src tests

format:             ## auto-fix lint issues where possible
	ruff check --fix src tests

check: lint test    ## what CI runs

tokenizer-check:    ## round-trip + fertility sanity check on the released tokenizers
	$(PY) scripts/tokenizer_report.py

reasoning-data:     ## regenerate the synthetic reasoning datasets (seed 42) and verify them
	$(PY) -m indic_lm.reasoning.generate_hindi
	$(PY) -m indic_lm.reasoning.generate_nepali
	$(PY) -m indic_lm.reasoning.verify_data
	$(PY) -m indic_lm.reasoning.deduplicate
	$(PY) -m indic_lm.reasoning.split
	$(PY) -m indic_lm.reasoning.verify_splits

eval-hindi:         ## held-out reasoning evaluation (needs models/hindi_finetuned_phase3.pt)
	$(PY) -m indic_lm.evaluation.reasoning --language hindi \
		--checkpoint models/hindi_finetuned_phase3.pt --out artifacts/eval/hindi_reasoning_qa.json

clean:
	rm -rf build dist .pytest_cache .ruff_cache src/*.egg-info
