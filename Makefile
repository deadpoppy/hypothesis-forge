# Makefile for AI Co-Scientist CLI

run:
	@if [ -z "$(GOAL)" ]; then \
		echo "Usage: make run GOAL=\"your research goal\" REFERENCE_ARXIV=\"https://arxiv.org/abs/2502.18864\" [OUTPUT_DIR=results/my_run]"; \
		exit 1; \
	fi
	@if [ -z "$(REFERENCE_ARXIV)" ]; then \
		echo "Usage: make run GOAL=\"your research goal\" REFERENCE_ARXIV=\"https://arxiv.org/abs/2502.18864\" [OUTPUT_DIR=results/my_run]"; \
		exit 1; \
	fi
	python app.py --goal "$(GOAL)" --reference-arxiv "$(REFERENCE_ARXIV)" $(if $(OUTPUT_DIR),--output-dir "$(OUTPUT_DIR)",)

test:
	python -m unittest discover -s tests

.PHONY: run test
