# Makefile for AI Co-Scientist CLI

run:
	@if [ -z "$(GOAL)" ]; then \
		echo "Usage: make run GOAL=\"your research goal\" [OUTPUT_DIR=results/my_run]"; \
		exit 1; \
	fi
	python app.py --goal "$(GOAL)" $(if $(OUTPUT_DIR),--output-dir "$(OUTPUT_DIR)",)

test:
	python -m unittest discover -s tests

.PHONY: run test
