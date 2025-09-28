VENV ?= .venv
PYTHON := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
ASSIGNMENT_MODULES := $(shell find assignment_1 -type f -name '*.py' ! -name '__init__.py' | sort | sed 's|^assignment_1/||; s|\.py$$||; s|/|.|g' | sed 's|^|assignment_1.|')

.PHONY: clean run-all copy-plots

clean:
	@echo "Removing generated plots, __pycache__ directories, and .DS_Store files"
	@if [ -d assignment_1/Plots ]; then \
		find assignment_1/Plots -mindepth 1 -exec rm -rf {} +; \
	fi
	@find assignment_1 utils -type d -name '__pycache__' -prune -exec rm -rf {} +
	@find . -type f -name '.DS_Store' -delete

run-all:
	@set -e; \
	echo "Running assignment modules with $(PYTHON)"; \
	for module in $(ASSIGNMENT_MODULES); do \
		echo "--- $$module"; \
		$(PYTHON) -m $$module; \
	done

copy-plots:
	@echo "Copying Plots directory to ../Advanced-Numerical-Methods---Assignment-1/"
	$(PYTHON) utils/copy_plots.py
