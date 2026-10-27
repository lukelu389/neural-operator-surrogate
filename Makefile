PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: help install data test lint typecheck check final-eval reproduce

help:
	@echo "install     Install the package and development tools into the active Python"
	@echo "check       Check formatting, lint, types, and tests"
	@echo "test        Run the test suite"
	@echo "lint        Check formatting and lint"
	@echo "typecheck   Check public Python types"
	@echo "data        Generate datasets (available at T1.4)"
	@echo "final-eval  Evaluate frozen models (available at T3.4)"
	@echo "reproduce   Rebuild headline results (available at T7.2)"

install:
	$(PYTHON) -m pip install -e '.[dev]'

data:
	$(PYTHON) scripts/make_data.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy

check: lint typecheck test

final-eval:
	$(PYTHON) scripts/final_eval.py

reproduce:
	$(PYTHON) scripts/reproduce.py
