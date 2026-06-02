.DEFAULT_GOAL := help
.PHONY: help install setup init doctor ui check lint fmt fmt-check type security test test-fast \
	playbook docs clean run-all panels bubbles inpaint clean-stage caption

PY := uv run

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create venv, install deps, set up pre-commit
	uv sync --all-extras
	$(PY) pre-commit install

setup: ## One-command first-time setup (deps + workspace + env check)
	bash scripts/setup.sh

init: ## Scaffold the workspace folders and seed .env
	$(PY) make-style-dataset init

doctor: ## Check this machine is ready (Python, GPU, workspace)
	$(PY) make-style-dataset doctor

ui: ## Launch the local web UI (installs the 'ui' group on first run)
	uv run --group ui make-style-dataset ui

check: lint fmt-check type security test ## Definition-of-Done gate (run before every PR)
	@echo "all checks passed"

lint: ## Ruff lint
	$(PY) ruff check .

fmt: ## Ruff format (write)
	$(PY) ruff format .

fmt-check: ## Ruff format (check only)
	$(PY) ruff format --check .

type: ## Pyright type check
	$(PY) pyright

security: ## Bandit + pip-audit
	$(PY) bandit -c pyproject.toml -r src/ -q
	$(PY) pip-audit

test: ## Full test suite with coverage gate
	$(PY) pytest --cov --cov-fail-under=90

test-fast: ## Unit tests only, parallel, no coverage
	$(PY) pytest tests/unit -n auto -q

run-all: ## Run the whole pipeline (all enabled stages)
	$(PY) make-style-dataset run-all

panels: ## Run the panel-detection stage
	$(PY) make-style-dataset panels

bubbles: ## Run the bubble-detection stage
	$(PY) make-style-dataset bubbles

inpaint: ## Run the inpainting stage
	$(PY) make-style-dataset inpaint

clean-stage: ## Run the dedup/size-filter stage (note: `clean` clears caches)
	$(PY) make-style-dataset clean

caption: ## Run the captioning / dataset-layout stage
	$(PY) make-style-dataset caption

playbook: ## Extract PLAYBOOK markers
	$(PY) python scripts/extract_playbook.py

docs: ## Serve docs locally (requires mkdocs)
	$(PY) mkdocs serve

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
