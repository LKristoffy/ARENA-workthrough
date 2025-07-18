# Makefile (minimal, Bash-based)

SHELL := /usr/bin/env bash   # use Bash so 'source' works if you really need it
VENV_DIR := .venv

.PHONY: setup clean

setup:
	@echo '--- Ensuring uv is installed ---'
	@if ! command -v uv >/dev/null 2>&1; then \
		echo 'Installing uv…'; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	else \
		echo 'uv found at $$(command -v uv)'; \
	fi
	@echo '--- Creating virtual environment ---'
	@uv venv $(VENV_DIR)
	@echo '--- Installing dependencies ---'
	@uv pip install -r requirements.txt
	@echo "✓ Setup complete — activate with: source $(VENV_DIR)/bin/activate"

clean:
	@echo '--- Removing venv ---'
	@rm -rf $(VENV_DIR)