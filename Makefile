# Makefile for setting up the project environment

# Define the virtual environment directory
VENV_DIR := .venv
# Uv is installed in ~/.cargo/bin/uv by default on macOS/Linux
UV := $(HOME)/.cargo/bin/uv

.PHONY: setup clean

setup:
	@echo "--- Checking for uv and installing if not present ---"
	@if ! [ -x "$(UV)" ]; then \
		echo "uv not found, installing..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	else \
		echo "uv is already installed."; \
	fi
	@echo "--- Creating virtual environment using uv ---"
	@uv venv
	@echo "--- Installing dependencies from requirements.txt ---"
	@uv pip install -r requirements.txt
	@echo "--- Setup complete. Activate the venv with 'source $(VENV_DIR)/bin/activate' ---"

clean:
	@echo "--- Removing virtual environment ---"
	@rm -rf $(VENV_DIR)
	@echo "--- Clean complete ---"
