# ==============================================================================
# PROJECT LEDGER: Developer Makefile
# Multi-Service Financial Document Intelligence RAG Agent
# ==============================================================================

SHELL := /bin/bash
PYTHON := python3.11
VENV_DIR := .venv
VENV_ACTIVATE := $(VENV_DIR)/bin/activate

.PHONY: help venv env install-dev lint format test up down logs status clean

help:
	@echo "================================================================================"
	@echo "PROJECT LEDGER - Available Developer Commands"
	@echo "================================================================================"
	@echo "  make venv         Create local Python 3.11 virtual environment"
	@echo "  make env          Initialize .env file from .env.example (if not exists)"
	@echo "  make install-dev  Install root dev tools (ruff, pytest, pydantic, httpx)"
	@echo "  make lint         Run Ruff linter on all code"
	@echo "  make format       Run Ruff code formatter"
	@echo "  make test         Execute automated test suite"
	@echo "  make up           Start all microservices via Docker Compose"
	@echo "  make down         Stop all running microservices"
	@echo "  make logs         Tail logs of all Docker Compose services"
	@echo "  make status       Check health and status of Docker services"
	@echo "  make clean        Remove cache files, byte-code, and temp test artifacts"
	@echo "================================================================================"

venv:
	@echo "Creating Python 3.11 virtual environment in $(VENV_DIR)..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo ""
	@echo "Virtual environment created successfully!"
	@echo "Activate it using: source $(VENV_ACTIVATE)"

env:
	@if [ ! -f .env ]; then \
		echo "Creating .env from .env.example..."; \
		cp .env.example .env; \
		echo ".env created! Edit it to insert your private API keys."; \
	else \
		echo ".env already exists. Skipping copy."; \
	fi

install-dev:
	@if [ ! -f $(VENV_ACTIVATE) ]; then \
		echo "Virtualenv not found. Please run 'make venv' first."; \
		exit 1; \
	fi
	@echo "Installing root development tools into virtualenv..."
	source $(VENV_ACTIVATE) && pip install --upgrade pip && pip install ruff pytest httpx pydantic python-dotenv fastapi "uvicorn[standard]" simpleeval

lint:
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check .; \
	elif [ -f $(VENV_ACTIVATE) ]; then \
		source $(VENV_ACTIVATE) && ruff check .; \
	else \
		echo "Ruff is not installed. Run 'make install-dev' first."; \
	fi

format:
	@if command -v ruff >/dev/null 2>&1; then \
		ruff format .; \
	elif [ -f $(VENV_ACTIVATE) ]; then \
		source $(VENV_ACTIVATE) && ruff format .; \
	else \
		echo "Ruff is not installed. Run 'make install-dev' first."; \
	fi

test:
	@if [ -d "tests" ]; then \
		if command -v pytest >/dev/null 2>&1; then \
			pytest tests/ -v; \
		elif [ -f $(VENV_ACTIVATE) ]; then \
			source $(VENV_ACTIVATE) && pytest tests/ -v; \
		else \
			echo "Pytest not found. Run 'make install-dev' first."; \
		fi \
	else \
		echo "No tests/ directory found yet. Tests will be added in Step 2."; \
	fi

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

status:
	docker compose ps

clean:
	@echo "Cleaning Python caches and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete
	find . -type f -name "*.log" -delete
	@echo "Clean complete."
