# ============================================
# DocuMind-Agent Makefile
# ============================================

.PHONY: help install install-dev install-all test lint format clean run docker-build docker-up docker-down

# Default target
.DEFAULT_GOAL := help

# ============================================
# Help
# ============================================
help: ## Show this help message
	@echo "DocuMind-Agent Development Commands"
	@echo "===================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============================================
# Installation
# ============================================
install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 isort mypy pre-commit pylint

install-all: install-dev ## Install all dependencies including docs
	pip install mkdocs mkdocs-material mkdocstrings mkdocstrings-python

setup: install-dev ## Setup development environment
	pre-commit install
	mkdir -p data/vector_store/chroma
	mkdir -p models/cache
	mkdir -p logs
	cp .env.example .env
	@echo "✅ Development environment ready!"

# ============================================
# Testing
# ============================================
test: ## Run all tests
	pytest tests/ -v

test-cov: ## Run tests with coverage report
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing -v

test-fast: ## Run tests without slow tests
	pytest tests/ -v -m "not slow"

test-integration: ## Run only integration tests
	pytest tests/ -v -m "integration"

test-unit: ## Run only unit tests
	pytest tests/ -v -m "unit"

# ============================================
# Code Quality
# ============================================
lint: ## Run all linters
	@echo "Running flake8..."
	flake8 src/ streamlit_app/ tests/ --max-line-length=100
	@echo "Running mypy..."
	#mypy src/ --ignore-missing-imports
	@echo "Running pylint..."
	pylint src/ streamlit_app/ --max-line-length=100 --disable=C0114,C0115,C0116

format: ## Format code with black and isort
	@echo "Running black..."
	black src/ streamlit_app/ tests/ --line-length=100
	@echo "Running isort..."
	isort src/ streamlit_app/ tests/ --profile=black

format-check: ## Check code formatting without modifying
	black --check src/ streamlit_app/ tests/ --line-length=100
	isort --check-only src/ streamlit_app/ tests/ --profile=black

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

security: ## Run security checks
	bandit -r src/ streamlit_app/ -f screen
	safety check

# ============================================
# Running
# ============================================
run: ## Run Streamlit app locally
	streamlit run streamlit_app/app.py

run-debug: ## Run app with debug logging
	LOG_LEVEL=DEBUG streamlit run streamlit_app/app.py

# ============================================
# Docker
# ============================================
docker-build: ## Build Docker image
	docker-compose build

docker-up: ## Start all services
	docker-compose up -d

docker-down: ## Stop all services
	docker-compose down

docker-logs: ## View service logs
	docker-compose logs -f documind

docker-shell: ## Open shell in container
	docker-compose exec documind /bin/bash

docker-rebuild: ## Rebuild and restart services
	docker-compose down
	docker-compose build --no-cache
	docker-compose up -d

# ============================================
# Ollama Management
# ============================================
ollama-pull: ## Pull required Ollama models
	ollama pull phi3:mini
	ollama pull llama3.1:8b
	ollama pull mistral:7b

ollama-list: ## List available Ollama models
	ollama list

# ============================================
# Data Management
# ============================================
clean-data: ## Clean vector store data
	rm -rf data/vector_store/chroma/*
	@echo "✅ Vector store cleaned"

clean-cache: ## Clean model cache
	rm -rf models/cache/*
	@echo "✅ Model cache cleaned"

clean-logs: ## Clean log files
	rm -rf logs/*
	@echo "✅ Logs cleaned"

# ============================================
# Cleaning
# ============================================
clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "✅ Build artifacts cleaned"

# ============================================
# Documentation
# ============================================
docs-serve: ## Serve documentation locally
	mkdocs serve

docs-build: ## Build documentation
	mkdocs build

docs-deploy: ## Deploy documentation to GitHub Pages
	mkdocs gh-deploy

# ============================================
# Release
# ============================================
version: ## Show current version
	@python -c "import tomli; print(tomli.load(open('pyproject.toml', 'rb'))['project']['version'])"

check-release: format-check lint test ## Run all checks before release
	@echo "✅ All checks passed! Ready for release."

# ============================================
# CI/CD
# ============================================
ci-test: ## Run tests as in CI
	pytest tests/ --cov=src --cov-report=xml --cov-report=term-missing -v

ci-lint: ## Run linting as in CI
	black --check src/ streamlit_app/ tests/
	isort --check-only src/ streamlit_app/ tests/
	flake8 src/ streamlit_app/ tests/ --max-line-length=150 --ignore=E203,W503
	#mypy src/ --ignore-missing-imports --python-version 3.11

# ============================================
# Development Helpers
# ============================================
notebook: ## Start Jupyter notebook
	jupyter notebook

shell: ## Start Python shell with project context
	python -i -c "import sys; sys.path.insert(0, '.'); from src.agent.agent_builder import *"

tree: ## Show project structure
	tree -I '__pycache__|*.pyc|venv|env|.git|models|data' -L 3
