.PHONY: help install test lint format clean docs docs-dev docs-build docs-preview

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linters"
	@echo "  make format        - Format code"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make docs-dev      - Start documentation dev server"
	@echo "  make docs-build    - Build documentation"
	@echo "  make docs-preview  - Preview built documentation"

install:
	pip install -e .
	pip install -r requirements.txt

test:
	pytest tests/

lint:
	flake8 relax/
	mypy relax/

format: # develop ## Code format using Ruff
	ruff check --fix relax tests scripts
	ruff format relax tests scripts

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Documentation commands
docs-dev:
	cd docs && npm install && npm run docs:dev

docs-build:
	cd docs && npm install && npm run docs:build

docs-preview:
	cd docs && npm run docs:preview

docs-install:
	cd docs && npm install
