# SnakeRL Makefile
# Reinforcement Learning Snake Game

.PHONY: help install dev run clean lint format typecheck ci

# Default target - show help
help:
	@echo ""
	@echo "  SnakeRL - Reinforcement Learning Snake Game"
	@echo "  ============================================"
	@echo ""
	@echo "  Available commands:"
	@echo ""
	@echo "    make install   Create virtual environment and install dependencies"
	@echo "    make dev       Run development server with hot reload"
	@echo "    make run       Run production server"
	@echo "    make clean     Remove cache files and build artifacts"
	@echo "    make lint      Run Ruff linter"
	@echo "    make format    Run Ruff formatter"
	@echo "    make typecheck Run MyPy type checker"
	@echo "    make ci        Run all CI checks (lint + typecheck)"
	@echo ""
	@echo "  Usage:"
	@echo "    1. Run 'make install' to set up the environment"
	@echo "    2. Activate venv: source venv/bin/activate"
	@echo "    3. Run 'make dev' for development"
	@echo ""

# Create virtual environment and install dependencies
install:
	@echo "Creating virtual environment..."
	python3 -m venv venv
	@echo "Installing dependencies..."
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo ""
	@echo "Installation complete!"
	@echo "Activate the virtual environment with: source venv/bin/activate"

# Find available port (default 8000, or next available)
PORT ?= $(shell python3 -c "import socket; s=socket.socket(); port=8000; exec('while True:\\n try:\\n  s.bind((\"0.0.0.0\",port)); s.close(); break\\n except: port+=1'); print(port)")

# Run development server with hot reload
dev:
	@echo "Starting development server on port $(PORT)..."
	@if [ "$(PORT)" != "8000" ]; then echo "Port 8000 was in use, using $(PORT) instead"; fi
	uvicorn main:app --reload --host 0.0.0.0 --port $(PORT)

# Run production server
run:
	@echo "Starting production server on port $(PORT)..."
	@if [ "$(PORT)" != "8000" ]; then echo "Port 8000 was in use, using $(PORT) instead"; fi
	uvicorn main:app --host 0.0.0.0 --port $(PORT)

# Clean up cache files and build artifacts
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete!"

# Linting and formatting
lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy . --ignore-missing-imports

ci: lint typecheck
