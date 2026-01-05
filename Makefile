# SnakeRL Makefile
# Reinforcement Learning Snake Game

.PHONY: help install dev run clean

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

# Run development server with hot reload
dev:
	@echo "Starting development server..."
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run production server
run:
	@echo "Starting production server..."
	uvicorn main:app --host 0.0.0.0 --port 8000

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
