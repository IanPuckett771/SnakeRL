# SnakeRL PowerShell Makefile Equivalent
# Usage: .\Makefile.ps1 [command]
# Commands: install, dev, run, clean, help

param(
    [Parameter(Position=0)]
    [ValidateSet("install", "dev", "run", "clean", "help")]
    [string]$Command = "help"
)

function Show-Help {
    Write-Host ""
    Write-Host "  SnakeRL - Reinforcement Learning Snake Game" -ForegroundColor Cyan
    Write-Host "  ============================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Available commands:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "    .\Makefile.ps1 install   Create virtual environment and install dependencies" -ForegroundColor White
    Write-Host "    .\Makefile.ps1 dev       Run development server with hot reload" -ForegroundColor White
    Write-Host "    .\Makefile.ps1 run       Run production server" -ForegroundColor White
    Write-Host "    .\Makefile.ps1 clean     Remove cache files and build artifacts" -ForegroundColor White
    Write-Host ""
    Write-Host "  Usage:" -ForegroundColor Yellow
    Write-Host "    1. Run '.\Makefile.ps1 install' to set up the environment" -ForegroundColor White
    Write-Host "    2. Activate venv: .\venv\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host "    3. Run '.\Makefile.ps1 dev' for development" -ForegroundColor White
    Write-Host ""
}

function Install-Dependencies {
    Write-Host "Creating virtual environment..." -ForegroundColor Green
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to create virtual environment. Make sure Python is installed." -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Installing dependencies..." -ForegroundColor Green
    .\venv\Scripts\pip install --upgrade pip
    .\venv\Scripts\pip install -r requirements.txt
    
    Write-Host ""
    Write-Host "Installation complete!" -ForegroundColor Green
    Write-Host "Activate the virtual environment with: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
}

function Start-DevServer {
    Write-Host "Starting development server..." -ForegroundColor Green
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
}

function Start-ProductionServer {
    Write-Host "Starting production server..." -ForegroundColor Green
    uvicorn main:app --host 0.0.0.0 --port 8000
}

function Clean-BuildArtifacts {
    Write-Host "Cleaning up..." -ForegroundColor Green
    
    # Remove __pycache__ directories
    Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    
    # Remove .pyc files
    Get-ChildItem -Path . -Recurse -File -Filter "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue
    
    # Remove .pyo files
    Get-ChildItem -Path . -Recurse -File -Filter "*.pyo" | Remove-Item -Force -ErrorAction SilentlyContinue
    
    # Remove .pytest_cache directories
    Get-ChildItem -Path . -Recurse -Directory -Filter ".pytest_cache" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    
    # Remove .egg-info directories
    Get-ChildItem -Path . -Recurse -Directory -Filter "*.egg-info" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    
    # Remove .mypy_cache directories
    Get-ChildItem -Path . -Recurse -Directory -Filter ".mypy_cache" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    
    Write-Host "Clean complete!" -ForegroundColor Green
}

# Execute the requested command
switch ($Command) {
    "install" { Install-Dependencies }
    "dev" { Start-DevServer }
    "run" { Start-ProductionServer }
    "clean" { Clean-BuildArtifacts }
    "help" { Show-Help }
    default { Show-Help }
}
