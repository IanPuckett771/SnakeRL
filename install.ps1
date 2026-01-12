# SnakeRL Installation Script for Windows
# Equivalent to: make install

Write-Host "Creating virtual environment..." -ForegroundColor Green
python -m venv venv

Write-Host "Installing dependencies..." -ForegroundColor Green
.\venv\Scripts\pip install --upgrade pip
.\venv\Scripts\pip install -r requirements.txt

Write-Host ""
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "Activate the virtual environment with: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
