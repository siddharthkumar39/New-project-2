# Starts SnapSight from this project folder, even if PowerShell was opened elsewhere.
# Run it with: .\start-snapsight.ps1

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "The project environment is missing." -ForegroundColor Red
    Write-Host "Run this once from the project folder: python -m venv .venv"
    exit 1
}

Set-Location $projectRoot
Write-Host "Starting SnapSight at http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "Keep this window open while using SnapSight. Press Ctrl+C to stop it." -ForegroundColor Yellow
& $python -m uvicorn backend.main:app --reload
