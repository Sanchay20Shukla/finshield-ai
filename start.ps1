param([int]$Port = 8000, [switch]$RebuildData)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create Python environment.' }
    & $pythonPath -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Could not install dependencies.' }
}
if ($RebuildData -or -not (Test-Path -LiteralPath 'data/processed/analytics.duckdb')) {
    & $pythonPath -m src.data.warehouse --csv data/raw/paysim.csv
    if ($LASTEXITCODE -ne 0) { throw 'PaySim warehouse build failed.' }
}
Write-Host "FinShield: http://127.0.0.1:$Port/platform/"
& $pythonPath -m uvicorn api.main:app --host 127.0.0.1 --port $Port
