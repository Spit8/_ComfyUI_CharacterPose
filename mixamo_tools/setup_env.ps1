# Create a local venv for mixamo_tools and install deps.
# Usage (from this folder):
#   .\setup_env.ps1
#   .\venv\Scripts\python.exe -m export_views --positions 16 --view SE

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Py = $null
foreach ($c in @(
    "C:\Users\wiwil\AppData\Local\Programs\Python\Python310\python.exe",
    "C:\Users\wiwil\AppData\Local\Python\pythoncore-3.12-64\python.exe",
    (Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)
)) {
    if ($c -and (Test-Path $c)) { $Py = $c; break }
}
if (-not $Py) { throw "No Python found" }

Write-Host "Using $Py"
& $Py -m venv "$Root\venv"
& "$Root\venv\Scripts\python.exe" -m pip install --upgrade pip
& "$Root\venv\Scripts\python.exe" -m pip install -r "$Root\requirements.txt"
Write-Host ""
Write-Host "Done. Run:"
Write-Host "  .\venv\Scripts\python.exe -m export_views --positions 16 --view SE"
