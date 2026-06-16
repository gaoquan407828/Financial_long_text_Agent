param(
    [string]$CacheDir = "D:\.uv_cache"
)

$ErrorActionPreference = "Stop"
$env:UV_CACHE_DIR = $CacheDir

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not available in PATH. Please install uv first: https://docs.astral.sh/uv/"
}

if (-not (Test-Path ".venv")) {
    uv venv
}

uv pip install -r requirements.txt

Write-Host "uv environment is ready."
Write-Host "Activate in PowerShell with: .\.venv\Scripts\Activate.ps1"
Write-Host "Or run commands without activation, for example: uv run python scripts/build_index.py"
