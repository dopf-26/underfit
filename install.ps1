# Underfit installer for Windows
#
# Usage:
#     .\install.ps1                  # full flow: install uv (if missing) + uv sync + underfit-setup
#     .\install.ps1 -NoSetup         # stop after `uv sync`, skip the underfit-setup wizard
#     .\install.ps1 -Backend sat     # opt into stable-audio-tools (default is sa3)
#
# Idempotent: re-running upgrades anything missing and leaves the rest alone.

param(
    [switch]$NoSetup,
    [ValidateSet("sa3", "sat")]
    [string]$Backend
)

$ErrorActionPreference = "Stop"

$UnderfitDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $UnderfitDir

function Say {
    param([string]$m)
    Write-Host "> $m" -ForegroundColor Cyan
}

function Err {
    param([string]$m)
    Write-Host "x $m" -ForegroundColor Red
    Write-Error $m
}

# -- 1. uv ------------------------------------------------------------------
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Say "uv not found, installing via official Astral installer ..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    # The Astral installer drops `uv` into %USERPROFILE%\.cargo\bin on Windows.
    # Prepend to PATH so this session can find it without a restart.
    $cargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
    if (Test-Path "$cargoBin\uv.exe") {
        $env:Path = "$cargoBin;$env:Path"
    }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Err "uv installed, but not on PATH. Open a new terminal and re-run."
        exit 1
    }
}
Say "uv $(uv --version) ready"

# -- 2. deps ----------------------------------------------------------------
Say "syncing dependencies (uv sync --inexact) ..."
uv sync --inexact

# -- 3. flash-attention prebuilt wheel --------------------------------------
$FlashAttnUrl = "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.6/flash_attn-2.8.3+cu130torch2.11-cp313-cp313-win_amd64.whl"
# Download the ~40MB wheel to TEMP (not the repo root) and remove it right
# after installing, so it doesn't sit around in the working directory.
$FlashAttnWhl = Join-Path $env:TEMP "flash_attn-2.8.3+cu130torch2.11-cp313-cp313-win_amd64.whl"

if (-not (Test-Path $FlashAttnWhl)) {
    Say "downloading flash-attention wheel ..."
    Invoke-WebRequest -Uri $FlashAttnUrl -OutFile $FlashAttnWhl
}
Say "installing flash-attention ..."
uv pip install --no-deps $FlashAttnWhl
Remove-Item $FlashAttnWhl -Force -ErrorAction SilentlyContinue

# -- 4. wizard --------------------------------------------------------------
if ($NoSetup) {
    Say "skipping underfit-setup (-NoSetup passed)"
    Say "done - now run .\run.ps1 to start the dashboard."
    exit 0
}
Say "launching underfit-setup ..."
if ($Backend) {
    uv run python -m underfit.cli.setup --backend $Backend
} else {
    uv run python -m underfit.cli.setup
}

Write-Host ""
Say "all done - now run .\run.ps1 to start the dashboard."
