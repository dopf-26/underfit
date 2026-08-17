# Underfit dashboard launcher for Windows
#
# Usage: .\run.ps1 [server.py args...]
#
# Runs dashboard/server.py inside the venv created by .\install.ps1.

param(
    [Parameter(Position = 0, Mandatory = $false, ValueFromRemainingArguments = $true)]
    [string[]]$ServerArgs
)

$ErrorActionPreference = "Stop"

$UnderfitDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $UnderfitDir

$PythonExe = Join-Path $UnderfitDir ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Host "x no .venv found - run .\install.ps1 first." -ForegroundColor Red
    exit 1
}

& $PythonExe dashboard/server.py @ServerArgs
