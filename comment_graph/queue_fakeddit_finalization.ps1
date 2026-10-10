param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$SourceRun,
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [Parameter(Mandatory=$true)][int]$PipelineProcessId,
    [Parameter(Mandatory=$true)][string]$PipelineStartedAt,
    [switch]$CheckOnly
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath ([IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..")))
$source = [IO.Path]::GetFullPath($SourceRun)
$destination = [IO.Path]::GetFullPath($OutputDirectory)
$script = Join-Path $PSScriptRoot "finalize_fakeddit_structure.py"
foreach ($path in @($Python, $script, (Join-Path $source "plan.json"))) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing prerequisite: $path" }
}
if (Test-Path -LiteralPath $destination) { throw "Output already exists: $destination" }
$process = Get-Process -Id $PipelineProcessId -ErrorAction SilentlyContinue
if ($null -ne $process) {
    $expected = [DateTime]::Parse($PipelineStartedAt)
    if ($process.ProcessName -ne "pwsh" -or [Math]::Abs(($process.StartTime - $expected).TotalSeconds) -gt 2) {
        throw "Pipeline process identity mismatch"
    }
} else {
    $state = Get-Content -LiteralPath (Join-Path $source "status.json") -Raw | ConvertFrom-Json
    if ($state.stage -ne "validation_complete") { throw "Pipeline ended without completed validation" }
}
if ($CheckOnly) { Write-Output "Queue preflight passed; no training or evaluation started."; exit 0 }
# Held for the full queue lifetime; a second waiter cannot start duplicate GPU work.
$lock = [IO.File]::Open((Join-Path $source "finalization.lock"), [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
try {
    Write-Output "Waiting for verified pipeline PID=$PipelineProcessId; GPU not acquired."
    if ($null -ne $process) { $process.WaitForExit() }
    & $Python -u $script --source-run $source --output-dir $destination
    if ($LASTEXITCODE -ne 0) { throw "Finalization failed; inspect finalization log/status." }
} finally {
    $lock.Dispose()
}
