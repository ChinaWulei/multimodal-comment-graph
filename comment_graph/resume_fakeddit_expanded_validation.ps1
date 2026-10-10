param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$RunDirectory,
    [switch]$CheckOnly
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath ([IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..")))
$runPath = [IO.Path]::GetFullPath($RunDirectory)
$plan = Get-Content -LiteralPath (Join-Path $runPath "plan.json") -Raw | ConvertFrom-Json
$amendment = Get-Content -LiteralPath (Join-Path $runPath "context_selection_amendment.json") -Raw | ConvertFrom-Json
$previous = Get-Content -LiteralPath (Join-Path $runPath "status.json") -Raw | ConvertFrom-Json
if ($previous.stage -ne "failed") { throw "Only a terminal failed pipeline can be recovered" }
if ($null -ne (Get-Process -Id $previous.pid -ErrorAction SilentlyContinue)) { throw "Previous pipeline PID is still present" }
if ((Get-FileHash -LiteralPath $plan.formal_config -Algorithm SHA256).Hash -ne $amendment.amended_classifier_config_sha256) {
    throw "Amended classifier configuration changed"
}
if (Test-Path -LiteralPath (Join-Path $runPath "recovery_plan.json")) { throw "Recovery plan already exists; inspect its outcome first" }
if ($CheckOnly) { Write-Output "Recovery preflight passed; original settings and existing results retained."; exit 0 }
$lock = [IO.File]::Open((Join-Path $runPath "pipeline.lock"), [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
function Set-Stage([string]$Stage, [string]$Detail) {
    [ordered]@{updated_at=(Get-Date).ToString("o");pid=$PID;stage=$Stage;detail=$Detail} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runPath "status.json") -Encoding UTF8
    Write-Output "$Stage : $Detail"
}
try {
    [ordered]@{created_at=(Get-Date).ToString("o");pid=$PID;previous_failure=$previous;
        reason="263-character Windows checkpoint path reproduced as failing; compact hashed filenames and atomic saves added";
        modeling_changes="none";seeds=$plan.seeds;classifier_config_sha256=$amendment.amended_classifier_config_sha256;
        policy="Verify and skip completed summaries/checkpoints; rerun only failed empty arm and not-yet-started arms; preserve original logs"} |
        ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $runPath "recovery_plan.json") -Encoding UTF8
    Set-Stage "recovery_graph_audit" "All graphs already generated; no regeneration"
    & $Python -u .\comment_graph\audit_generated_grafts.py --config $plan.formal_config --split train --split validate --split test --output-json (Join-Path $runPath "recovery_graft_audit.json") *> (Join-Path $runPath "recovery_audit.log")
    if ($LASTEXITCODE -ne 0) { throw "Recovery graph audit failed" }
    $audit = Get-Content -LiteralPath (Join-Path $runPath "recovery_graft_audit.json") -Raw | ConvertFrom-Json
    foreach ($name in @("train", "validate", "test")) {
        if (-not $audit.splits.$name.all_predicted_parents_preserved) { throw "Missing predicted parents in $name" }
    }
    Set-Stage "validation_experiments" "Recovered after file-path failure; same arms and seeds"
    foreach ($seed in $plan.seeds) {
        & .\comment_graph\run_fakeddit_expanded_gahr_validation.ps1 -Python $Python -Config $plan.formal_config -Seed $seed -RunRoot (Join-Path $runPath "validation_seed$seed") -ResumeCompleted *> (Join-Path $runPath "recovery_validation_seed$seed.log")
    }
    Set-Stage "validation_complete" "All declared validation arms complete; frozen evaluation can proceed"
} catch {
    Set-Stage "failed" $_.Exception.Message
    throw
} finally { $lock.Dispose() }
