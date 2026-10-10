param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$RunDirectory,
    [int]$TrainingProcessId = 0,
    [string]$TrainingStartedAt = "",
    [int[]]$Seeds = @(42, 13, 24),
    [ValidateSet("news_root", "observed_boundary")][string]$DepthReference = "news_root",
    [ValidatePattern('^[a-z0-9_]+$')][string]$OutputTag = "expanded_bestval_depthbounded",
    [string]$GraphConfig = ".\comment_graph\config_fakeddit_structure_expanded.yaml",
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\.."))
Set-Location -LiteralPath $projectRoot
$runPath = [IO.Path]::GetFullPath($RunDirectory)
$adapter = Join-Path $projectRoot "comment_generator\partial_observation\outputs\qwen2_5_vl_3b_structure_comment_official_expanded_lora"
$generateScript = Join-Path $PSScriptRoot "generate_official_expanded_structured_graphs.ps1"
$validationScript = Join-Path $projectRoot "comment_graph\run_fakeddit_expanded_gahr_validation.ps1"
foreach ($path in @($Python, $generateScript, $validationScript, $GraphConfig)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing prerequisite: $path" }
}
if ($Seeds.Count -eq 0 -or @($Seeds | Select-Object -Unique).Count -ne $Seeds.Count) {
    throw "Seeds must be a nonempty unique list"
}
if ($TrainingProcessId -ne 0 -and [string]::IsNullOrWhiteSpace($TrainingStartedAt)) {
    throw "TrainingStartedAt is required to guard against PID reuse"
}
if ($CheckOnly) {
    Write-Output "Preflight passed. Seeds=$($Seeds -join ','); generation then validation only; no test evaluation."
    exit 0
}
New-Item -ItemType Directory -Force -Path $runPath | Out-Null
# An exclusive open prevents a second copy from running this pipeline at the
# same time. A completed/failed plan is kept immutable; use a fresh directory.
$lockPath = Join-Path $runPath "pipeline.lock"
$lock = [IO.File]::Open($lockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
$planPath = Join-Path $runPath "plan.json"
if (Test-Path -LiteralPath $planPath) {
    $lock.Dispose()
    throw "Existing pipeline plan. Use a fresh RunDirectory to preserve results."
}
$plan = [ordered]@{
    created_at = (Get-Date).ToString("o")
    training_pid = $TrainingProcessId
    training_started_at = $TrainingStartedAt
    adapter = $adapter
    seeds = @($Seeds)
    arms = @("real_tree_typed_concat", "real_tree_typed_global_local", "generated_tree_typed_concat", "generated_tree_typed_global_local")
    selection = "validation macro-F1; paired generated-news coverage; test not loaded by this runner"
    graph_structure = "observed_tree"
    message_passing = "typed_rgcn"
    formal_config = $GraphConfig
    depth_reference = $DepthReference
    generated_output_tag = $OutputTag
    generator_config_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $PSScriptRoot "..\config\generator_structure_official_expanded.yaml")).Hash
    classifier_config_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $GraphConfig).Hash
}
$plan | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $planPath -Encoding UTF8
function Set-Stage([string]$Stage, [string]$Detail) {
    [ordered]@{updated_at=(Get-Date).ToString("o"); pid=$PID; stage=$Stage; detail=$Detail} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runPath "status.json") -Encoding UTF8
    Write-Output "$Stage : $Detail"
}
try {
    if ($TrainingProcessId -ne 0) {
        $training = Get-Process -Id $TrainingProcessId -ErrorAction SilentlyContinue
        if ($null -ne $training) {
            $expected = [DateTime]::Parse($TrainingStartedAt)
            if ($training.ProcessName -ne "python" -or [Math]::Abs(($training.StartTime - $expected).TotalSeconds) -gt 2) {
                throw "PID identity does not match the authorized training process"
            }
            Set-Stage "waiting_for_training" "PID=$TrainingProcessId"
            # Runs in this background pipeline, so no GPU is acquired while
            # the existing training process still owns the model.
            $training.WaitForExit()
        }
    }
    Set-Stage "checking_adapter" "Require completed official export and matching validation-selected weights"
    foreach ($name in @("adapter_config.json", "adapter_model.safetensors", "official_split.json", "train_metrics.json", "run_config.json", "preprocessor_config.json")) {
        if (-not (Test-Path -LiteralPath (Join-Path $adapter $name) -PathType Leaf)) { throw "Incomplete adapter export: $name" }
    }
    $split = Get-Content -LiteralPath (Join-Path $adapter "official_split.json") -Raw | ConvertFrom-Json
    if ($split.training_mode -ne "official" -or $split.news_overlap -ne 0 -or $null -eq $split.best_validation_loss) {
        throw "Invalid official training metadata"
    }
    $selectedWeight = Join-Path $split.best_model_checkpoint "adapter_model.safetensors"
    $exportedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $adapter "adapter_model.safetensors")).Hash
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $selectedWeight).Hash -ne $exportedHash) {
        throw "Root adapter differs from the validation-selected checkpoint"
    }
    [ordered]@{best_checkpoint=$split.best_model_checkpoint; best_validation_loss=$split.best_validation_loss; exported_sha256=$exportedHash} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runPath "adapter_verification.json") -Encoding UTF8
    Set-Stage "generating_graphs" "train, validate, test; labels excluded from generation"
    & $generateScript -Python $Python -DepthReference $DepthReference -OutputTag $OutputTag *> (Join-Path $runPath "generation.log")
    Set-Stage "validation_experiments" "Four paired arms for each of the predeclared seeds"
    foreach ($seed in $Seeds) {
        $seedRoot = Join-Path $runPath "validation_seed$seed"
        & $validationScript -Python $Python -Config $GraphConfig -Seed $seed -RunRoot $seedRoot *> (Join-Path $runPath "validation_seed$seed.log")
    }
    Set-Stage "validation_complete" "All predeclared validation runs completed; frozen test evaluation pending review of artifacts"
} catch {
    Set-Stage "failed" $_.Exception.Message
    throw
} finally {
    $lock.Dispose()
}
