param(
    [string]$Python = "python",
    [int[]]$Seeds = @(42, 13, 24),
    [string]$OutputDirectory = ".\comment_graph\outputs\fakeddit_ancestry_anchor_validation_20261008"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$config = Join-Path $PSScriptRoot "config_fakeddit_structure_expanded_generateddepth.yaml"
$trainer = Join-Path $PSScriptRoot "train.py"
$output = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $OutputDirectory))
New-Item -ItemType Directory -Force -Path $output | Out-Null

foreach ($seed in $Seeds) {
    $stem = "ancestry_anchor_seed${seed}"
    $summary = Join-Path $output "${stem}.json"
    $checkpointDir = Join-Path $output $stem
    if (Test-Path -LiteralPath $summary -PathType Leaf) {
        $existing = Get-Content -LiteralPath $summary -Raw | ConvertFrom-Json
        if ($existing.seed -ne $seed -or $existing.mode -ne "real_generated" -or
            $existing.graph_structure -ne "observed_tree" -or
            $existing.message_passing -ne "typed_rgcn" -or
            $existing.decision_mode -ne "concat" -or
            $existing.generated_reliability -ne "ancestry_anchor" -or
            $existing.max_real_comments -ne 16 -or
            -not (Test-Path -LiteralPath $existing.checkpoint -PathType Leaf)) {
            throw "Existing run mismatch: $summary"
        }
        Write-Host "Verified completed: $stem"
        continue
    }
    if (Test-Path -LiteralPath $checkpointDir) {
        $files = @(Get-ChildItem -LiteralPath $checkpointDir -Force)
        if ($files.Count -gt 0) { throw "Incomplete nonempty run directory: $checkpointDir" }
    }
    Write-Host "Training validation-only: $stem"
    & $Python $trainer --config $config --mode real_generated `
        --graph-structure observed_tree --message-passing typed_rgcn `
        --pooling-mode mean --decision-mode concat `
        --generated-reliability ancestry_anchor `
        --max-real-comments 16 --max-generated-comments 5 `
        --seed $seed --allow-formal-training `
        --checkpoint-dir $checkpointDir --summary-json $summary
    if ($LASTEXITCODE -ne 0) { throw "Run failed: $stem (exit $LASTEXITCODE)" }
    if (-not (Test-Path -LiteralPath $summary -PathType Leaf)) { throw "Missing summary: $summary" }
}
Write-Host "Validation-only ancestry-anchor runs complete: $output"
