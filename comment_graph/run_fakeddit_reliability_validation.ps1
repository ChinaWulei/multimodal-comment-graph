param(
    [string]$Python = "python",
    [int[]]$Seeds = @(42, 13, 24),
    [ValidateSet("none", "quality", "confidence", "quality_confidence", "placement", "both")]
    [string[]]$Modes = @("none", "quality", "confidence", "quality_confidence"),
    [string]$OutputDirectory = ".\comment_graph\outputs\fakeddit_reliability_validation_20261008_confidence"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$config = Join-Path $PSScriptRoot "config_fakeddit_structure_expanded_generateddepth.yaml"
$trainer = Join-Path $PSScriptRoot "train.py"
$output = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $OutputDirectory))
if (-not (Test-Path -LiteralPath $config)) { throw "Missing config: $config" }
if (-not (Test-Path -LiteralPath $trainer)) { throw "Missing trainer: $trainer" }
New-Item -ItemType Directory -Force -Path $output | Out-Null

foreach ($seed in $Seeds) {
    foreach ($mode in $Modes) {
        $stem = "${mode}_seed${seed}"
        $summary = Join-Path $output "${stem}.json"
        $checkpoints = Join-Path $output $stem
        if (Test-Path -LiteralPath $summary) {
            $existing = Get-Content -LiteralPath $summary -Raw | ConvertFrom-Json
            if ($existing.seed -ne $seed -or $existing.generated_reliability -ne $mode -or
                -not (Test-Path -LiteralPath $existing.checkpoint)) {
                throw "Existing summary/checkpoint mismatch: $summary"
            }
            Write-Host "Verified completed: $stem"
            continue
        }
        if (Test-Path -LiteralPath $checkpoints) {
            $files = @(Get-ChildItem -LiteralPath $checkpoints -Force)
            if ($files.Count -gt 0) {
                throw "Incomplete nonempty run directory; inspect manually: $checkpoints"
            }
        }
        Write-Host "Training validation-only: $stem"
        & $Python $trainer --config $config --mode real_generated `
            --graph-structure observed_tree --message-passing typed_rgcn `
            --pooling-mode mean --decision-mode concat `
            --generated-reliability $mode --seed $seed --allow-formal-training `
            --checkpoint-dir $checkpoints --summary-json $summary
        if ($LASTEXITCODE -ne 0) { throw "Run failed: $stem (exit $LASTEXITCODE)" }
        if (-not (Test-Path -LiteralPath $summary)) { throw "Missing run summary: $summary" }
    }
}
Write-Host "Validation-only ablations complete: $output"
