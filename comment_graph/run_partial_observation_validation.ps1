param(
    [string]$Python = "python",
    [int[]]$Seeds = @(42, 13, 24),
    [string]$OutputDirectory = ".\comment_graph\outputs\fakeddit_partial_real16_validation_20261008"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$config = Join-Path $PSScriptRoot "config_fakeddit_structure_expanded_generateddepth.yaml"
$trainer = Join-Path $PSScriptRoot "train.py"
$output = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $OutputDirectory))
New-Item -ItemType Directory -Force -Path $output | Out-Null
$arms = @(
    @{ name = "real_only"; mode = "real_only"; decision = "concat"; reliability = "none" },
    @{ name = "generated_plain"; mode = "real_generated"; decision = "concat"; reliability = "none" },
    @{ name = "generated_gated"; mode = "real_generated"; decision = "concat"; reliability = "quality_confidence" },
    @{ name = "generated_residual"; mode = "real_generated"; decision = "reliability_residual"; reliability = "quality_confidence" }
)
foreach ($seed in $Seeds) {
    foreach ($arm in $arms) {
        $stem = "$($arm.name)_seed${seed}"
        $summary = Join-Path $output "${stem}.json"
        $checkpoints = Join-Path $output $stem
        if (Test-Path -LiteralPath $summary) {
            $existing = Get-Content -LiteralPath $summary -Raw | ConvertFrom-Json
            if ($existing.seed -ne $seed -or $existing.mode -ne $arm.mode -or
                $existing.decision_mode -ne $arm.decision -or
                $existing.generated_reliability -ne $arm.reliability -or
                $existing.max_real_comments -ne 16 -or
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
        & $Python $trainer --config $config --mode $arm.mode `
            --graph-structure observed_tree --message-passing typed_rgcn `
            --pooling-mode mean --decision-mode $arm.decision `
            --generated-reliability $arm.reliability `
            --max-real-comments 16 --max-generated-comments 5 `
            --seed $seed --allow-formal-training `
            --checkpoint-dir $checkpoints --summary-json $summary
        if ($LASTEXITCODE -ne 0) { throw "Run failed: $stem (exit $LASTEXITCODE)" }
        if (-not (Test-Path -LiteralPath $summary)) { throw "Missing summary: $summary" }
    }
}
Write-Host "Validation-only partial-observation ablations complete: $output"
