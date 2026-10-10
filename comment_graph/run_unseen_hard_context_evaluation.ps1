param(
    [string]$Python = "python",
    [string]$OutputDirectory = ".\comment_graph\outputs\fakeddit_unseen_test_hard_context_20261008"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$config = Join-Path $PSScriptRoot "config_fakeddit_structure_expanded_generateddepth.yaml"
$evaluator = Join-Path $PSScriptRoot "evaluate_frozen_checkpoint.py"
$validator = Join-Path $root "comment_generator\partial_observation\scripts\validate_generated_structured_graphs.py"
$data = Join-Path $root "comment_generator\partial_observation\data\official_subset"
$contexts = Join-Path $data "structure_test_unseen_contexts.jsonl"
$generated = Join-Path $data "generated_test_unseen_hard_contexts.jsonl"
$summaries = Join-Path $PSScriptRoot "outputs\fakeddit_partial_real16_validation_20261008"
$output = [System.IO.Path]::GetFullPath((Join-Path $root $OutputDirectory))

if (-not (Test-Path -LiteralPath $contexts -PathType Leaf) -or
    -not (Test-Path -LiteralPath $generated -PathType Leaf)) {
    throw "Missing frozen test contexts or generated graphs"
}
& $Python $validator --input-jsonl $generated --expected-split test `
    --expected-context-jsonl $contexts --min-observed-nodes 1
if ($LASTEXITCODE -ne 0) { throw "Generated graph validation failed" }
$contextRows = @(Get-Content -LiteralPath $contexts | ForEach-Object { $_ | ConvertFrom-Json })
$graphRows = @(Get-Content -LiteralPath $generated | ForEach-Object { $_ | ConvertFrom-Json })
if ($contextRows.Count -ne 28 -or $graphRows.Count -ne 28) {
    throw "Predeclared 28-news coverage failed: contexts=$($contextRows.Count) graphs=$($graphRows.Count)"
}
$bad = @($graphRows | Where-Object { @($_.generated_nodes).Count -gt 5 })
if ($bad.Count -gt 0) {
    throw "More than five generated nodes for: $(($bad.news_id) -join ',')"
}
$nodeCounts = @($graphRows | ForEach-Object { @($_.generated_nodes).Count } | Group-Object | Sort-Object Name)
Write-Host "Realized generated-node counts (including failures): $(($nodeCounts | ForEach-Object { "$($_.Name):$($_.Count)" }) -join ', ')"

New-Item -ItemType Directory -Force -Path $output | Out-Null
$arms = @(
    @{ name = "real_only"; mode = "real_only"; decision = "concat"; reliability = "none" },
    @{ name = "generated_plain"; mode = "real_generated"; decision = "concat"; reliability = "none" },
    @{ name = "generated_gated"; mode = "real_generated"; decision = "concat"; reliability = "quality_confidence" },
    @{ name = "generated_residual"; mode = "real_generated"; decision = "reliability_residual"; reliability = "quality_confidence" }
)
foreach ($seed in @(42, 13, 24)) {
    foreach ($arm in $arms) {
        $name = "$($arm.name)_seed${seed}"
        $summaryPath = Join-Path $summaries "${name}.json"
        if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
            throw "Missing validation-selected run: $summaryPath"
        }
        $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
        if ($summary.seed -ne $seed -or $summary.mode -ne $arm.mode -or
            $summary.decision_mode -ne $arm.decision -or
            $summary.generated_reliability -ne $arm.reliability -or
            $summary.max_real_comments -ne 16 -or $summary.max_generated_comments -ne 5 -or
            -not (Test-Path -LiteralPath $summary.checkpoint -PathType Leaf)) {
            throw "Validation/checkpoint mismatch: $summaryPath"
        }
        $resultPath = Join-Path $output "${name}.json"
        if (Test-Path -LiteralPath $resultPath -PathType Leaf) {
            $existing = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
            $currentGraphHash = (Get-FileHash -LiteralPath $generated -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($existing.checkpoint -ne $summary.checkpoint -or
                $existing.generated_test_sha256 -ne $currentGraphHash -or
                $existing.test_size -ne 28) {
                throw "Existing result does not match frozen inputs: $resultPath"
            }
            Write-Host "Verified completed: $name"
            continue
        }
        Write-Host "Evaluating frozen checkpoint once: $name"
        & $Python $evaluator --config $config --checkpoint $summary.checkpoint `
            --generated-test-jsonl $generated --expected-context-jsonl $contexts `
            --output-json $resultPath
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $name" }
    }
}
Write-Host "Hard-context exploratory evaluation complete: $output"
