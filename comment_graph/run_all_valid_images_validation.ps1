param(
    [string]$Python = "python",
    [int[]]$Seeds = @(42, 13, 24),
    [string]$OutputDirectory = ".\comment_graph\outputs\fakeddit_all_valid_images_validation_20261008"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$data = Join-Path $root "comment_generator\partial_observation\data\official_subset"
$trainGraphs = Join-Path $data "generated_train_all_valid_images_observed_boundary.jsonl"
$validationGraphs = Join-Path $data "generated_validate_all_valid_images_observed_boundary.jsonl"
$config = Join-Path $PSScriptRoot "config_fakeddit_structure_expanded_generateddepth.yaml"
$trainer = Join-Path $PSScriptRoot "train.py"
$output = [System.IO.Path]::GetFullPath((Join-Path $root $OutputDirectory))
foreach ($path in @($trainGraphs, $validationGraphs)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing verified merged graph file: $path" }
}
$trainStats = Get-Content (Join-Path $data "generated_train_all_valid_images_observed_boundary_stats.json") -Raw | ConvertFrom-Json
$valStats = Get-Content (Join-Path $data "generated_validate_all_valid_images_observed_boundary_stats.json") -Raw | ConvertFrom-Json
if ($trainStats.merged_graphs -ne 292 -or $valStats.merged_graphs -ne 77 -or
    $trainStats.depth_reference -ne 'observed_boundary' -or
    $valStats.depth_reference -ne 'observed_boundary') {
    throw "Expected 292/77 split-preserving graph coverage"
}
New-Item -ItemType Directory -Force -Path $output | Out-Null
$arms = @(
    @{ name = "real_only"; mode = "real_only"; structure = "observed_tree"; reliability = "none" },
    @{ name = "generated_root"; mode = "real_generated"; structure = "generated_root"; reliability = "none" },
    @{ name = "generated_plain"; mode = "real_generated"; structure = "observed_tree"; reliability = "none" },
    @{ name = "generated_gated"; mode = "real_generated"; structure = "observed_tree"; reliability = "quality_confidence" }
)
foreach ($seed in $Seeds) {
    foreach ($arm in $arms) {
        $stem = "$($arm.name)_seed${seed}"
        $summary = Join-Path $output "${stem}.json"
        $checkpointDir = Join-Path $output $stem
        if (Test-Path -LiteralPath $summary -PathType Leaf) {
            $existing = Get-Content -LiteralPath $summary -Raw | ConvertFrom-Json
            if ($existing.seed -ne $seed -or $existing.mode -ne $arm.mode -or
                $existing.graph_structure -ne $arm.structure -or
                $existing.generated_reliability -ne $arm.reliability -or
                $existing.train_size -ne 292 -or $existing.validate_size -ne 77 -or
                $existing.max_real_comments -ne 16 -or
                -not (Test-Path -LiteralPath $existing.checkpoint -PathType Leaf)) {
                throw "Existing result mismatch: $summary"
            }
            $trainHash = (Get-FileHash -LiteralPath $trainGraphs -Algorithm SHA256).Hash.ToLowerInvariant()
            $valHash = (Get-FileHash -LiteralPath $validationGraphs -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($existing.generated_sources.train.sha256 -ne $trainHash -or
                $existing.generated_sources.validate.sha256 -ne $valHash) {
                throw "Merged graph sources changed since completed result: $summary"
            }
            Write-Host "Verified completed: $stem"
            continue
        }
        if (Test-Path -LiteralPath $checkpointDir) {
            $files = @(Get-ChildItem -LiteralPath $checkpointDir -Force)
            if ($files.Count -gt 0) { throw "Incomplete nonempty run directory: $checkpointDir" }
        }
        Write-Host "Training validation-only: $stem"
        & $Python $trainer --config $config --mode $arm.mode `
            --graph-structure $arm.structure --message-passing typed_rgcn `
            --pooling-mode mean --decision-mode concat `
            --generated-reliability $arm.reliability `
            --generated-train-jsonl $trainGraphs --generated-validate-jsonl $validationGraphs `
            --max-real-comments 16 --max-generated-comments 5 `
            --seed $seed --allow-formal-training `
            --checkpoint-dir $checkpointDir --summary-json $summary
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $stem (exit $LASTEXITCODE)" }
    }
}
Write-Host "All-valid-images validation-only comparison complete: $output"
