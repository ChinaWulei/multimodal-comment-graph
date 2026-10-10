param(
    [string]$Python = "python",
    [ValidateSet("train", "validate")][string[]]$Splits = @("train", "validate")
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$config = Join-Path $projectRoot "comment_generator\partial_observation\config\generator_structure_official_expanded.yaml"
$adapter = Join-Path $projectRoot "comment_generator\partial_observation\outputs\qwen2_5_vl_3b_structure_comment_official_expanded_lora"
$data = Join-Path $projectRoot "comment_generator\partial_observation\data\official_subset"
$generator = Join-Path $PSScriptRoot "generate_structured_comments.py"
$validator = Join-Path $PSScriptRoot "validate_generated_structured_graphs.py"
$merger = Join-Path $PSScriptRoot "merge_generated_structure_coverage.py"
foreach ($split in $Splits) {
    $contexts = Join-Path $data "structure_${split}_missing_contexts.jsonl"
    $missing = Join-Path $data "generated_${split}_missing_contexts_observed_boundary.jsonl"
    $original = Join-Path $data "generated_${split}_structured_graphs_expanded_generateddepth.jsonl"
    $merged = Join-Path $data "generated_${split}_all_valid_images_observed_boundary.jsonl"
    $stats = Join-Path $data "generated_${split}_all_valid_images_observed_boundary_stats.json"
    foreach ($path in @($contexts, $original)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing split source: $path" }
    }
    & $Python $generator --config $config --adapter-dir $adapter `
        --input-jsonl $contexts --output-jsonl $missing `
        --official-split $split --one-per-news --min-observed-nodes 1 `
        --limit 0 --nodes-per-graph 5 --depth-reference observed_boundary `
        --resume --local-files-only
    if ($LASTEXITCODE -ne 0) { throw "Generation failed: $split" }
    & $Python $validator --input-jsonl $missing --expected-split $split `
        --expected-context-jsonl $contexts --min-observed-nodes 1
    if ($LASTEXITCODE -ne 0) { throw "Missing-graph validation failed: $split" }
    & $Python $merger --manifest-csv (Join-Path $data 'manifest.csv') `
        --existing-jsonl $original --missing-jsonl $missing `
        --output-jsonl $merged --stats-json $stats --split $split `
        --expected-depth-reference observed_boundary
    if ($LASTEXITCODE -ne 0) { throw "Split coverage merge failed: $split" }
    & $Python $validator --input-jsonl $merged --expected-split $split
    if ($LASTEXITCODE -ne 0) { throw "Merged-graph validation failed: $split" }
}
Write-Host "Requested split graphs generated, validated, and merged. No classifier training was started."
