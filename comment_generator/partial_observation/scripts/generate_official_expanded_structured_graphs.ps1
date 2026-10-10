param(
    [string]$Python = "python",
    [string]$Config = ".\comment_generator\partial_observation\config\generator_structure_official_expanded.yaml",
    [string]$AdapterDir = ".\comment_generator\partial_observation\outputs\qwen2_5_vl_3b_structure_comment_official_expanded_lora",
    [int]$NodesPerGraph = 5,
    [ValidateSet("news_root", "observed_boundary")][string]$DepthReference = "news_root",
    [ValidatePattern('^[a-z0-9_]+$')][string]$OutputTag = "expanded_bestval_depthbounded"
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path $AdapterDir -PathType Container)) {
    throw "Missing expanded validation-selected adapter directory: $AdapterDir"
}
# During training the output directory appears before the final adapter export.
# Requiring the root PEFT files prevents generation from accidentally loading
# an incomplete run (for example only checkpoint-50 is present).
if (-not (Test-Path (Join-Path $AdapterDir "adapter_config.json") -PathType Leaf) -or
    -not (Test-Path (Join-Path $AdapterDir "adapter_model.safetensors") -PathType Leaf)) {
    throw "Expanded adapter is not finalized yet. Expected adapter_config.json and adapter_model.safetensors in: $AdapterDir"
}

$dataDir = ".\comment_generator\partial_observation\data\official_subset"
foreach ($split in @("train", "validate", "test")) {
    $input = Join-Path $dataDir "structure_${split}_expanded.jsonl"
    $output = Join-Path $dataDir "generated_${split}_structured_graphs_${OutputTag}.jsonl"
    & $Python .\comment_generator\partial_observation\scripts\generate_structured_comments.py `
        --config $Config `
        --adapter-dir $AdapterDir `
        --input-jsonl $input `
        --official-split $split `
        --one-per-news `
        --min-observed-nodes 1 `
        --limit 0 `
        --resume `
        --nodes-per-graph $NodesPerGraph `
        --depth-reference $DepthReference `
        --output-jsonl $output `
        --local-files-only
    if ($LASTEXITCODE -ne 0) {
        throw "Expanded structured generation failed for $split"
    }
    & $Python .\comment_generator\partial_observation\scripts\validate_generated_structured_graphs.py `
        --input-jsonl $output `
        --expected-split $split `
        --expected-context-jsonl $input `
        --min-observed-nodes 1
    if ($LASTEXITCODE -ne 0) {
        throw "Expanded graph validation failed for $split"
    }
}
