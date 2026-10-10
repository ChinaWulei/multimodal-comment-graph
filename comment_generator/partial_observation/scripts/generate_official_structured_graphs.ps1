param(
    [string]$Python = "python",
    [string]$Config = ".\comment_generator\partial_observation\config\generator_structure_official.yaml",
    # This is the exported adapter selected by the official validation loss.
    # Do not silently fall back to the earlier audit run.
    [string]$AdapterDir = ".\comment_generator\partial_observation\outputs\qwen2_5_vl_3b_structure_comment_official_bestval_lora",
    [int]$NodesPerGraph = 5
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $AdapterDir -PathType Container)) {
    throw "Missing trained adapter: $AdapterDir"
}

$dataDir = ".\comment_generator\partial_observation\data\official_subset"
foreach ($split in @("train", "validate", "test")) {
    $input = Join-Path $dataDir "structure_$split.jsonl"
    # Keep inference from the validation-selected adapter separate from the
    # earlier audit output, whose adapter was not validation-selected.
    $output = Join-Path $dataDir "generated_${split}_structured_graphs_bestval_depthbounded.jsonl"
    if (-not (Test-Path $input -PathType Leaf)) {
        throw "Missing structured input for ${split}: $input"
    }
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
        --output-jsonl $output `
        --local-files-only
    if ($LASTEXITCODE -ne 0) {
        throw "Structured generation failed for $split"
    }
}
