param(
    [int]$GenerationParentPid,
    [string]$Python = "python",
    [string]$GeneratorDataDir = ".\comment_generator\partial_observation\data\official_subset",
    [string]$ClassifierConfig = ".\comment_graph\config_fakeddit_structure.yaml",
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

if ($GenerationParentPid -gt 0 -and (Get-Process -Id $GenerationParentPid -ErrorAction SilentlyContinue)) {
    Wait-Process -Id $GenerationParentPid
}

foreach ($split in @("train", "validate", "test")) {
    $graph = Join-Path $GeneratorDataDir "generated_${split}_structured_graphs_bestval_depthbounded.jsonl"
    $contexts = Join-Path $GeneratorDataDir "structure_${split}.jsonl"
    & $Python .\comment_generator\partial_observation\scripts\validate_generated_structured_graphs.py `
        --input-jsonl $graph `
        --expected-split $split `
        --expected-context-jsonl $contexts `
        --min-observed-nodes 1
    if ($LASTEXITCODE -ne 0) {
        throw "Generated-graph validation failed for $split; validation experiments will not start."
    }
}

# This runner contains validation-only arms and deliberately omits --run-test.
& .\comment_graph\run_fakeddit_structure_validation.ps1 `
    -Python $Python `
    -Config $ClassifierConfig `
    -Seed $Seed
if ($LASTEXITCODE -ne 0) {
    throw "Fakeddit validation ablations failed"
}
