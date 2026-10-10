param(
    [string]$Python = "python",
    [string]$Config = ".\comment_graph\config_fakeddit_structure.yaml",
    [int]$Seed = 42,
    # Keeping a separate root is essential when a new generator changes the
    # paired-news coverage.  It prevents expanded runs from overwriting the
    # earlier, smaller-generator validation records.
    [string]$RunRoot = ""
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RunRoot)) {
    $runRoot = ".\comment_graph\outputs\fakeddit_structure_validation_seed$Seed"
} else {
    $runRoot = $RunRoot
}
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

function Invoke-Arm(
    [string]$Name,
    [string]$Mode,
    [string]$Structure,
    [string]$MessagePassing,
    [string]$Pooling = "mean"
) {
    $summary = Join-Path $runRoot "$Name.json"
    $checkpoints = Join-Path $runRoot $Name
    & $Python .\comment_graph\train.py `
        --config $Config `
        --mode $Mode `
        --graph-structure $Structure `
        --message-passing $MessagePassing `
        --pooling-mode $Pooling `
        --seed $Seed `
        --checkpoint-dir $checkpoints `
        --summary-json $summary `
        --allow-formal-training
    if ($LASTEXITCODE -ne 0) {
        throw "Validation arm failed: $Name"
    }
}

# All arms are restricted by config to the exact generated-graph coverage of
# each official split.  These runs deliberately omit --run-test.
Invoke-Arm "baseline" "baseline" "star" "legacy_gcn"
Invoke-Arm "real_star" "real_only" "star" "legacy_gcn"
Invoke-Arm "real_tree" "real_only" "observed_tree" "legacy_gcn"
Invoke-Arm "generated_star" "real_generated" "star" "legacy_gcn"
Invoke-Arm "generated_tree" "real_generated" "observed_tree" "legacy_gcn"
Invoke-Arm "generated_tree_typed" "real_generated" "observed_tree" "typed_rgcn"
# The following two arms isolate whether cross-modal quality gating helps
# generated nodes.  They keep split coverage, generated graphs, seed and
# topology fixed; only the aggregation mechanism changes.
Invoke-Arm "generated_tree_quality_gated" "real_generated" "observed_tree" "legacy_gcn" "quality_gated_attention"
Invoke-Arm "generated_tree_typed_quality_gated" "real_generated" "observed_tree" "typed_rgcn" "quality_gated_attention"
