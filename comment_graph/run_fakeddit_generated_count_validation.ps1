param(
    [string]$Python = "python",
    [string]$Config = ".\comment_graph\config_fakeddit_structure.yaml",
    [int]$Seed = 42
)

# This is a validation-only noise ablation.  The exact same persisted
# generated graphs and exact same news IDs are used in every arm; only the
# prefix length of generated nodes is changed.  It intentionally omits
# --run-test so the held-out test split cannot choose K.
$ErrorActionPreference = "Stop"
$runRoot = ".\comment_graph\outputs\fakeddit_generated_count_validation_seed$Seed"
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

foreach ($count in @(1, 2, 3, 4, 5)) {
    $name = "typed_tree_generated_$count"
    & $Python .\comment_graph\train.py `
        --config $Config `
        --mode real_generated `
        --graph-structure observed_tree `
        --message-passing typed_rgcn `
        --pooling-mode mean `
        --max-generated-comments $count `
        --seed $Seed `
        --checkpoint-dir (Join-Path $runRoot $name) `
        --summary-json (Join-Path $runRoot "$name.json") `
        --allow-formal-training
    if ($LASTEXITCODE -ne 0) {
        throw "Generated-count validation arm failed: K=$count"
    }
}
