param(
    [string]$Python = "python",
    [string]$Config = ".\comment_graph\config_fakeddit_structure.yaml",
    [int]$Seed = 42
)

# These are the core controls fixed before reading any test prediction.  The
# typed structure-aware arm has already been evaluated once in the frozen
# final run, so this script deliberately evaluates only the five remaining
# controls.  It must not be used to select a new model from test metrics.
$ErrorActionPreference = "Stop"
$runRoot = ".\comment_graph\outputs\fakeddit_structure_core_tests_seed$Seed"
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

function Invoke-TestArm(
    [string]$Name,
    [string]$Mode,
    [string]$Structure,
    [string]$MessagePassing
) {
    $armDir = Join-Path $runRoot $Name
    $summary = Join-Path $runRoot "$Name.json"
    & $Python .\comment_graph\train.py `
        --config $Config `
        --mode $Mode `
        --graph-structure $Structure `
        --message-passing $MessagePassing `
        --pooling-mode mean `
        --seed $Seed `
        --checkpoint-dir $armDir `
        --summary-json $summary `
        --allow-formal-training `
        --run-test
    if ($LASTEXITCODE -ne 0) {
        throw "Frozen test arm failed: $Name"
    }
}

Invoke-TestArm "baseline" "baseline" "star" "legacy_gcn"
Invoke-TestArm "real_star" "real_only" "star" "legacy_gcn"
Invoke-TestArm "real_tree" "real_only" "observed_tree" "legacy_gcn"
Invoke-TestArm "generated_star" "real_generated" "star" "legacy_gcn"
Invoke-TestArm "generated_tree" "real_generated" "observed_tree" "legacy_gcn"
