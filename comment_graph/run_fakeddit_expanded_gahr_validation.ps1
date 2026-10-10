param(
    [string]$Python = "python",
    [string]$Config = ".\comment_graph\config_fakeddit_structure_expanded.yaml",
    [int]$Seed = 42,
    [string]$RunRoot = "",
    [switch]$ResumeCompleted
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RunRoot)) {
    $RunRoot = ".\comment_graph\outputs\fakeddit_expanded_gahr_validation_seed$Seed"
}
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

function Invoke-Arm(
    [string]$Name,
    [string]$Mode,
    [string]$Decision
) {
    $summary = Join-Path $RunRoot "$Name.json"
    $checkpoints = Join-Path $RunRoot $Name
    if ($ResumeCompleted -and (Test-Path -LiteralPath $summary -PathType Leaf)) {
        & $Python -c "import json,sys; from comment_graph.finalize_fakeddit_structure import validate_summary, validate_checkpoint; r=json.load(open(sys.argv[1],encoding='utf-8')); validate_summary(r,sys.argv[2],int(sys.argv[3])); validate_checkpoint(r)" $summary $Name $Seed
        if ($LASTEXITCODE -ne 0) { throw "Completed arm failed verification: $Name" }
        Write-Output "Verified completed arm, preserving weights and summary: $Name"
        return
    }
    $directoryExists = Test-Path -LiteralPath $checkpoints
    $emptyRecoveryDirectory = $ResumeCompleted -and $directoryExists -and (@(Get-ChildItem -LiteralPath $checkpoints -Force).Count -eq 0)
    if ((Test-Path -LiteralPath $summary) -or ($directoryExists -and -not $emptyRecoveryDirectory)) {
        throw "Existing experiment artifacts for $Name. Use a new -RunRoot to preserve previous results."
    }
    & $Python -u .\comment_graph\train.py `
        --config $Config `
        --mode $Mode `
        --graph-structure observed_tree `
        --message-passing typed_rgcn `
        --pooling-mode mean `
        --decision-mode $Decision `
        --seed $Seed `
        --checkpoint-dir $checkpoints `
        --summary-json $summary `
        --allow-formal-training
    if ($LASTEXITCODE -ne 0) {
        throw "Validation arm failed: $Name"
    }
}

# All arms use the generated-graph coverage restriction declared in the
# expanded config.  They deliberately omit --run-test: choose on validation,
# then evaluate each predeclared arm's validation-selected checkpoint on test.
Invoke-Arm "real_tree_typed_concat" "real_only" "concat"
Invoke-Arm "real_tree_typed_global_local" "real_only" "global_local_correction"
Invoke-Arm "generated_tree_typed_concat" "real_generated" "concat"
Invoke-Arm "generated_tree_typed_global_local" "real_generated" "global_local_correction"
