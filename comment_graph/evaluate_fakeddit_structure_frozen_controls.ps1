param(
    [string]$Python = "python",
    [string]$Config = ".\comment_graph\config_fakeddit_structure.yaml",
    [int]$Seed = 42
)

# Evaluates the exact validation-selected checkpoints.  There is no optimizer
# or model selection in this script; outputs are final held-out measurements.
$ErrorActionPreference = "Stop"
$validationRoot = ".\comment_graph\outputs\fakeddit_structure_validation_seed$Seed"
$outputRoot = ".\comment_graph\outputs\fakeddit_structure_frozen_checkpoint_tests_seed$Seed"
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$arms = @(
    @{ name = "baseline"; checkpoint = "fakeddit_structure_mfcl_baseline_star_mean_balanced_seed42.pt" },
    @{ name = "real_star"; checkpoint = "fakeddit_structure_mfcl_real_only_star_mean_balanced_seed42.pt" },
    @{ name = "real_tree"; checkpoint = "fakeddit_structure_mfcl_real_only_observed_tree_mean_balanced_seed42.pt" },
    @{ name = "generated_star"; checkpoint = "fakeddit_structure_mfcl_real_generated_star_mean_balanced_seed42.pt" },
    @{ name = "generated_tree"; checkpoint = "fakeddit_structure_mfcl_real_generated_observed_tree_mean_balanced_seed42.pt" },
    @{ name = "generated_tree_typed"; checkpoint = "fakeddit_structure_mfcl_real_generated_observed_tree_mean_typed_rgcn_balanced_seed42.pt" }
)

foreach ($arm in $arms) {
    $checkpoint = Join-Path (Join-Path $validationRoot $arm.name) $arm.checkpoint
    if (-not (Test-Path $checkpoint -PathType Leaf)) {
        throw "Missing frozen checkpoint for $($arm.name): $checkpoint"
    }
    & $Python .\comment_graph\evaluate_frozen_checkpoint.py `
        --config $Config `
        --checkpoint $checkpoint `
        --output-json (Join-Path $outputRoot "$($arm.name).json")
    if ($LASTEXITCODE -ne 0) {
        throw "Frozen checkpoint evaluation failed: $($arm.name)"
    }
}
