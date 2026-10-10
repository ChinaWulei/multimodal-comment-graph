param(
    [string]$Python = "python",
    [string]$Config = ".\comment_graph\config_fakeddit_structure.yaml"
)

# Fixed-seed replication of a pre-specified baseline vs. observed-real-tree
# comparison.  This script does not vary architecture after seeing test data;
# it estimates run-to-run variability of the same two configurations.
$ErrorActionPreference = "Stop"
$runRoot = ".\comment_graph\outputs\fakeddit_real_tree_replications"
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

foreach ($seed in @(13, 24)) {
    foreach ($arm in @(
        @{ name = "baseline"; mode = "baseline"; graph = "star"; message = "legacy_gcn" },
        @{ name = "real_tree"; mode = "real_only"; graph = "observed_tree"; message = "legacy_gcn" }
    )) {
        $tag = "$($arm.name)_seed$seed"
        & $Python .\comment_graph\train.py `
            --config $Config `
            --mode $arm.mode `
            --graph-structure $arm.graph `
            --message-passing $arm.message `
            --pooling-mode mean `
            --seed $seed `
            --checkpoint-dir (Join-Path $runRoot $tag) `
            --summary-json (Join-Path $runRoot "$tag.json") `
            --allow-formal-training `
            --run-test
        if ($LASTEXITCODE -ne 0) {
            throw "Replication failed: $tag"
        }
    }
}
