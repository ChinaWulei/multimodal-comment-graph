param(
    [string]$Python = "C:\Users\Administrator\.conda\envs\multimodal-fakenews\python.exe",
    [int[]]$Seeds = @(42, 13, 24),
    [string]$GatedDirectory = ".\comment_graph\outputs\human_quality_gate",
    [string]$OutputDirectory = ".\comment_graph\outputs\human_quality_gate_validation",
    [switch]$RunTraining
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$data = Join-Path $root "comment_generator\partial_observation\data\official_subset"
$originalTrain = Join-Path $data "generated_train_all_valid_images_observed_boundary.jsonl"
$originalValidate = Join-Path $data "generated_validate_all_valid_images_observed_boundary.jsonl"
$gated = [System.IO.Path]::GetFullPath((Join-Path $root $GatedDirectory))
$output = [System.IO.Path]::GetFullPath((Join-Path $root $OutputDirectory))
$gateModel = Join-Path $gated "gate.joblib"
$gateFit = Join-Path $gated "fit_report.json"
$gatedTrain = Join-Path $gated "gated_train.jsonl"
$gatedValidate = Join-Path $gated "gated_validate.jsonl"
$gatedTrainReport = Join-Path $gated "gated_train_report.json"
$gatedValidateReport = Join-Path $gated "gated_validate_report.json"
$config = Join-Path $PSScriptRoot "config_fakeddit_structure_expanded_generateddepth.yaml"
$validator = Join-Path $root "comment_generator\partial_observation\scripts\validate_generated_structured_graphs.py"
$auditor = Join-Path $PSScriptRoot "audit_pruned_graph_pair.py"
$trainer = Join-Path $PSScriptRoot "train.py"

foreach ($path in @($Python, $originalTrain, $originalValidate, $gateModel, $gateFit,
                    $gatedTrain, $gatedValidate, $gatedTrainReport, $gatedValidateReport,
                    $config, $validator, $auditor, $trainer)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing prerequisite: $path"
    }
}
if ($Seeds.Count -lt 1 -or @($Seeds | Sort-Object -Unique).Count -ne $Seeds.Count) {
    throw "Seeds must be a nonempty unique list"
}
$fit = Get-Content -LiteralPath $gateFit -Raw | ConvertFrom-Json
if ($fit.split -ne 'official_train' -or $fit.quality_labels -ne 'human only' -or
    -not $fit.news_disjoint_cross_validation -or
    $fit.content.oof_roc_auc -le 0.60 -or $fit.parent.oof_roc_auc -le 0.60) {
    throw "Gate fit report does not certify the required human-label train-only checks"
}
$gateHash = (Get-FileHash -LiteralPath $gateModel -Algorithm SHA256).Hash.ToLowerInvariant()
$sourceTrainHash = (Get-FileHash -LiteralPath $originalTrain -Algorithm SHA256).Hash.ToLowerInvariant()
$sourceValidateHash = (Get-FileHash -LiteralPath $originalValidate -Algorithm SHA256).Hash.ToLowerInvariant()
$gatedTrainHash = (Get-FileHash -LiteralPath $gatedTrain -Algorithm SHA256).Hash.ToLowerInvariant()
$gatedValidateHash = (Get-FileHash -LiteralPath $gatedValidate -Algorithm SHA256).Hash.ToLowerInvariant()
$trainReport = Get-Content -LiteralPath $gatedTrainReport -Raw | ConvertFrom-Json
$validateReport = Get-Content -LiteralPath $gatedValidateReport -Raw | ConvertFrom-Json
if ($trainReport.split -ne 'train' -or $validateReport.split -ne 'validate' -or
    $trainReport.gate_model_sha256 -ne $gateHash -or $validateReport.gate_model_sha256 -ne $gateHash -or
    $trainReport.source_graph_sha256 -ne $sourceTrainHash -or
    $validateReport.source_graph_sha256 -ne $sourceValidateHash -or
    $trainReport.news -ne 292 -or $validateReport.news -ne 77 -or
    ($trainReport.generated_after + $validateReport.generated_after) -lt 1) {
    throw "Gated graph report provenance or coverage mismatch"
}

New-Item -ItemType Directory -Force -Path $output | Out-Null
foreach ($split in @('train', 'validate')) {
    if ($split -eq 'train') {
        $originalFile = $originalTrain
        $gatedFile = $gatedTrain
        $gatedHash = $gatedTrainHash
    } else {
        $originalFile = $originalValidate
        $gatedFile = $gatedValidate
        $gatedHash = $gatedValidateHash
    }
    & $Python $validator --input-jsonl $gatedFile --expected-split $split
    if ($LASTEXITCODE -ne 0) { throw "Structured graph validation failed: $split" }
    $auditPath = Join-Path $output "audit_${split}.json"
    if (-not (Test-Path -LiteralPath $auditPath -PathType Leaf)) {
        & $Python $auditor --original-jsonl $originalFile --gated-jsonl $gatedFile `
            --split $split --output-json $auditPath
        if ($LASTEXITCODE -ne 0) { throw "Paired graph audit failed: $split" }
    }
    $audit = Get-Content -LiteralPath $auditPath -Raw | ConvertFrom-Json
    if ($audit.gated_sha256 -ne $gatedHash -or
        -not $audit.news_real_nodes_edges_and_survivor_parents_identical) {
        throw "Existing paired graph audit does not match current gated graph: $split"
    }
}
Write-Host "Human gate and paired train/validation graphs verified."
if (-not $RunTraining) {
    Write-Host "Preflight only. Add -RunTraining to start four-arm validation training."
    return
}

$arms = @(
    @{ name = "real_only"; mode = "real_only"; structure = "observed_tree"; train = $originalTrain; validate = $originalValidate },
    @{ name = "generated_root"; mode = "real_generated"; structure = "generated_root"; train = $originalTrain; validate = $originalValidate },
    @{ name = "generated_parent"; mode = "real_generated"; structure = "observed_tree"; train = $originalTrain; validate = $originalValidate },
    @{ name = "human_gated_parent"; mode = "real_generated"; structure = "observed_tree"; train = $gatedTrain; validate = $gatedValidate }
)
foreach ($seed in $Seeds) {
    foreach ($arm in $arms) {
        $stem = "$($arm.name)_seed${seed}"
        $summary = Join-Path $output "${stem}.json"
        $checkpointDir = Join-Path $output $stem
        $trainHash = (Get-FileHash -LiteralPath $arm.train -Algorithm SHA256).Hash.ToLowerInvariant()
        $validateHash = (Get-FileHash -LiteralPath $arm.validate -Algorithm SHA256).Hash.ToLowerInvariant()
        if (Test-Path -LiteralPath $summary -PathType Leaf) {
            $existing = Get-Content -LiteralPath $summary -Raw | ConvertFrom-Json
            if ($existing.seed -ne $seed -or $existing.mode -ne $arm.mode -or
                $existing.graph_structure -ne $arm.structure -or
                $existing.message_passing -ne 'typed_rgcn' -or
                $existing.pooling_mode -ne 'mean' -or
                $existing.decision_mode -ne 'concat' -or
                $existing.generated_reliability -ne 'none' -or
                $existing.train_size -ne 292 -or $existing.validate_size -ne 77 -or
                $existing.max_real_comments -ne 16 -or $existing.max_generated_comments -ne 5 -or
                $existing.generated_sources.train.sha256 -ne $trainHash -or
                $existing.generated_sources.validate.sha256 -ne $validateHash -or
                -not (Test-Path -LiteralPath $existing.checkpoint -PathType Leaf)) {
                throw "Existing result mismatch: $summary"
            }
            Write-Host "Verified completed: $stem"
            continue
        }
        if (Test-Path -LiteralPath $checkpointDir) {
            if (@(Get-ChildItem -LiteralPath $checkpointDir -Force).Count -gt 0) {
                throw "Incomplete nonempty run directory: $checkpointDir"
            }
        }
        Write-Host "Training validation-only: $stem"
        & $Python $trainer --config $config --mode $arm.mode `
            --graph-structure $arm.structure --message-passing typed_rgcn `
            --pooling-mode mean --decision-mode concat `
            --generated-reliability none `
            --generated-train-jsonl $arm.train --generated-validate-jsonl $arm.validate `
            --max-real-comments 16 --max-generated-comments 5 `
            --seed $seed --allow-formal-training `
            --checkpoint-dir $checkpointDir --summary-json $summary
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $stem (exit $LASTEXITCODE)" }
    }
}
Write-Host "Four-arm validation-only comparison complete: $output"
