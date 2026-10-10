param(
    [string]$PythonExe = "",
    [string]$RunName = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot

try {
    if ([string]::IsNullOrWhiteSpace($PythonExe)) {
        $PythonExe = (Get-Command python -ErrorAction Stop).Source
    }
    if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
        throw "Python executable not found: $PythonExe"
    }

    if ([string]::IsNullOrWhiteSpace($RunName)) {
        $RunName = Get-Date -Format "yyyyMMdd_HHmmss"
    }

    $RunDir = Join-Path $ProjectRoot "comment_graph\outputs\ablation_runs\$RunName"
    $SummaryDir = Join-Path $RunDir "summaries"
    $LogDir = Join-Path $RunDir "logs"
    New-Item -ItemType Directory -Force -Path $SummaryDir, $LogDir | Out-Null

    $Config = Join-Path $ProjectRoot "comment_graph\config.yaml"
    $TrainScript = Join-Path $ProjectRoot "comment_graph\train.py"
    $GeneratorScript = Join-Path $ProjectRoot "comment_generator\partial_observation\scripts\generate_comments.py"
    $GeneratorConfig = Join-Path $ProjectRoot "comment_generator\checked\config\config.yaml"
    $AdapterDir = Join-Path $ProjectRoot "comment_generator\checked\outputs\qwen2_5_vl_3b_checked_comment_lora"
    $GeneratedTest = Join-Path $ProjectRoot "comment_generator\checked\outputs\generated_test.jsonl"
    $Manifest = Join-Path $ProjectRoot "comment_generator\checked\data\manifest.csv"

    foreach ($RequiredPath in @($Config, $TrainScript, $GeneratorScript, $GeneratorConfig, $AdapterDir, $Manifest)) {
        if (-not (Test-Path -LiteralPath $RequiredPath)) {
            throw "Required path not found: $RequiredPath"
        }
    }

    $ExpectedTestIds = @(
        Import-Csv -LiteralPath $Manifest |
        Where-Object { $_.split -eq "test" } |
        ForEach-Object { [string]$_.news_id } |
        Sort-Object -Unique
    )
    if ($ExpectedTestIds.Count -eq 0) {
        throw "No test records found in manifest: $Manifest"
    }

    if (-not (Test-Path -LiteralPath $GeneratedTest -PathType Leaf)) {
        Write-Host "generated_test.jsonl is missing; generating label-free test comments first."
        $GenerationLog = Join-Path $LogDir "00_generate_test.log"
        $PreviousErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $PythonExe $GeneratorScript `
            --config $GeneratorConfig `
            --adapter-dir $AdapterDir `
            --output-jsonl $GeneratedTest `
            --split test `
            --limit 0 `
            --unique-news `
            --comments-per-news 5 `
            --local-files-only 2>&1 | Tee-Object -FilePath $GenerationLog
        $GenerationExitCode = $LASTEXITCODE
        $ErrorActionPreference = $PreviousErrorAction
        if ($GenerationExitCode -ne 0) {
            throw "Test comment generation failed with exit code $GenerationExitCode. See $GenerationLog"
        }
    }

    $GeneratedRecords = @(Get-Content -LiteralPath $GeneratedTest -Encoding UTF8 | ForEach-Object {
        if (-not [string]::IsNullOrWhiteSpace($_)) { $_ | ConvertFrom-Json }
    })
    $GeneratedIds = @($GeneratedRecords | ForEach-Object { [string]$_.news_id } | Sort-Object -Unique)
    $WrongSplit = @($GeneratedRecords | Where-Object { $_.split -ne "test" })
    $MissingIds = @($ExpectedTestIds | Where-Object { $_ -notin $GeneratedIds })
    $EmptyComments = @(
        $GeneratedRecords |
        Where-Object { @($_.generated_comments | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) }).Count -eq 0 }
    )
    if ($WrongSplit.Count -gt 0 -or $MissingIds.Count -gt 0 -or $EmptyComments.Count -gt 0) {
        throw (
            "Generated test validation failed: records={0}, unique_ids={1}, wrong_split={2}, " +
            "missing_ids={3}, empty_comments={4}" -f
            $GeneratedRecords.Count, $GeneratedIds.Count, $WrongSplit.Count,
            $MissingIds.Count, $EmptyComments.Count
        )
    }
    Write-Host "Validated generated test comments: $($GeneratedIds.Count)/$($ExpectedTestIds.Count) news."

    $Experiments = @(
        @{ Name = "01_star_mean"; Structure = "star"; Pooling = "mean" },
        @{ Name = "02_star_dual_attention"; Structure = "star"; Pooling = "dual_modal_attention" },
        @{ Name = "03_dynamic_mean"; Structure = "dynamic_cross_modal"; Pooling = "mean" },
        @{ Name = "04_dynamic_dual_attention"; Structure = "dynamic_cross_modal"; Pooling = "dual_modal_attention" }
    )

    foreach ($Experiment in $Experiments) {
        $Name = $Experiment.Name
        $SummaryPath = Join-Path $SummaryDir "$Name.json"
        $LogPath = Join-Path $LogDir "$Name.log"
        Write-Host ""
        Write-Host "===== Starting $Name ====="
        $PreviousErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $PythonExe $TrainScript `
            --config $Config `
            --mode real_generated `
            --graph-structure $Experiment.Structure `
            --pooling-mode $Experiment.Pooling `
            --allow-formal-training `
            --run-test `
            --summary-json $SummaryPath 2>&1 | Tee-Object -FilePath $LogPath
        $TrainingExitCode = $LASTEXITCODE
        $ErrorActionPreference = $PreviousErrorAction
        if ($TrainingExitCode -ne 0) {
            throw "Experiment $Name failed with exit code $TrainingExitCode. See $LogPath"
        }
        if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
            throw "Experiment $Name did not create summary: $SummaryPath"
        }
    }

    $Results = foreach ($Experiment in $Experiments) {
        $SummaryPath = Join-Path $SummaryDir "$($Experiment.Name).json"
        $Value = Get-Content -LiteralPath $SummaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
        [PSCustomObject]@{
            experiment = $Experiment.Name
            graph_structure = $Value.graph_structure
            pooling_mode = $Value.pooling_mode
            best_epoch = $Value.best_epoch
            validation_loss = $Value.validation.loss
            validation_accuracy = $Value.validation.accuracy
            validation_macro_f1 = $Value.validation.macro_f1
            validation_weighted_f1 = $Value.validation.weighted_f1
            test_loss = $Value.test.loss
            test_accuracy = $Value.test.accuracy
            test_macro_f1 = $Value.test.macro_f1
            test_weighted_f1 = $Value.test.weighted_f1
            checkpoint = $Value.checkpoint
        }
    }

    $AggregateJson = Join-Path $RunDir "all_results.json"
    $AggregateCsv = Join-Path $RunDir "all_results.csv"
    $Results | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $AggregateJson -Encoding UTF8
    $Results | Export-Csv -LiteralPath $AggregateCsv -NoTypeInformation -Encoding UTF8

    Write-Host ""
    Write-Host "===== All four experiments completed ====="
    $Results | Format-Table experiment, validation_macro_f1, test_accuracy, test_macro_f1, test_weighted_f1 -AutoSize
    Write-Host "JSON: $AggregateJson"
    Write-Host "CSV:  $AggregateCsv"
}
finally {
    Pop-Location
}
