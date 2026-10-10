param(
    [string]$Python = 'C:\Users\Administrator\.conda\envs\multimodal-fakenews\python.exe'
)
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$runner = Join-Path $PSScriptRoot 'run_grounded_structure.py'
$output = Join-Path $PSScriptRoot 'outputs\grounded_structure_20261010'
$smoke = Join-Path $output 'smoke_report.json'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing Python: $Python" }
if (-not (Test-Path -LiteralPath $smoke -PathType Leaf)) { throw 'Run smoke first.' }
$checked = Get-Content -LiteralPath $smoke -Raw | ConvertFrom-Json
if (-not $checked.passed) { throw 'Smoke gate not passed.' }
$tag = [Guid]::NewGuid().ToString('N').Substring(0, 12)
$stdout = Join-Path $output "pipeline_$tag.log"
$stderr = Join-Path $output "pipeline_${tag}_stderr.log"
$process = Start-Process -FilePath $Python -WorkingDirectory $project -WindowStyle Hidden -PassThru `
    -ArgumentList @('-u', "`"$runner`"", '--stage', 'full', '--output-dir', "`"$output`"") `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr
[PSCustomObject]@{ ProcessId = $process.Id; OutputLog = $stdout; ErrorLog = $stderr; Status = (Join-Path $output 'status.json') } | ConvertTo-Json
