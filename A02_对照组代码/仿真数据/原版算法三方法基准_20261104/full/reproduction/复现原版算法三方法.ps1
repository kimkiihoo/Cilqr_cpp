param(
    [string]$RunTag = (Get-Date -Format 'yyyyMMdd_HHmmss'),
    [string]$PythonExe = 'C:\Users\snh48\AppData\Local\Programs\Python\Python311\python.exe'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$rootDirectory = Get-Item -LiteralPath $PSScriptRoot
while (-not (Test-Path -LiteralPath (Join-Path $rootDirectory.FullName 'AGENTS.md'))) {
    if ($null -eq $rootDirectory.Parent) { throw '未找到包含 AGENTS.md 的项目根目录。' }
    $rootDirectory = $rootDirectory.Parent
}
$projectRoot = $rootDirectory.FullName
if (-not (Test-Path -LiteralPath $PythonExe)) {
    $PythonExe = (Get-Command python -ErrorAction Stop).Source
}
$dataOutput = Join-Path $projectRoot ('A02_对照组代码\仿真数据\原版算法三方法基准_' + $RunTag)
$figureOutput = Join-Path $projectRoot ('A02_对照组代码\图片\原版算法三方法基准_' + $RunTag)
if ((Test-Path -LiteralPath $dataOutput) -or (Test-Path -LiteralPath $figureOutput)) {
    throw '本次输出目录已存在，请换一个 RunTag，已有数据不会被覆盖。'
}

Write-Host '使用已核验的冻结场景和原交接算法；5/6/15 m/s 为基准重建设置。'
& $PythonExe -B (Join-Path $PSScriptRoot 'build_bridge.py')
if ($LASTEXITCODE -ne 0) { throw '原模型接口编译失败。' }
& $PythonExe -B (Join-Path $PSScriptRoot 'run_maps.py') --mode full --maps B201 B207 B301 --methods r al sqp --output $dataOutput
if ($LASTEXITCODE -ne 0) { throw '运行器报告错误，请保留输出记录并检查日志。' }
& $PythonExe -B (Join-Path $PSScriptRoot 'audit_original_model_mpc.py') $dataOutput
if ($LASTEXITCODE -ne 0) { throw '记录核验报告错误，请检查输出中的审计说明。' }
$recordAudit = Get-Content -LiteralPath (Join-Path $dataOutput 'independent_record_audit.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $recordAudit.record_audit_pass) { throw '记录一致性核验未通过，请保留数据并检查审计说明。' }
& $PythonExe -B (Join-Path $PSScriptRoot 'plot_original_model_comparison.py') --input $dataOutput --output $figureOutput
if ($LASTEXITCODE -ne 0) { throw '图表导出失败。' }
Write-Host ('数据：' + $dataOutput)
Write-Host ('图表：' + $figureOutput)
