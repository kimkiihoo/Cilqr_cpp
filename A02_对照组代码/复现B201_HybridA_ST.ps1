param([string]$RunTag = (Get-Date -Format 'yyyyMMdd_HHmmss'))
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$reproduction = Join-Path $PSScriptRoot '仿真数据/原版算法三方法基准_20261104/full/reproduction'
$baseline = Split-Path -Parent $reproduction
$output = Join-Path $PSScriptRoot ('仿真数据/HybridA_ST_B201_' + $RunTag)
$figures = Join-Path $PSScriptRoot ('图片/HybridA_ST_B201_' + $RunTag)
if ((Test-Path -LiteralPath $output) -or (Test-Path -LiteralPath $figures)) {
    throw '输出目录已存在，请更换 RunTag；已有结果不会覆盖。'
}
& py -3.11 -X utf8 (Join-Path $reproduction 'run_hybrid_astar_st.py') --baseline $baseline --output $output
if ($LASTEXITCODE -ne 0) { throw 'B201 规划或执行失败，请检查输出。' }
& py -3.11 -X utf8 (Join-Path $reproduction 'audit_hybrid_st_b201.py') --run $output
if ($LASTEXITCODE -ne 0) { throw 'B201 结果核验未通过。' }
& py -3.11 -X utf8 (Join-Path $reproduction 'plot_hybrid_st_b201.py') --baseline $baseline --run $output --output $figures
if ($LASTEXITCODE -ne 0) { throw 'B201 图表生成失败。' }
Write-Host ('仿真数据：' + $output)
Write-Host ('图片：' + $figures)
