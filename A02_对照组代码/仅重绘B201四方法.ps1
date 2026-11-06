Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$baseline = Join-Path $PSScriptRoot '仿真数据/原版算法三方法基准_20261104/full'
& py -3.11 -X utf8 (Join-Path $baseline 'reproduction/plot_hybrid_st_b201.py') --baseline $baseline --run (Join-Path $PSScriptRoot '仿真数据/HybridA_ST_B201_20261105') --output (Join-Path $PSScriptRoot '图片/HybridA_ST_B201_20261105')
if ($LASTEXITCODE -ne 0) { throw 'B201 图表生成失败。' }
