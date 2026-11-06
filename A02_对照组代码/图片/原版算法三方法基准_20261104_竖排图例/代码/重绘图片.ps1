# 在当前目录运行本脚本，从“绘图数据”重绘“图片”；无需重跑仿真。
param(
    [string]$PythonPath = '',
    [string]$OutputDirectory = ''
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$plotArguments = @((Join-Path $PSScriptRoot 'plot_comparison.py'))
if ($OutputDirectory) { $plotArguments += @('--output', $OutputDirectory) }
if ($PythonPath) {
    & $PythonPath @plotArguments
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.11 @plotArguments
} else {
    & python @plotArguments
}
if ($LASTEXITCODE -ne 0) { throw '绘图失败。需要 Python、NumPy、Matplotlib 和 Times New Roman 字体。' }
