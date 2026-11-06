param([string]$PythonExe = 'C:\Users\snh48\AppData\Local\Programs\Python\Python311\python.exe')
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $PythonExe)) { throw '请通过 -PythonExe 指定已安装所需依赖的 Python 3.11。' }
& $PythonExe -B (Join-Path $PSScriptRoot '代码\plot_ltv_mpc.py')
if ($LASTEXITCODE -ne 0) { throw '绘图失败。' }
Write-Host (Join-Path $PSScriptRoot '图片')
