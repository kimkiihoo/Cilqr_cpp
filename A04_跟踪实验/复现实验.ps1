param(
    [string]$PythonExe = 'C:\Users\snh48\AppData\Local\Programs\Python\Python311\python.exe',
    [string]$OutputDirectory = ''
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $PythonExe)) { throw '请通过 -PythonExe 指定已安装所需依赖的 Python 3.11。' }
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $PSScriptRoot ('复现输出_' + (Get-Date -Format 'yyyyMMdd_HHmmss')) }
if (Test-Path -LiteralPath $OutputDirectory) { throw '输出目录已存在，请指定新目录，以免覆盖原始数据。' }
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot '输入快照'),(Join-Path $PSScriptRoot '代码') -Destination $OutputDirectory -Recurse
& $PythonExe -B (Join-Path $OutputDirectory '代码\ltv_mpc_tracking.py') --map B201 --horizon 60 --cruise-speed 3 --rate-fraction 0.72
if ($LASTEXITCODE -ne 0) { throw '跟踪实验运行失败，已保留记录。' }
& $PythonExe -B (Join-Path $OutputDirectory '代码\audit_ltv_mpc.py')
if ($LASTEXITCODE -ne 0) { throw '核验脚本运行失败。' }
& $PythonExe -B (Join-Path $OutputDirectory '代码\plot_ltv_mpc.py')
if ($LASTEXITCODE -ne 0) { throw '绘图失败。' }
& $PythonExe -B (Join-Path $OutputDirectory '代码\summarize_ltv_mpc.py')
if ($LASTEXITCODE -ne 0) { throw '汇总失败。' }
Write-Host ('复现输出：' + $OutputDirectory)
