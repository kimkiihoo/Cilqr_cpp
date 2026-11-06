# 原模型 MPC 对照适配模块 (基准复现副本)

本目录为 `A02_对照组代码/原模型MPC对照` 在基准仿真数据集合下的镜像复现副本，保证三方法（R-CILQR, AL-iLQR, SQP-NMPC）基准测试环境与生成数据的自包含性。

详细说明请参见主模块文档: [A02_对照组代码/原模型MPC对照/README.md](../../../../原模型MPC对照/README.md)。

## 快速复现命令

在当前目录下执行独立复现与核验：

```powershell
# 1. 验证三方法仿真结果与冻结基准的逐位全等性 (Bitwise Equal)
py -3.11 .\verify_consistency.py

# 2. 执行独立物理指标与几何约束违例度核验
py -3.11 .\audit_original_model_mpc.py

# 3. 绘制多方法出版级对比图表
py -3.11 .\plot_original_model_comparison.py
```
