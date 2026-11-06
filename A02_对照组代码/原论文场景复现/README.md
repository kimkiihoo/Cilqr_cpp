# 原论文地图场景复跑模块

本目录基于 `A01_原始论文与原始代码/原始代码/Cilqr_cpp-CAV/Cilqr_cpp-CAV` 中的原优化器、原运动学模型及 B201 / B207 / B301 地图重新生成仿真记录。其主要用途是复核原场景工况与验证算法基线链路，所有生成结果均保留完整逐帧记录与审计元数据。

## 运行方式

在 PowerShell 终端执行（输出目录若已存在则会保护性拒绝覆盖）：

```powershell
py -3.11 .\run_replay.py --mode handoff --output ..\..\仿真数据\原论文场景复跑_新批次
```

### 参数与控制逻辑说明
- 默认串行遍历 B301、B201、B207 三张基准地图，每张场景依次执行 C-iLQR 与 AL-iLQR，固定仿真窗口 200 周期、离散时间步长 $\Delta t = 0.1$ s；
- 环境变量 `OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1` 施加外部并发限制，但原 AL 求解器内部的 `std::async` 并行机制完整保留；
- 命令行参数支持：
  - `--build-only`: 仅执行 CMake / Ninja 编译并生成 `paper_replay.exe`；
  - `--maps B201`: 显式挑选指定的测试地图；
  - `--steps 400`: 覆盖默认步数（如扩展至 400 步）；
  - `--speed 5.0`: 覆盖期望巡航车速（如设定为 5.0 m/s）；
  - `--compiler` 与 `--ninja`: 指定 MinGW-w64 编译器与 Ninja 构建器绝对路径。

### 配置模式差异说明
- `handoff` 模式：保持原交接工程默认配置（预测时域 $N = 50$，期望巡航速度 $v_{des} = 15$ m/s）；
- `paper_table1` 模式：将预测时域切换为 $N = 80$。注意原论文部分绘图工况（如车速多在 0–5.5 m/s 范围）带有场景特异性，本模块不作未经确认的经验倒推。

## 数据流与结构

- `manifest.json`: 运行批次配置、环境元数据、源码 SHA-256 校验和与审计摘要；
- `<map>/scene/scene.json`: 场景冻结文件（全局参考路径、动态障碍物初态、场景哈希）；
- `<map>/<solver>/effective_config.json`: 三份 JSON 配置文件内容及 C++ 实际生效的关键参数；
- `<map>/<solver>/outputs/<map>/<solver>/data/<solver>_data_<cycle>.json`: 各周期离散优化解与环境帧序列化数据；
- `<map>/<solver>/cycles.csv`: 状态跟踪时序数据表（包含各周期自车位姿、实际首步控制量、求解耗时与收敛标志）；
- `<map>/<solver>/run_audit.json`: 非有限数值检测 (NaN/Inf) 与成对场景路径/障碍物全同性校验报告。

### 时序与物理量纲语义
- 规划周期 `cycle` 从 0 开始索引；
- 规划轨迹起点时刻为 $t_{plan} = cycle \cdot \Delta t$；
- 执行首步控制并更新后的自车状态与障碍物时刻为 $t = (cycle + 1) \cdot \Delta t$；
- 状态量量纲：$x, y$ [m], $\theta, \gamma$ [rad]；
- 控制输入量纲：$v$ [m/s], $\dot{\gamma}$ [rad/s]（历史字段名曾记为 `control_sequence.gamma`，实际物理意义为铰接角速度 $\dot{\gamma}$）；
- 原 AL-iLQR 的导出结构体未将内部收敛标志与迭代数写回 `Solution` 返回结构（保持默认 false/0/0），审计报告以 `convergence_fields_available=false` 如实标注，避免误判为优化失败。

## 源码透传适配说明

`adapt_original.py` 从 `A01_原始论文与原始代码` 生成 `source/` 构建代码，并生成差异清单 `adaptation.diff` 与哈希清单 `source_hashes.json`:
1. 移除运行时未启用的 C++ 绘图库 (matplotlib-cpp) 与 OpenCV 窗口调试调用，规避无显示环境下的程序阻塞；
2. 资源与输出路径统一改用独立工作目录下的相对路径，支持跨平台与含空格/非 ASCII 字符路径；
3. 为全局几何路径回退采样固定伪随机种子（`20261030u`），确保场景几何完全确定并跨求解器可复现；
4. 修正跨编译单元辅助函数 `find_closest_point` 的 `inline` 符号导出问题；
5. 优化算法内核 (`ilqr.cpp`, `hybrid_astar.cpp`, `config_loader.cpp`) 逻辑与数值计算公式保持 100% 原始等价。
