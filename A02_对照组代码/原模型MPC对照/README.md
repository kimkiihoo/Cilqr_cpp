# 原模型 MPC 对照适配模块

本模块将 `A01_原始论文与原始代码/原始代码/Cilqr_cpp-CAV/Cilqr_cpp-CAV` 中的运动学模型、R-CILQR / AL-iLQR 优化器及精确代价函数接入独立 C++ 动态链接库 (`bridge.dll`)。基准算法 SQP-NMPC 通过 Python `ctypes` 接口直接复用此动态库执行非线性运动学单打靶状态前向滚推与标量目标泛函计算，建立完全一致的同台比较环境。

## 核心复用与对齐内容

- **源代码完全对齐**：`generated/ilqr.cpp` 和 `generated/config_loader.cpp` 与原始代码保持算法和数学公式全等；系统运动学模型、优化器主循环、对数屏障惩罚与增广拉格朗日代价评估均在该两份源文件中实现；
- **桥接友元结构**：`ilqr.h` 头文件中引入 `friend struct OriginalBridgeAccess`，以便 C++ 导出层直接获取优化上下文，并消除跨编译单元内联符号链接警告；
- **物理极限与约束评估**：
  - `compute_articulated_limits`: 计算铰接转向最大曲率 $\kappa_{max} = \frac{\tan\gamma_{mech}}{l_r + l_f \cos\gamma_{mech}}$ 与最小转弯半径 $R_{min} = 1/\kappa_{max}$；
  - `finalize_parameters`: 根据期望巡航车速 $v_{des}$ 动态派生铰接角速度动态约束区间，例如 $v_{des}=5$ m/s 对应 $\dot{\gamma} \in [-0.15685, 0.15685]$ rad/s，$v_{des}=15$ m/s 对应 $[-0.40, 0.40]$ rad/s；
  - `predict_obstacle_trajectory`: 采用匀速直线运动模型外推 $N$ 步动态障碍物状态；
  - `compute_max_violation`: 全量核验 4 类非线性约束残差 (铰接角、铰接角速度、纵向加速度、避障椭圆净距与道路走廊边界)；
- **终点自适应减速机制**：保留 R-CILQR 在距终点 10 m 范围内的连续乘法衰减逻辑；AL-iLQR 保持其独立目标速度，不产生混淆。

## R-CILQR 与 AL-iLQR 的算法数学差异

| 评估维度 | R-CILQR (Control-limited Barrier iLQR) | AL-iLQR (Augmented Lagrangian iLQR) |
| :--- | :--- | :--- |
| **基础二次项** | 采用状态对角矩阵 $Q \in \mathbb{R}^{4 \times 4}$ 与控制矩阵 $R \in \mathbb{R}^{2 \times 2}$ 评估 | 采用完整二次型形式评估 (在当前对角配置下与 R 相同) |
| **车道与避障惩罚** | 针对左右道路边界与全部动态障碍物累加指数对数屏障罚项 | 每时刻取最劣边界与最近障碍物进入增广拉格朗日罚项 |
| **铰接角与角速率** | 使用平滑饱和指数屏障罚项 (作为目标泛函中的软约束) | 外层引入对偶拉格朗日乘子，前向滚推阶段按物理界限执行硬截断 |
| **控制输入外限幅** | 前向通道实施 $v \in [-5, 30]$ m/s, $\dot{\gamma} \in [-1.5, 1.5]$ rad/s，并限制预测内部相邻时步速度差 | 前向通道实施 $v \in [0, 25]$ m/s，角速率按配置截断，无加速度额外截断 |
| **终点减速逻辑** | 当与目标点距离 $d < 10.0$ m 时，期望车速按比例 $v_{des} \cdot (d / 10)$ 线性衰减 | 无内置终点速度衰减机制 |

## SQP-NMPC 的实现方式

- 基于 `scipy.optimize.minimize` (SLSQP 算法)，调用底层 C++ 动态库的 `SystemModel::dynamics` 实现单打靶连续模拟；
- 目标泛函直接复用 R-CILQR 的 `cal_cost_with_logging`，包含完全相同的状态偏差、控制增量及几何软屏障罚项；
- 控制变量上下界施加盒式约束 ($v \in [0, 25]$ m/s, $\dot{\gamma} \in [\dot{\gamma}_{min}, \dot{\gamma}_{max}]$)，预测序列相邻速度差施加加速度线性不等式约束；
- 因此定义为“**采用原 R-CILQR 运动学模型、参考点匹配逻辑与精确代价目标的 SQP-NMPC 对照基准**”。

## 运行与操作命令

在 PowerShell 环境下执行：

### 1. 动态链接库编译
```powershell
py -3.11 .\build_bridge.py
```
调用 MinGW-w64 GCC 与 Ninja 编译生成 `bridge.dll`。

### 2. 单状态接口一致性核验
```powershell
py -3.11 .\single_state.py --scene <scene.json> --effective-config <effective_config.json> --mode baseline --output baseline_eval.json
py -3.11 .\single_state.py --scene <scene.json> --effective-config <effective_config.json> --mode sqp --maxiter 2 --output sqp_eval.json
```

### 3. 多场景完整批量推进
```powershell
# 先导性 10 周期测试
py -3.11 .\run_maps.py --mode pilot --maps B201 --methods r al sqp --pilot-cycles 10 --output .\pilot_test

# 正式全场景固定窗口仿真 (B201: 400步, B207: 300步, B301: 200步)
py -3.11 .\run_maps.py --mode full --maps B201 B207 B301 --methods r al sqp --output .\full_benchmark
```

## 数据记录与指标格式

每批实验输出包含：
- `cycles.csv`: 记录每周期的规划时间、首步控制输入 $[v, \dot{\gamma}]$、状态量 $[x, y, \theta, \gamma]$、求解耗时与收敛状态；
- `cycles.jsonl` / `attempts.jsonl`: 记录详细优化迭代历程与求解器尝试记录；
- `trajectories.npz`: 结构化多维数组，存储全部周期完整的未来 $N$ 步预测轨迹与控制序列；
- `run_audit.json`: 记录运行审计报告（非有限数值统计、最大违例度、时序一致性）。
