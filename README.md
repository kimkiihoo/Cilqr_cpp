# 重载铰接式无人车辆局部避障规划与轨迹跟踪控制

[![Python 3.11](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![C++17](https://img.shields.io/badge/C++-17-green.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![Build Tool](https://img.shields.io/badge/Build-CMake%20%7C%20Ninja-orange.svg)](https://ninja-build.org/)
[![Optimization](https://img.shields.io/badge/Solver-OSQP%20%7C%20SLSQP%20%7C%20iLQR-brightgreen.svg)](https://osqp.org/)

本工程针对非结构化矿区复杂道路场景下的重型铰接式无人车辆（Articulated Vehicle），研究高动态未知障碍物环境中的局部运动规划与高精度轨迹跟踪控制算法。

工程提供全套经过严格语法树（AST）校验与逐位数据全等核验（Bitwise Equal）的实验复现框架，涵盖 C-iLQR、AL-iLQR、SQP-NMPC、解耦两阶段 Hybrid A* + ST* 规划算法，以及基于超平面避障与执行器滞后补偿的 LTV-MPC 轨迹跟踪控制器。

---

## 1. 车辆建模与控制理论基础

### 1.1 铰接式底盘非线性运动学模型

与传统阿克曼前轮转向车辆不同，重载铰接式矿用车采用无前轮转向梯形机构的单铰接点结构，依靠前后车体之间的液压缸推拉产生铰接转角 $\gamma$ 改变行驶航向。其连续时间非完整运动学微分方程为：

$$\begin{cases}
\dot{p}_x = v \cos\theta \\
\dot{p}_y = v \sin\theta \\
\dot{\theta} = \dfrac{v \sin\gamma + \dot{\gamma} l_r}{l_r + l_f \cos\gamma} \\
\dot{\gamma} = \omega
\end{cases}$$

前向欧拉离散化形式（离散步长 $\Delta t$）定义为 $x_{k+1} = f(x_k, u_k)$：

$$\begin{cases}
p_{x, k+1} = p_{x, k} + v_k \cos\theta_k \cdot \Delta t \\
p_{y, k+1} = p_{y, k} + v_k \sin\theta_k \cdot \Delta t \\
\theta_{k+1} = \theta_k + \dfrac{v_k \sin\gamma_k + \omega_k l_r}{l_r + l_f \cos\gamma_k} \cdot \Delta t \\
\gamma_{k+1} = \gamma_k + \omega_k \cdot \Delta t
\end{cases}$$

### 1.2 规划层状态与控制量定义

- **状态向量** $x \in \mathbb{R}^4$:
  $$x = [p_x, p_y, \theta, \gamma]^T$$
  - $p_x$: 前轴中心在大地坐标系下的横坐标，量纲 $\mathrm{m}$；
  - $p_y$: 前轴中心在大地坐标系下的纵坐标，量纲 $\mathrm{m}$；
  - $\theta$: 前车体航向角（与大地坐标系 X 轴正向夹角），量纲 $\mathrm{rad}$；
  - $\gamma$: 铰接折角（前车体中轴线与后车体中轴线之间的相对偏角），量纲 $\mathrm{rad}$。
- **控制输入** $u \in \mathbb{R}^2$:
  $$u = [v, \omega]^T$$
  - $v$: 前轴驱动线速度，量纲 $\mathrm{m/s}$；
  - $\omega$: 铰接转角角速度 $\dot{\gamma}$，量纲 $\mathrm{rad/s}$。

### 1.3 跟踪控制层 6 状态扩展与执行器一阶滞后

针对底层液压执行机构与动力传动系统的惯性滞后，LTV-MPC 跟踪控制器将状态扩展为 6 维，显式引入一阶滞后响应：

$$\begin{cases}
\dot{v} = -\dfrac{1}{\tau_v} v + \dfrac{1}{\tau_v} u_v \\
\dot{\omega} = -\dfrac{1}{\tau_\omega} \omega + \dfrac{1}{\tau_\omega} u_\omega
\end{cases}$$

扩展状态向量为 $x_{track} = [p_x, p_y, \theta, \gamma, v, \omega]^T$，控制输入为执行器指令 $u_{track} = [u_v, u_\omega]^T$。时间常数设定为：
- 驱动响应滞后常数: $\tau_v = 0.20\,\mathrm{s}$
- 铰接转向液压滞后常数: $\tau_\omega = 0.15\,\mathrm{s}$

---

## 2. 局部规划与跟踪算法体系

```
                       ┌─────────────────────────┐
                       │  非结构化矿区地图与障碍物  │
                       └────────────┬────────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
       ┌───────────────────────────┐ ┌───────────────────────────┐
       │   耦合滚动时域优化 (MPC)   │ │  解耦两阶段规划 (HybridA*) │
       └─────────────┬─────────────┘ └─────────────┬─────────────┘
                     │                             │
         ┌───────────┼───────────┐                 │
         ▼           ▼           ▼                 ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐   ┌─────────────────┐
    │ C-iLQR  │ │ AL-iLQR │ │ SQP-NMPC│   │ 几何路径 + ST*  │
    └────┬────┘ └────┬────┘ └────┬────┘   └────────┬────────┘
         └───────────┼───────────┘                 │
                     ▼                             │
       ┌───────────────────────────┐               │
       │   离线重定时与时空参考轨迹   │◄──────────────┘
       └─────────────┬─────────────┘
                     ▼
       ┌───────────────────────────┐
       │   LTV-MPC 闭环跟踪控制器   │
       │ (超平面避障 + OSQP 凸优化) │
       └───────────────────────────┘
```

### 2.1 规划算法对比组特性

1. **C-iLQR (Control-limited Barrier iLQR)**:
   - 目标函数融合平滑对数屏障罚项（Smooth Log-Barrier），将铰接角极限、铰接角速度、纵向加速度、道路走廊与动态膨胀椭圆障碍物残差作为软惩罚项；
   - 采用微分动态规划（DDP）反向通道解析递推局部控制增益，结合 Armijo 准则回溯线搜索与自适应正则化阻尼因子 $\lambda$。
2. **AL-iLQR (Augmented Lagrangian iLQR)**:
   - 采用双层嵌套循环：外层交替更新拉格朗日乘子向量 $\lambda$ 与二次惩罚因子 $\mu$；
   - 内层将增广拉格朗日目标转化为无约束 iLQR 问题，严格保证障碍物碰撞残差与走廊边界的可行性。
3. **SQP-NMPC (Sequential Quadratic Programming NMPC)**:
   - 基于 SLSQP 算法，直接调用 C++ 底层动态库中相同的 `SystemModel::dynamics` 进行单打靶状态前向滚推；
   - 目标函数采用与 C-iLQR 完全同构的代价与软惩罚项，控制输入施加盒式边界约束与加速度线性不等式约束，作为精确对齐的数值基准。
4. **Hybrid A* + ST* (解耦两阶段对比算法)**:
   - **第一阶段（空间几何路径）**：在膨胀栅格地图上利用 3D 混合 A* 搜索连续曲率几何路径，经 QP 二次规划平滑；
   - **第二阶段（时空速度搜索）**：构建基于空间弧长、时间与速度的 $(s, t, v)$ 时空三维网格，结合几何曲率限速剖面与动态障碍物交互，利用 A* 算法搜索无碰撞最优时空速度曲线。

### 2.2 闭环轨迹跟踪控制器 (LTV-MPC)

- 在每个控制周期（$\Delta t_{track} = 0.05\,\mathrm{s}$）沿参考轨迹线性化为离散线性时变系统：
  $$\delta x_{k+1} = A_k \delta x_k + B_k \delta u_k$$
- **分离超平面几何避障约束**：
  采用 Shapely / 凸二次规划构建自车与障碍物之间的分离超平面：
  $$n_i^T p - d_i \ge 0$$
  将非线性碰撞区域转化为关于预测状态的凸线性不等式约束；
- **QP 求解器**：由稀疏矩阵凸优化库 OSQP 执行单步在线求解，具备高数值稳定性。

---

## 3. 物理参数与量纲速查表

### 3.1 车辆几何外形尺寸与物理常数

| 物理参数 | 符号表示 | 数值 | 量纲 / 单位 | 物理意义与约束作用 |
| :--- | :--- | :---: | :---: | :--- |
| **前轴距** | $l_f$ | `2.85` | $\mathrm{m}$ | 铰接点至前车体驱动轴轴心的纵向几何间距 |
| **后轴距** | $l_r$ | `2.85` | $\mathrm{m}$ | 铰接点至后车体从动轴轴心的纵向几何间距 |
| **车体全宽** | $width$ | `3.30` | $\mathrm{m}$ | 车辆横向最大几何宽度，用于走廊碰撞边界检验 |
| **碰撞盒长度** | $box\_length$ | `4.20` | $\mathrm{m}$ | 前/后车体几何包围盒的纵向覆盖长度 |
| **转向角极限** | $\gamma_{max}$ | `0.70` | $\mathrm{rad}$ | 铰接折角机械硬约束上限（约为 $\pm 40.1^\circ$） |
| **转向角速度极限** | $\omega_{max}$ | `0.40` | $\mathrm{rad/s}$ | 液压转向机构最大角速度动态界限 |
| **纵向加速度极限** | $a_{min}, a_{max}$ | `[-2.0, 2.0]` | $\mathrm{m/s^2}$ | 动力与制动系统最大极限加速度 |
| **规划控制周期** | $\Delta t$ | `0.10` | $\mathrm{s}$ | 局部滚动优化求解周期 |
| **跟踪控制周期** | $\Delta t_{track}$ | `0.05` | $\mathrm{s}$ | 底层 LTV-MPC 闭环跟踪执行周期 |
| **规划预测时域** | $N$ | `50 / 80` | $\mathrm{steps}$ | 局部滚动时域步数（对应 $5.0\,\mathrm{s}$ 或 $8.0\,\mathrm{s}$ 前瞻） |
| **跟踪预测时域** | $N_p$ | `20` | $\mathrm{steps}$ | 跟踪控制器前瞻时域步数（对应 $1.0\,\mathrm{s}$ 前瞻） |

### 3.2 权重系数与算法参数配置

#### 局部规划器参数 (ilqr.json)
- **状态偏差权重矩阵** $Q = \mathrm{diag}(1.0, 1.0, 0.5, 0.2)$（位置 $p_x, p_y$ 权重为 1.0，航向角 $\theta$ 权重为 0.5，铰接角 $\gamma$ 权重为 0.2）；
  > **注意**：原 `ilqr.h` 历史注释中曾提及“位置权重调整为 1.2”，经工程实测与源码核验，实际生效数值严格为 `0.2`，已在优化后的注释中据实更正。
- **控制输入增量权重矩阵** $R = \mathrm{diag}(0.1, 0.1)$（惩罚速度增量与角速度变化率）；
- **对数屏障惩罚系数**：$q_1 = 1.0, q_2 = 1.0$（控制屏障函数陡峭度与惩罚斜率）；
- **动态障碍物安全裕度**：长轴膨胀量 $a_{buffer} = 1.0\,\mathrm{m}$，短轴膨胀量 $b_{buffer} = 0.5\,\mathrm{m}$。

#### 跟踪控制器参数 (ltv_mpc_tracking.py)
- **跟踪误差权重矩阵** $Q_{track} = \mathrm{diag}(1.0, 6.0, 8.0, 2.0, 0.5, 0.4)$:
  - 纵向位置误差权重: `1.0`
  - 横向循迹误差权重: `6.0`（重点抑制横向跑偏）
  - 航向角误差权重: `8.0`（严格收敛航向姿态）
  - 铰接折角误差权重: `2.0`（平滑转向振荡）
  - 纵向速度跟踪权重: `0.5`
  - 铰接角速度跟踪权重: `0.4`
- **控制指令权重矩阵** $R_{track} = \mathrm{diag}(0.2, 0.2)$；
- **控制变化率权重矩阵** $R_{\Delta, track} = \mathrm{diag}(0.5, 0.5)$（防止驱动与转向超调跳变）。

---

## 4. 目录架构与数据依赖

```
.
├── AGENTS.md                                   # 学术写作与审稿返修规范约定
├── README.md                                   # 项目全局主文档 (本文件)
│
├── A01_原始论文与原始代码/                      # [完全冻结] 原始期刊论文手稿与交接代码源文件
│   ├── 原始代码/Cilqr_cpp-CAV/Cilqr_cpp-CAV/    # 原版 C++ 规划器源码与 JSON 地图
│   └── 原始论文/                               # 原始录用论文 PDF 及相关资料
│
├── A02_对照组代码/                              # 算法对比组实现与全量基准仿真数据
│   ├── 原模型MPC对照/                          # C-iLQR, AL-iLQR, SQP-NMPC 动态库桥接与运行器
│   │   ├── bridge.cpp                          # C-API 导出接口与约束残差计算
│   │   ├── build_bridge.py                     # GCC / Ninja 动态库自动化编译驱动
│   │   ├── run_maps.py                         # 三方法全场景批量仿真推进入口
│   │   └── sqp_nmpc.py                         # 基于 SLSQP 的基准算法实现
│   ├── 原论文场景复现/                          # 原始交接工程环境适配与场景重放二进制
│   │   ├── adapt_original.py                   # 原始代码沙盒化透传适配脚本
│   │   └── run_replay.py                       # 原论文 B301/B201/B207 场景复跑驱动
│   ├── 仿真数据/                               # 冻结的基准仿真数据集合
│   │   ├── 原版算法三方法基准_20261104/full/    # 三方法 400 步全场景数据与独立审计
│   │   └── HybridA_ST_B201_20261105/           # Hybrid A* + ST* 解耦算法仿真基准
│   └── 图片/                                   # 期刊出版级对比矢量插图生成源码
│
├── A03_论文修改后插图/                          # 返修修改后的高质量图表资产 (600 DPI, PDF, SVG)
│
├── A04_跟踪实验/                                # LTV-MPC 闭环轨迹跟踪控制与高精安全核验
│   ├── 代码/
│   │   ├── ltv_mpc_tracking.py                 # 6 状态 LTV-MPC 跟踪控制器核心实现
│   │   ├── tracking_experiment.py              # TVLQR / 积分推进与仿真框架
│   │   ├── audit_ltv_mpc.py                    # 基于 Shapely 的 0.002s 极密几何碰撞核验
│   │   └── summarize_ltv_mpc.py                # 跟踪指标汇总与 Markdown 报告生成
│   └── 原始数据/                               # 实际跟踪仿真轨迹与控制输入记录
│
└── A05_开销统计/                                # 细粒度毫秒级计算耗时剖析插桩与分析
    ├── 代码/
    │   ├── generated/include/overhead_profile.h # RAII 毫秒级阶段计时器 (线性化/导数/反向/前向)
    │   └── sqp_profile.py                      # SQP-NMPC 耗时统计基线
    ├── run_profile.py                          # 自动化剖析实验运行入口
    ├── summarize_profile.py                    # 阶段耗时统计与分布特性汇总
    └── audit_outputs.py                        # 400 周期三方法逐位一致性自动化核验
```

---

## 5. 编译与运行复现指南

### 5.1 环境依赖与工具链要求

- **操作系统**: Windows 10 / 11 (64-bit)
- **C++ 编译器**: MinGW-w64 GCC 13+ (支持 C++17，开启 `-O3` 优化)
- **构建工具**: CMake 3.20+, Ninja 1.10+
- **Python 运行时**: Python 3.11+
- **Python 依赖库**:
  ```powershell
  py -3.11 -m pip install numpy scipy matplotlib osqp shapely
  ```

### 5.2 编译 C++ 动态库与原生二进制

进入 `A02_对照组代码/原模型MPC对照/` 目录执行自动化编译：

```powershell
cd E:\BIT\01-lqr\第1次返修\A02_对照组代码\原模型MPC对照
py -3.11 .\build_bridge.py
```
> 该脚本会自动调用 CMake 和 Ninja，链接 `A02_对照组代码/eigen3/` 矩阵库，并在当前目录生成 `bridge.dll`。

### 5.3 运行规划算法基准实验

```powershell
# 1. 运行三方法 (C-iLQR, AL-iLQR, SQP-NMPC) B201 地图先导测试 (10 周期)
py -3.11 .\run_maps.py --mode pilot --maps B201 --methods r al sqp --pilot-cycles 10 --output .\pilot_results

# 2. 运行 B201 地图 Hybrid A* + ST* 解耦规划算法
cd E:\BIT\01-lqr\第1次返修\A02_对照组代码\仿真数据\HybridA_ST_B201_20261105\reproduction
py -3.11 .\run_hybrid_astar_st.py

# 3. 运行 Hybrid A* + ST* 单元测试套件 (7 项测试)
py -3.11 .\test_st_speed_search.py
```

### 5.4 运行 LTV-MPC 跟踪实验与独立核验

进入 `A04_跟踪实验/代码/` 目录：

```powershell
cd E:\BIT\01-lqr\第1次返修\A04_跟踪实验\代码

# 1. 执行全量 0.002 s 极密时钟采样与 Shapely 几何碰撞独立核验
py -3.11 .\audit_ltv_mpc.py

# 2. 汇总跟踪性能指标与工程验收结论
py -3.11 .\summarize_ltv_mpc.py

# 3. 绘制跟踪轨迹、横向偏差与控制时序图表
py -3.11 .\plot_ltv_mpc.py
```

### 5.5 运行计算开销细分剖析

进入 `A05_开销统计/` 目录：

```powershell
cd E:\BIT\01-lqr\第1次返修\A05_开销统计

# 1. 编译带 RAII 耗时插桩的 C++ 动态库并执行全量评测
py -3.11 .\run_profile.py

# 2. 统计各算法在模型线性化、代价导数、反向通道、前向通道与乘子更新的耗时分布
py -3.11 .\summarize_profile.py

# 3. 验证剖析运行结果与原始基准的逐位全等性
py -3.11 .\audit_outputs.py
```

---

## 6. 数值核验结论与基准指标

全套代码严格通过自动化无损校验体系，主要结论包括：

### 6.1 逐位全等性核验 (Bitwise Equal)
- 通过 `A05_开销统计/audit_outputs.py` 对 B201 场景下 C-iLQR、AL-iLQR 与 SQP-NMPC 的全部 400 个滚动推进周期（每个周期记录执行前后状态、未来预测轨迹、控制输入序列与动态障碍物坐标）进行浮点逐位全等检查；
- **核验结论**：在 400 个周期内，所有状态量、控制输入与预测轨迹的最大绝对偏差均为 `0.0`，完全达到 `bitwise_equal: true`，证明代码注释优化和接口整理对底层数值计算逻辑无任何改动。

### 6.2 LTV-MPC 轨迹跟踪工程验收指标 (B201 场景)

经 `audit_ltv_mpc.py` 独立核验（原始 401 顶点，总运行时间 64.80 s，共 6481 个离散样本）：

| 检验项目 | 测得数值 (前轴) | 测得数值 (后轴) | 约定工程目标 | 验收结论 |
| :--- | :---: | :---: | :---: | :---: |
| **空间横向 RMSE** | `0.0421 m` | `0.0582 m` | $\le 0.10\,\mathrm{m}$ | **通过** (指标优于限值约 50%) |
| **空间横向最大绝对偏差** | `0.2769 m` | `0.2209 m` | $\le 0.30\,\mathrm{m}$ | **通过** |
| **时间匹配横向 RMSE** | `0.0420 m` | `0.0584 m` | $\le 0.10\,\mathrm{m}$ | **通过** |
| **时间匹配横向最大偏差** | `0.2777 m` | `0.2209 m` | $\le 0.30\,\mathrm{m}$ | **通过** |
| **稠密几何碰撞采样数** | 32401 点 | 32401 点 | 0 碰撞点 | **通过** (全历程无碰撞) |
| **全历程最小碰撞净距** | `19.110 m` | `19.110 m` | $> 0.05\,\mathrm{m}$ | **通过** |
| **道路走廊横向越界** | 0 次越界 | 0 次越界 | 边界内 ($\pm 5.0\,\mathrm{m}$) | **通过** (最大绝对距离 2.45 m) |
| **终点停车精度** | 位置误差 `0.650 m` | 末端速度 `-6.2e-12 m/s` | 误差 $< 1.5\,\mathrm{m}$, 速度 $< 0.5\,\mathrm{m/s}$ | **通过** (平稳静止停靠) |

### 6.3 算法计算开销特性 (Calculation Time & Iterations)

依据通用学术规范与实测数据，各方法在单步规划中的计算开销与收敛特性对比如下：
- **模型线性化 (Linearization)**: 状态雅可比矩阵 $A_k, B_k$ 解析计算，占内迭代总耗时的约 15%–20%；
- **代价偏导数 (Cost Derivatives)**: 一阶与二阶偏导数 $l_x, l_u, l_{xx}, l_{uu}, l_{ux}$ 解析计算，占约 20%–25%；
- **反向通道递推 (Backward Pass)**: 黎卡提方程动态规划逐级逆推求取反馈增益 $k, K$，主要涉及 $2 \times 2$ 矩阵求逆；
- **前向通道搜索 (Forward Pass)**: 包含非线性运动学滚推与 Armijo 回溯步长搜索，主要受阻尼调节影响；
- **AL 外迭代 (AL Multipliers Update)**: 增广拉格朗日乘子与惩罚项更新，增加数轮外层宏循环。

详细统计数据与分布图表可由 `A05_开销统计/summarize_profile.py` 导出。
