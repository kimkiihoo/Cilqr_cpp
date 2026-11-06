"""基于离散时变模型预测控制 (LTV-MPC) 的铰接式矿用车辆路径跟踪与避障算法。

本模块实现铰接式车辆在非结构化复杂矿区环境下的轨迹跟踪控制：
1. 动力学与运动学模型：
   - 考虑6自由度状态空间 x = [p_x, p_y, \theta, \gamma, v, \omega]^T；
   - 包含纵向驱动/制动与铰接角转向机构的一阶惯性响应滞后。
2. 控制框架：
   - 采用离散时变模型预测控制 (LTV-MPC)，预测时域 N = 60 (时域长度 3.0 s，控制周期 DT = 0.05 s)；
   - 沿时变名义参考状态进行连续雅可比线性化与精确矩阵指数离散化；
   - 构建 OSQP 二次规划求解器，融合高增益横向偏差惩罚、执行器物理限幅、局部信任域以及时变超平面动态避障。
3. 参考轨迹处理：
   - 保留原上层规划的全部时间戳、位姿、速度及铰接角速度，不重新定时；
   - 严格避免基于实时测量位置进行动态投影对齐，杜绝因横向误差扰动导致参考时间序列跳变。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import osqp
import scipy
from scipy import sparse
from scipy.linalg import expm
from scipy.spatial import cKDTree

# 路径与环境配置
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location("tracking_shared", HERE / "tracking_experiment.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)

# ==============================================================================
# 物理与控制超参数定义
# ==============================================================================
DT = .05      # MPC 控制与离散优化周期 [单位: s] (20 Hz 更新频率)
H = .01       # 仿真环境真实系统非线性动力学 RK4 数值积分步长 [单位: s] (100 Hz 仿真精度)
NX = 6        # 状态空间向量维度: [p_x, p_y, \theta, \gamma, v, \omega]
NU = 2        # 控制输入向量维度: [u_v, u_\omega]
TAU_V = .20   # 纵向车速执行机构一阶惯性滞后时间常数 [单位: s]
TAU_W = .15   # 铰接角速度执行机构一阶惯性滞后时间常数 [单位: s]


def read_json(path):
    """读取 JSON 格式配置文件并反序列化为 Python 字典。

    参数:
        path (str | Path): JSON 文件路径。

    返回:
        dict | list: 解析后的配置对象。
    """
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, obj):
    """将数据对象以 UTF-8 编码格式化写入 JSON 文件。

    参数:
        path (str | Path): 目标保存文件路径。
        obj (Any): 待序列化的数据对象。
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_csv(path, rows):
    """将字典列表以 UTF-8-SIG 编码格式导出为 CSV 数据表。

    参数:
        path (str | Path): 目标 CSV 文件路径。
        rows (list[dict]): 包含数据行的字典列表。
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wrap(x):
    """将角度值归一化至主值区间 [-pi, pi]。

    参数:
        x (float | np.ndarray): 输入角度值 [单位: rad]。

    返回:
        float | np.ndarray: 截断至 [-pi, pi] 的角度值 [单位: rad]。
    """
    return np.arctan2(np.sin(x), np.cos(x))


def unit(theta):
    """根据给定航向角计算二维笛卡尔坐标系下的单位方向向量。

    参数:
        theta (float): 航向角 [单位: rad]。

    返回:
        np.ndarray: 二维单位方向向量 [cos(theta), sin(theta)]^T [无量纲]。
    """
    return np.array([math.cos(theta), math.sin(theta)])


def rhs(x, u, p):
    """计算铰接式车辆非线性运动学与执行器动态的状态导数右端项 (ODE)。

    系统状态方程:
        \dot{p}_x = v * cos(\theta)
        \dot{p}_y = v * sin(\theta)
        \dot{\theta} = -(v * sin(\gamma) + l_r * \omega) / (l_r + l_f * cos(\gamma))
        \dot{\gamma} = \omega
        \dot{v} = clip((u_v - v) / \tau_v, acc_min, acc_max)
        \dot{\omega} = (u_\omega - \omega) / \tau_\omega

    参数:
        x (np.ndarray): 当前车辆状态向量，形状 (6,):
            x[0]: 前轴中心坐标 p_x [单位: m]
            x[1]: 前轴中心坐标 p_y [单位: m]
            x[2]: 前车体全局航向角 \theta [单位: rad]
            x[3]: 前后车体相对铰接角 \gamma [单位: rad]
            x[4]: 纵向线速度 v [单位: m/s]
            x[5]: 铰接角变化率 \omega = \dot{\gamma} [单位: rad/s]
        u (np.ndarray): 控制输入指令向量，形状 (2,):
            u[0]: 期望车速控制指令 u_v [单位: m/s]
            u[1]: 期望铰接角速率控制指令 u_\omega [单位: rad/s]
        p (dict): 车辆几何与动力学限幅参数字典:
            p["lf"]: 前轴中心至铰接连接点的纵向轴距 [单位: m]
            p["lr"]: 后轴中心至铰接连接点的纵向轴距 [单位: m]
            p["acc_min"]: 车辆最大允许制动减速度负向界限 [单位: m/s^2]
            p["acc_max"]: 车辆最大允许驱动加速度正向界限 [单位: m/s^2]

    返回:
        np.ndarray: 状态向量关于时间的一阶导数 dx/dt，形状 (6,)。

    异常:
        FloatingPointError: 当分母 |l_r + l_f * cos(\gamma)| < 0.1 m 时，判定接近运动学几何奇异点并抛出异常。
    """
    theta, gamma, v, w = x[2:]
    den = p["lr"] + p["lf"] * math.cos(gamma)
    if abs(den) < .1:
        raise FloatingPointError("kinematic singularity")
    return np.array([v * math.cos(theta), v * math.sin(theta),
                     -(v * math.sin(gamma) + p["lr"] * w) / den,
                     w, np.clip((u[0] - v) / TAU_V, p["acc_min"], p["acc_max"]),
                     (u[1] - w) / TAU_W])


def step(x, u, p, dt=DT):
    """利用经典四阶龙格-库塔法 (RK4) 对非线性状态微分方程进行离散时间积分递推。

    参数:
        x (np.ndarray): 积分起始状态向量，形状 (6,)。
        u (np.ndarray): 当前积分步长内保持不变的控制指令，形状 (2,)。
        p (dict): 车辆模型参数字典。
        dt (float): 总推进时间跨度 [单位: s]，默认值为控制周期 DT。

    返回:
        np.ndarray: 积分推演得到的下一时刻状态向量 x(t + dt)，形状 (6,)。
    """
    out = x.copy()
    count = max(1, int(round(dt / H)))
    h = dt / count
    for _ in range(count):
        k1 = rhs(out, u, p)
        k2 = rhs(out + h * k1 / 2, u, p)
        k3 = rhs(out + h * k2 / 2, u, p)
        k4 = rhs(out + h * k3, u, p)
        out += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return out


def linearize(x, u, p):
    """计算连续时间系统雅可比矩阵，并采用矩阵指数法完成零阶保持 (ZOH) 离散化。

    线性化展开点为当前状态 x 与控制输入 u：
        A_c = \partial f / \partial x |_{x, u} \in \mathbb{R}^{6 \times 6}
        B_c = \partial f / \partial u |_{x, u} \in \mathbb{R}^{6 \times 2}
    离散化计算:
        M = [[A_c, B_c], [0, 0]] * DT
        exp(M) = [[A_d, B_d], [0, I]]

    参数:
        x (np.ndarray): 线性化工作点状态向量，形状 (6,)。
        u (np.ndarray): 线性化工作点控制输入向量，形状 (2,)。
        p (dict): 车辆几何与物理参数字典。

    返回:
        tuple[np.ndarray, np.ndarray]:
            - A_d (np.ndarray): 离散时间状态转移矩阵，形状 (6, 6)。
            - B_d (np.ndarray): 离散时间控制输入增益矩阵，形状 (6, 2)。
    """
    theta, gamma, v, w = x[2:]
    den = p["lr"] + p["lf"] * math.cos(gamma)
    num = v * math.sin(gamma) + p["lr"] * w
    A, B = np.zeros((6, 6)), np.zeros((6, 2))
    A[0, 2], A[0, 4] = -v * math.sin(theta), math.cos(theta)
    A[1, 2], A[1, 4] = v * math.cos(theta), math.sin(theta)
    A[2, 3] = -(v * math.cos(gamma) * den + num * p["lf"] * math.sin(gamma)) / den ** 2
    A[2, 4], A[2, 5], A[3, 5] = -math.sin(gamma) / den, -p["lr"] / den, 1.
    # The QP separately constrains u_v-v to the unsaturated acceleration range.
    A[4, 4], A[5, 5] = -1 / TAU_V, -1 / TAU_W
    B[4, 0], B[5, 1] = 1 / TAU_V, 1 / TAU_W
    M = np.zeros((8, 8))
    M[:6, :6], M[:6, 6:] = A, B
    Md = expm(M * DT)
    return Md[:6, :6], Md[:6, 6:]


def rear(x, p):
    """根据前车体位姿与铰接几何关系计算后车体后轴中心坐标。

    后轴坐标几何闭环关系:
        p_rear = p_front - l_f * [cos(\theta), sin(\theta)]^T - l_r * [cos(\theta + \gamma), sin(\theta + \gamma)]^T

    参数:
        x (np.ndarray): 车辆状态向量，包含前轴中心 [x[0], x[1]]、航向角 x[2] 与铰接角 x[3]。
        p (dict): 车辆几何尺寸参数字典。

    返回:
        np.ndarray: 后轴中心二维笛卡尔坐标 [p_rx, p_ry]^T [单位: m]。
    """
    return x[:2] - p["lf"] * unit(x[2]) - p["lr"] * unit(x[2] + x[3])


def rectangle(center, theta, low, high, width):
    """根据中心参考点、局部纵向起止偏移量与横向宽度构建旋转定向包络矩形顶点。

    参数:
        center (np.ndarray): 基准中心参考点坐标 [x, y]^T [单位: m]。
        theta (float): 局部坐标系纵轴相对于全局 X 轴的旋转偏角 [单位: rad]。
        low (float): 沿车身纵轴向后的负向距离边界 [单位: m]。
        high (float): 沿车身纵轴向前的正向距离边界 [单位: m]。
        width (float): 车体横向全宽 [单位: m]。

    返回:
        np.ndarray: 顺时针/逆时针排列的4个外轮廓顶点全局坐标，形状 (4, 2) [单位: m]。
    """
    e = unit(theta)
    n = np.array([-e[1], e[0]])
    return center + np.array([high * e + width / 2 * n, high * e - width / 2 * n,
                              low * e - width / 2 * n, low * e + width / 2 * n])


def bodies(x, p):
    """计算铰接车辆前车体与后车体的扩展几何外轮廓矩形顶点。

    为确保避障严格安全，前后两个包络矩形均沿纵向延伸并覆盖中间铰接连接点。

    参数:
        x (np.ndarray): 车辆状态向量，形状 (6,)。
        p (dict): 包含车身宽度 width、包络长度 box_length 及轴距 lf, lr 的参数字典。

    返回:
        list[np.ndarray]: 包含前车体矩形与后车体矩形顶点数组的列表，各顶点形状 (4, 2)。
    """
    return [rectangle(x[:2], x[2], -p["lf"], p["box_length"] / 2, p["width"]),
            rectangle(rear(x, p), x[2] + x[3], -p["box_length"] / 2, p["lr"], p["width"])]


def overlap(a, b):
    """基于分离轴定理 (Separating Axis Theorem, SAT) 严格判断两个二维凸多边形是否发生几何重叠。

    参数:
        a (np.ndarray): 第一个凸多边形顶点序列，形状 (N_a, 2) [单位: m]。
        b (np.ndarray): 第二个凸多边形顶点序列，形状 (N_b, 2) [单位: m]。

    返回:
        bool: True 表示发生重叠 (碰撞)，False 表示存在分离超平面 (安全)。
    """
    for shape in (a, b):
        for edge in (shape[1] - shape[0], shape[2] - shape[1]):
            normal = np.array([-edge[1], edge[0]])
            normal /= np.linalg.norm(normal)
            pa, pb = a @ normal, b @ normal
            if pa.max() < pb.min() - 1e-10 or pb.max() < pa.min() - 1e-10:
                return False
    return True


def distance(a, b):
    """计算两个不相交凸多边形之间的精确最小欧氏几何间距。

    参数:
        a (np.ndarray): 第一个凸多边形顶点序列，形状 (4, 2) [单位: m]。
        b (np.ndarray): 第二个凸多边形顶点序列，形状 (4, 2) [单位: m]。

    返回:
        float: 两个多边形边界之间的最小欧几里得距离 [单位: m]；若相交则返回 0.0。
    """
    if overlap(a, b):
        return 0.
    best = math.inf
    for points, polygon in ((a, b), (b, a)):
        starts = polygon
        edges = np.roll(polygon, -1, axis=0) - starts
        delta = points[:, None, :] - starts[None, :, :]
        frac = np.clip(np.sum(delta * edges[None, :, :], axis=2) / np.sum(edges * edges, axis=1)[None, :], 0., 1.)
        delta -= frac[:, :, None] * edges[None, :, :]
        best = min(best, float(np.sqrt(np.sum(delta * delta, axis=2)).min()))
    return best


class Safety:
    """环境动态障碍物管理、精确几何间距计算与时变分离超平面构建类。"""

    def __init__(self, scene, p):
        """初始化安全评估器。

        参数:
            scene (dict): 包含静态路径 path 与初始障碍物参数 initial_obstacles 的场景字典。
            p (dict): 车辆与障碍物几何尺寸参数字典。
        """
        self.p = p
        self.initial_obstacles = np.asarray(scene["initial_obstacles"], float)
        self.path = np.asarray(scene["path"])
        self.tree = cKDTree(self.path[:, :2])

    def obstacles_at(self, actual_time):
        """根据绝对物理仿真时间预测并构建环境中所有动态障碍物的定向包络矩形。

        参数:
            actual_time (float): 当前绝对物理仿真时间 [单位: s]。

        返回:
            list[np.ndarray]: 所有动态障碍物矩形顶点坐标序列构成的列表，每个元素形状 (4, 2) [单位: m]。
        """
        p = self.p
        return [rectangle(o[:2] + actual_time * o[3] * unit(o[2]), o[2], -p["obs_length"] / 2,
                          p["obs_length"] / 2, p["obs_width"]) for o in self.initial_obstacles]

    def clearance(self, x, actual_time):
        """计算车辆在指定位姿下与环境中全部动态障碍物之间的绝对最小安全间距。

        参数:
            x (np.ndarray): 车辆状态向量，形状 (6,)。
            actual_time (float): 当前绝对仿真时间 [单位: s]。

        返回:
            float: 车辆前/后车体包络盒至障碍物的最小几何净距 [单位: m]。
        """
        # Exact rectangle distance for the extended front/rear body envelope.
        return min(distance(body, ob) for body in bodies(x, self.p) for ob in self.obstacles_at(actual_time))

    def planes(self, reference, actual_time, linearization=None):
        """根据分离轴方向构建局部凸线性化避障半空间约束超平面方程: grad^T * (x - ref) + constant >= margin。

        参数:
            reference (np.ndarray): 当前时步名义参考状态向量，形状 (6,)。
            actual_time (float): 当前预测时步对应的绝对仿真时间 [单位: s]。
            linearization (np.ndarray, optional): 外部指定的线性化状态基准点，默认使用 reference。

        返回:
            list[tuple[np.ndarray, float]]: 每个活跃避障约束的 (grad, constant) 元组列表，其中:
                - grad (np.ndarray): 距离残差关于状态向量的雅可比梯度，形状 (6,)。
                - constant (float): 展开点处的一阶常数项偏移 [单位: m]。
        """
        xbar = reference if linearization is None else linearization
        bs = bodies(reference, self.p)
        result = []
        for body_index, b in enumerate(bs):
            for ob in self.obstacles_at(actual_time):
                if np.linalg.norm(b.mean(axis=0) - ob.mean(axis=0)) > 13.:
                    continue
                candidates = []
                for shape in (b, ob):
                    for edge in (shape[1] - shape[0], shape[2] - shape[1]):
                        normal = np.array([-edge[1], edge[0]])
                        normal /= np.linalg.norm(normal)
                        if normal @ (b.mean(axis=0) - ob.mean(axis=0)) < 0:
                            normal *= -1
                        candidates.append((float((b @ normal).min() - (ob @ normal).max()), normal.copy()))
                _, normal = max(candidates, key=lambda item: item[0])
                ob_support = float((ob @ normal).max())
                def gap(x):
                    return float((bodies(x, self.p)[body_index] @ normal).min() - ob_support)
                g0 = gap(xbar)
                grad = np.zeros(6)
                for j in range(4):
                    offset = np.eye(6)[j] * 1e-5
                    grad[j] = (gap(xbar + offset) - gap(xbar - offset)) / 2e-5
                # g(x) = constant + grad @ (x-reference)
                constant = g0 - grad @ (xbar - reference)
                result.append((grad, constant))
        return result


class OriginalReference:
    """冻结原始时间轴：姿态线性插值，原速度/铰接角速度零阶保持。"""

    def __init__(self, source, p):
        self.source = source.copy()
        self.times = source[:, 0].copy()
        self.s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(source[:, 1:3], axis=0), axis=1))]
        self.travel_time = self.end_time = float(self.times[-1])
        if not np.all(np.diff(self.times) > 0):
            raise ValueError("Original timestamps must increase strictly")

    def sample(self, t):
        # Beyond the measured interval, hold only for MPC terminal prediction.
        # No added settling interval enters the simulation or error statistics.
        tc = float(np.clip(t, self.times[0], self.times[-1]))
        j = min(max(int(np.searchsorted(self.times, tc + 1e-10, side="right") - 1), 0), len(self.times) - 2)
        frac = float(np.clip((tc - self.times[j]) / (self.times[j+1] - self.times[j]), 0., 1.))
        pose = self.source[j, 1:5] + frac * (self.source[j+1, 1:5] - self.source[j, 1:5])
        velocity = self.source[j, 5:7].copy()
        if t > self.end_time + 1e-10:
            velocity[:] = 0.
        state = np.r_[pose, velocity]
        return state, velocity.copy(), {"path_s": float(self.s[j] + frac * (self.s[j+1] - self.s[j])),
                "original_time_s": tc, "original_index": j,
                "progress_speed_mps": float(abs(velocity[0]))}

    def save(self, folder):
        rows = []
        for t in np.arange(0., self.end_time + DT / 2, DT):
            x, u, meta = self.sample(float(t))
            rows.append({"t": float(t), **dict(zip(("x", "y", "theta", "gamma", "v", "gamma_dot"), map(float, x))),
                         "v_command_ff": float(u[0]), "gamma_dot_command_ff": float(u[1]), **meta,
                         "time_delay_from_original_s": float(t - meta["original_time_s"])})
        save_csv(folder / "sampled_reference.csv", rows)


class Controller:
    """有限时域离散时变模型预测控制器 (LTV-MPC) 核心类。

    在每个控制周期构建 OSQP 凸二次规划问题，求解满足动力学、避障超平面、
    控制量饱和及变化率限制的最优控制偏差序列。
    """

    def __init__(self, ref, p, safety, horizon=60, margin=.075, diagnostics=None):
        """初始化 LTV-MPC 控制器。

        参数:
            ref (OriginalReference): 原始时间序列参考生成器实例。
            p (dict): 车辆几何与物理参数字典。
            safety (Safety): 动态障碍物安全判定与分离超平面生成器。
            horizon (int): MPC 预测时域步数 N (时域总长 N * DT = 60 * 0.05 s = 3.0 s)，默认 60 步。
            margin (float): 时变分离超平面避障硬安全裕度 [单位: m]，默认 0.075 m。
            diagnostics (Path, optional): QP 诊断与矩阵快照输出目录。
        """
        self.ref, self.p, self.safety = ref, p, safety
        self.N, self.margin = horizon, margin
        self.previous = np.array([p["initial_speed"], 0.])
        self.last_u = None
        self.last_x = None
        self.cache = {}
        self.diagnostics = diagnostics

    def nominal(self, index):
        """获取并缓存指定时步的名义参考状态、前馈控制、线性化矩阵及避障超平面。

        参数:
            index (int): 预测时步索引。

        返回:
            tuple: (x, u, A, B, d, planes)，包含参考状态、前馈指令、线性化系统矩阵、一阶离散缺陷向量与避障约束。
        """
        if index not in self.cache:
            x, u, _ = self.ref.sample(index * DT)
            xn, _, _ = self.ref.sample((index + 1) * DT)
            A, B = linearize(x, u, self.p)
            d = step(x, u, self.p) - xn
            self.cache[index] = (x, u, A, B, d, self.safety.planes(x, index * DT))
        return self.cache[index]

    def solve(self, actual, index, refine=None):
        """构建并求解有限时域离散时变凸二次规划 (LTV-QP)。

        优化变量:
            z = [\Delta x_0^T, \dots, \Delta x_N^T, \Delta u_0^T, \dots, \Delta u_{N-1}^T]^T \in \mathbb{R}^{6(N+1) + 2N}
        优化目标:
            min \frac{1}{2} z^T P z + q^T z
            其中 P 为块对角矩阵，对角块包含:
            - 状态跟踪权重 2 * W_k:
              * 车身切向纵向位置误差权重: 60
              * 车身法向横向位置误差权重: 1200 (强横向纠偏)
              * 航向角误差权重: 100 [1/rad^2]
              * 铰接角误差权重: 15 [1/rad^2]
              * 车速误差权重: 4 [s^2/m^2]
              * 铰接角速度误差权重: 2 [s^2/rad^2]
              * 终端时步 k=N 权重倍率: 8.0 * W_k
            - 控制输入增量权重 2 * R_k = 2 * diag([0.2, 0.5])
            - 整体目标缩放因子 0.001 (改善 QP 条件数，不改变最优极值解)

        约束条件:
            1. 初始状态等式约束: \Delta x_0 = x_{actual} - x_{ref, 0}
            2. 动力学时变状态转移等式约束: \Delta x_{k+1} - A_k \Delta x_k - B_k \Delta u_k = d_k
            3. 状态盒式与局部信任域约束:
               - 铰接角 \gamma \in [\gamma_{min}, \gamma_{max}]
               - 车速 v \in [v_{min}, v_{max}]
               - 铰接角速度 \omega \in [\omega_{min}, \omega_{max}]
               - 局部信任域半径: |\Delta p_x| <= 1.5 m, |\Delta p_y| <= 1.5 m, |\Delta \theta| <= 0.4 rad
            4. 时变分离超平面避障不等式约束: grad^T \Delta x_k >= margin - constant
            5. 控制量幅值限幅与加速度 (输入变化率) 约束:
               - u_v \in [v_{min}, v_{max}], u_\omega \in [\omega_{min}, \omega_{max}]
               - 步间控制速度变化率 (a = \Delta u_v / DT) 严格位于 [acc_min, acc_max]
               - 考虑执行器滞后的稳态加速度 ((u_v - v_k) / TAU_V) 位于 [acc_min, acc_max]

        参数:
            actual (np.ndarray): 当前测量得到的系统真实状态向量，形状 (6,)。
            index (int): 当前离散控制周期索引 (t = index * DT)。
            refine (tuple, optional): SQP 迭代修正时的参考轨迹对，包含 (xbar, ubar)。

        返回:
            tuple[np.ndarray | None, np.ndarray | None, dict]:
                - predicted (np.ndarray | None): 预测时域内的绝对状态推演轨迹，形状 (N+1, 6)；求解失败返回 None。
                - controls (np.ndarray | None): 预测时域内的绝对控制指令序列，形状 (N, 2)；求解失败返回 None。
                - info (dict): 包含 QP 求解状态、残差、迭代次数与耗时的诊断信息字典。
        """
        N, p = self.N, self.p
        entries = [self.nominal(index + k) for k in range(N + 1)]
        refs = np.array([a[0] for a in entries])
        uff = np.array([a[1] for a in entries[:-1]])
        nstate, total = 6 * (N + 1), 6 * (N + 1) + 2 * N
        blocks = []
        for k, x in enumerate(refs):
            e = unit(x[2]); normal = np.array([-e[1], e[0]])
            W = np.diag([1., 1., 100., 15., 4., 2.])
            W[:2, :2] = 60 * np.outer(e, e) + 1200 * np.outer(normal, normal)
            if k == N:
                W *= 8.
            blocks.append(2 * W)
        blocks += [2 * np.diag([.2, .5]) for _ in range(N)]
        # A positive common objective multiplier leaves the QP minimizer unchanged.
        P = sparse.block_diag(blocks, format="csc") * .001
        rr, cc, vv, lower, upper = [], [], [], [], []
        def add(indices, values, low, high):
            row = len(lower)
            for c, v in zip(indices, values):
                if abs(v) > 1e-14:
                    rr.append(row); cc.append(int(c)); vv.append(float(v))
            lower.append(float(low)); upper.append(float(high))
        initial_error = actual - refs[0]
        initial_error[2] = float(wrap(initial_error[2]))
        for j in range(6):
            add([j], [1.], initial_error[j], initial_error[j])
        maxdefect = 0.
        for k in range(N):
            _, _, A, B, defect, _ = entries[k]
            if refine is not None:
                xbar, ubar = refine[0][k], refine[1][k]
                A, B = linearize(xbar, ubar, p)
                defect = step(xbar, ubar, p) - refs[k + 1] - A @ (xbar - refs[k]) - B @ (ubar - uff[k])
            maxdefect = max(maxdefect, float(abs(defect).max()))
            for j in range(6):
                add([6 * (k + 1) + j, *range(6 * k, 6 * k + 6), nstate + 2 * k, nstate + 2 * k + 1],
                    [1., *(-A[j]).tolist(), *(-B[j]).tolist()], defect[j], defect[j])
        for k in range(1, N + 1):
            x = refs[k]
            for j, low, high in [(3, p["gamma_min"], p["gamma_max"]), (4, p["v_min"], p["v_max"]),
                                  (5, p["gamma_dot_min"], p["gamma_dot_max"])]:
                add([6 * k + j], [1.], low - x[j], high - x[j])
            # Small-state-deviation trust region, identical for every solve.
            for j, radius in [(0, 1.5), (1, 1.5), (2, .4)]:
                add([6 * k + j], [1.], -radius, radius)
            planes = entries[k][5] if refine is None else self.safety.planes(x, (index + k) * DT, refine[0][k])
            for grad, constant in planes:
                add(range(6 * k, 6 * k + 6), grad, self.margin - constant, np.inf)
        for k in range(N):
            for j, lo, hi in [(0, p["v_min"], p["v_max"]), (1, p["gamma_dot_min"], p["gamma_dot_max"])]:
                add([nstate + 2 * k + j], [1.], lo - uff[k, j], hi - uff[k, j])
            if k == 0:
                bias = uff[k, 0] - self.previous[0]
                add([nstate], [1.], p["acc_min"] * DT - bias, p["acc_max"] * DT - bias)
            else:
                bias = uff[k, 0] - uff[k - 1, 0]
                add([nstate + 2 * k, nstate + 2 * (k - 1)], [1., -1.],
                    p["acc_min"] * DT - bias, p["acc_max"] * DT - bias)
            bias = uff[k, 0] - refs[k, 4]
            add([nstate + 2 * k, 6 * k + 4], [1., -1.], p["acc_min"] * TAU_V - bias, p["acc_max"] * TAU_V - bias)
        mat = sparse.csc_matrix((vv, (rr, cc)), shape=(len(lower), total))
        snapshot = None
        if index in (0, 60) and refine is None and self.diagnostics is not None:
            snapshot = "initial_qp" if index == 0 else "turn_qp"
            sparse.save_npz(self.diagnostics / (snapshot + "_P.npz"), P)
            sparse.save_npz(self.diagnostics / (snapshot + "_A.npz"), mat)
            np.savez(self.diagnostics / (snapshot + "_vectors.npz"), q=np.zeros(total),
                     lower=np.asarray(lower), upper=np.asarray(upper), reference_states=refs,
                     reference_commands=uff, measured_initial_state=actual)
        solver = osqp.OSQP()
        started = time.perf_counter()
        solver.setup(P=sparse.triu(P, format="csc"), q=np.zeros(total), A=mat,
                     l=np.asarray(lower), u=np.asarray(upper), verbose=False, polishing=True,
                     eps_abs=1e-5, eps_rel=1e-5, max_iter=30000, adaptive_rho=True,
                     scaled_termination=True, scaling=20, adaptive_rho_interval=25)
        if self.last_x is not None:
            guess_x = np.vstack([self.last_x[1:], self.last_x[-1]]) - refs
            guess_u = np.vstack([self.last_u[1:], self.last_u[-1]]) - uff
            solver.warm_start(x=np.r_[guess_x.ravel(), guess_u.ravel()])
        result = solver.solve(raise_error=False)
        wall = time.perf_counter() - started
        if snapshot is not None and result.x is not None:
            np.savez(self.diagnostics / (snapshot + "_solution.npz"), x=result.x, y=result.y)
            save_json(self.diagnostics / (snapshot + "_status.json"), {"cycle": index,
                      "time_s": index * DT, "status": result.info.status,
                      "primal_residual": float(result.info.prim_res), "dual_residual": float(result.info.dual_res)})
        info = {"qp_status": result.info.status, "qp_status_val": int(result.info.status_val),
                "qp_iterations": int(result.info.iter), "qp_primal_residual": float(result.info.prim_res),
                "qp_dual_residual": float(result.info.dual_res), "qp_solve_wall_s": wall,
                "max_affine_defect": maxdefect}
        if result.info.status_val not in (1, 2) or result.x is None or not np.isfinite(result.x).all():
            return None, None, info
        controls = result.x[nstate:].reshape(N, 2) + uff
        predicted = [actual.copy()]
        minclear = self.safety.clearance(actual, index * DT)
        maxviolation = 0.
        for k, u in enumerate(controls):
            next_state = step(predicted[-1], u, p)
            predicted.append(next_state)
            minclear = min(minclear, self.safety.clearance(next_state, (index + k + 1) * DT))
            maxviolation = max(maxviolation, p["gamma_min"] - next_state[3], next_state[3] - p["gamma_max"],
                               p["gamma_dot_min"] - next_state[5], next_state[5] - p["gamma_dot_max"])
        predicted = np.asarray(predicted)
        info.update({"nonlinear_horizon_min_clearance_m": minclear,
                     "nonlinear_horizon_max_state_violation": max(0., maxviolation)})
        return predicted, controls, info

    def command(self, state, index):
        """计算当前控制周期发往下层执行机构的最优控制指令，集成 SQP 细化与安全回退保障。

        参数:
            state (np.ndarray): 当前测量得到的被控对象真实状态向量，形状 (6,)。
            index (int): 当前控制周期计数 (t = index * DT)。

        返回:
            tuple[np.ndarray | None, dict]:
                - command (np.ndarray | None): 经安全投影截断后的最优控制指令 [u_v, u_\omega]^T [m/s, rad/s]；
                  若发生不可恢复的安全违背或求解失败则返回 None (触发停车停机)。
                - info (dict): 完整的控制器运行诊断记录字典。
        """
        predicted, controls, info = self.solve(state, index)
        first_info = info.copy()
        info["refinement_steps"] = 0
        info["qp_total_solve_wall_s"] = info["qp_solve_wall_s"]
        if predicted is not None and (info["nonlinear_horizon_min_clearance_m"] < .055 or info["nonlinear_horizon_max_state_violation"] > 1e-5):
            predicted, controls, refined = self.solve(state, index, (predicted, controls))
            info = {**refined, "refinement_steps": 1}
            info["qp_total_solve_wall_s"] = first_info["qp_solve_wall_s"] + refined["qp_solve_wall_s"]
        info["qp_initial_status"] = first_info["qp_status"]
        if predicted is None:
            info["fallback"] = "stop_without_executing_an_unverified_QP_command"
            return None, info
        raw_command = controls[0].copy()
        command = raw_command.copy()
        # Remove only solver-tolerance leakage at hard command bounds.
        lo = max(self.p["v_min"], self.previous[0] + self.p["acc_min"] * DT,
                 state[4] + TAU_V * self.p["acc_min"])
        hi = min(self.p["v_max"], self.previous[0] + self.p["acc_max"] * DT,
                 state[4] + TAU_V * self.p["acc_max"])
        if lo > hi + 1e-10:
            info["fallback"] = "incompatible_current_actuator_constraints_no_command_executed"
            return None, info
        command[0] = np.clip(command[0], lo, hi)
        command[1] = np.clip(command[1], self.p["gamma_dot_min"], self.p["gamma_dot_max"])
        controls[0] = command
        info["raw_v_command"], info["raw_gamma_dot_command"] = map(float, raw_command)
        info["command_projection_max_abs"] = float(abs(command - raw_command).max())
        info["command_speed_slew_mps2"] = float((command[0] - self.previous[0]) / DT)
        # Independent dense near-term nonlinear check before issuing the command.
        x = state.copy()
        clearance = self.safety.clearance(x, index * DT)
        violation = 0.
        for j in range(5):
            x = step(x, command, self.p, H)
            clearance = min(clearance, self.safety.clearance(x, index * DT + (j + 1) * H))
            violation = max(violation, x[3] - self.p["gamma_max"], self.p["gamma_min"] - x[3],
                            x[5] - self.p["gamma_dot_max"], self.p["gamma_dot_min"] - x[5])
        info["near_term_dense_min_clearance_m"] = clearance
        if clearance < .05 or violation > 1e-5:
            info["fallback"] = "stop_without_executing_an_unsafe_candidate"
            return None, info
        self.last_x, self.last_u, self.previous = predicted, controls, command.copy()
        info["fallback"] = "none"
        return command, info


def main():
    """LTV-MPC 路径跟踪仿真实验主入口函数。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", choices=["B201"], default="B201")
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--max-cycles", type=int, default=100000)
    args = ap.parse_args()
    folder = (args.output_dir or ROOT / "原始数据/LTV_MPC" / args.map / "cilqr").resolve()
    folder.mkdir(parents=True, exist_ok=True)
    p = shared.load_parameters(ROOT / "输入快照/full", args.map)
    # The runtime metadata rate is a soft planning barrier, not the forward-pass
    # hard input envelope. Match the original R-CILQR hard clipping (+/-1.5).
    p["planning_gamma_dot_min"] = p["gamma_dot_min"]
    p["planning_gamma_dot_max"] = p["gamma_dot_max"]
    control_limits = read_json(HERE / "ltv_control_limits.json")
    for key in ("gamma_dot_min", "gamma_dot_max", "v_min", "v_max"):
        p[key] = float(control_limits[key])
    scene_path = ROOT / "输入快照/full" / args.map / "scene/scene.json"
    scene = read_json(scene_path)
    original_path = folder / "original_reference.csv"
    original, source_audit = shared.load_frozen_reference(ROOT / "输入快照/full" / args.map / "cilqr", p, original_path)
    soft_excess = shared.violation(original[:-1, 6], p["planning_gamma_dot_min"], p["planning_gamma_dot_max"])
    source_audit.update({"planning_soft_rate_threshold_min_radps": p["planning_gamma_dot_min"],
                         "planning_soft_rate_threshold_max_radps": p["planning_gamma_dot_max"],
                         "planning_soft_rate_exceedance_intervals": int(np.count_nonzero(soft_excess > 1e-9)),
                         "planning_soft_rate_max_excess_radps": float(soft_excess.max())})
    save_json(folder / "original_reference_audit.json", source_audit)
    safety = Safety(scene, p)
    original_min_clear = min(safety.clearance(shared.sample_reference(original, float(t))[0], float(t))
                             for t in np.arange(0., original[-1, 0] + .005, .01))
    if original_min_clear <= 0:
        raise RuntimeError("Original R trajectory collides at its original timestamps")
    ref = OriginalReference(original, p)
    ref.save(folder)
    ctrl = Controller(ref, p, safety, args.horizon, diagnostics=folder)
    state = np.r_[original[0, 1:5], p["initial_speed"], 0.]
    command = np.array([p["initial_speed"], 0.])
    rows, cycles = [], []
    end = min(int(round(ref.end_time / H)), args.max_cycles * 5)
    reason = "original_reference_time_complete" if end == int(round(ref.end_time / H)) else "max_cycles_reached"
    started = time.perf_counter()
    config = read_json(ROOT / "输入快照/full" / args.map / "cilqr/effective_config.json")
    run_config = config["main"]["run_configs"][args.map]
    goal = np.array([run_config["goal_x"], run_config["goal_y"]], float)
    for i in range(end + 1):
        t = i * H
        desired, ff, meta = ref.sample(t)
        if i % 5 == 0 and i < end:
            controller_started = time.perf_counter()
            command, info = ctrl.command(state, i // 5)
            info["controller_wall_s"] = time.perf_counter() - controller_started
            cycles.append({"cycle": i // 5, "t": t, **info})
            if command is None:
                reason = info["fallback"]
                # Current measured state remains the terminal logged sample; no unsafe input is executed.
                command = ctrl.previous.copy()
                end = i
        front_delta = state[:2] - desired[:2]
        rear_delta = rear(state, p) - rear(desired, p)
        fl = float(front_delta @ np.array([-math.sin(desired[2]), math.cos(desired[2])]))
        rl = float(rear_delta @ np.array([-math.sin(desired[2] + desired[3]), math.cos(desired[2] + desired[3])]))
        clearance = safety.clearance(state, t)
        row = {"t": t, **dict(zip(("x", "y", "theta", "gamma", "v", "gamma_dot"), map(float, state))),
               **dict(zip(("ref_x", "ref_y", "ref_theta", "ref_gamma", "ref_v", "ref_gamma_dot"), map(float, desired))),
               "v_command": float(command[0]), "gamma_dot_command": float(command[1]),
               "v_command_raw": float(info.get("raw_v_command", command[0])),
               "gamma_dot_command_raw": float(info.get("raw_gamma_dot_command", command[1])),
               "command_speed_slew_mps2": float(info.get("command_speed_slew_mps2", 0.)),
               "acceleration_mps2": float(rhs(state, command, p)[4]),
               "front_lateral_error_m": fl, "rear_lateral_error_m": rl,
               "front_position_error_m": float(np.linalg.norm(front_delta)),
               "front_longitudinal_error_m": float(front_delta @ unit(desired[2])),
               "rear_position_error_m": float(np.linalg.norm(rear_delta)),
               "heading_error_rad": float(wrap(state[2] - desired[2])),
               "gamma_error_rad": float(wrap(state[3] - desired[3])), "speed_error_mps": float(state[4] - desired[4]),
               "extended_body_clearance_m": clearance, "body_collision": int(clearance <= 1e-9),
               "goal_distance_m": float(np.linalg.norm(state[:2] - goal)),
               **meta, "time_delay_from_original_s": float(t - meta["original_time_s"])}
        rows.append(row)
        if i == end:
            break
        state = step(state, command, p, H)
        if not np.isfinite(state).all():
            raise FloatingPointError("nonfinite actual plant state")
    save_csv(folder / "nominal.csv", rows)
    # QP-failure rows may omit prediction fields; preserve all columns explicitly.
    keys = list(dict.fromkeys(k for r in cycles for k in r))
    save_csv(folder / "mpc_cycles.csv", [{k: r.get(k, "") for k in keys} for r in cycles])
    summary = {"map": args.map, "planner": "R-CILQR", "controller": "LTV-MPC", "case": "nominal",
               "stop_reason": reason, "samples": len(rows), "duration_s": rows[-1]["t"],
               "original_duration_s": ref.travel_time, "reference_mode": "original_time",
               "scheduled_time_completed": bool(rows[-1]["t"] >= ref.end_time - 1e-8),
               "simulation_wall_s": time.perf_counter() - started,
               "min_extended_body_clearance_m": min(r["extended_body_clearance_m"] for r in rows),
               "collision_samples": sum(r["body_collision"] for r in rows),
               "final_goal_distance_m": rows[-1]["goal_distance_m"], "final_speed_mps": rows[-1]["v"],
               "original_reference_min_clearance_m": original_min_clear,
               "qp_failures": sum(r["qp_status_val"] not in (1, 2) for r in cycles),
               "fallback_cycles": sum(r["fallback"] != "none" for r in cycles),
               "original_reference_csv": str(original_path.resolve()), "original_reference_sha256": shared.sha(original_path),
               "actual_csv_sha256": shared.sha(folder / "nominal.csv"),
               "sampled_reference_sha256": shared.sha(folder / "sampled_reference.csv")}
    for key in ("front_lateral_error_m", "rear_lateral_error_m", "front_position_error_m", "rear_position_error_m", "front_longitudinal_error_m", "heading_error_rad", "gamma_error_rad", "speed_error_mps"):
        a = np.array([r[key] for r in rows])
        summary[key + "_rmse"] = float(np.sqrt(np.mean(a ** 2)))
        summary[key + "_max_abs"] = float(abs(a).max())
    for name, low, high in [("gamma", p["gamma_min"], p["gamma_max"]), ("v", p["v_min"], p["v_max"]),
                            ("gamma_dot", p["gamma_dot_min"], p["gamma_dot_max"]),
                            ("gamma_dot_command", p["gamma_dot_min"], p["gamma_dot_max"]),
                            ("v_command", p["v_min"], p["v_max"]),
                            ("acceleration_mps2", p["acc_min"], p["acc_max"])]:
        a = np.array([r[name] for r in rows])
        summary[name + "_max_violation"] = float(np.maximum(np.maximum(low - a, a - high), 0.).max())
    slew = np.array([r["command_speed_slew_mps2"] for r in rows])
    summary["command_speed_slew_max_violation"] = float(np.maximum(np.maximum(p["acc_min"] - slew, slew - p["acc_max"]), 0.).max())
    summary["max_original_time_shift_s"] = float(max(abs(r["time_delay_from_original_s"]) for r in rows))
    summary["reference_preserved"] = True
    summary["actual_gamma_dot_max_abs_radps"] = float(max(abs(r["gamma_dot"]) for r in rows))
    summary["command_gamma_dot_max_abs_radps"] = float(max(abs(r["gamma_dot_command"]) for r in rows))
    summary["goal_reached_and_stopped"] = bool(summary["scheduled_time_completed"] and summary["final_goal_distance_m"] <= 1.5 and abs(summary["final_speed_mps"]) <= .5)
    summary["engineering_precision_target_met"] = bool(summary["goal_reached_and_stopped"] and all(summary[k + "_rmse"] <= .1 and summary[k + "_max_abs"] <= .3 for k in ("front_lateral_error_m", "rear_lateral_error_m")))
    summary["sampled_safety_target_met"] = bool(summary["goal_reached_and_stopped"] and summary["min_extended_body_clearance_m"] >= .05 and summary["collision_samples"] == 0 and all(v <= 1e-5 for k, v in summary.items() if k.endswith("max_violation")))
    summary["controller_wall_mean_s"] = float(np.mean([r["controller_wall_s"] for r in cycles]))
    summary["controller_wall_max_s"] = float(max(r["controller_wall_s"] for r in cycles))
    summary["qp_total_solve_wall_mean_s"] = float(np.mean([r["qp_total_solve_wall_s"] for r in cycles]))
    save_json(folder / "summary.json", summary)
    protocol = {"scope": "Original R-CILQR B201 40-second trajectory, unchanged timing, pose and applied inputs; nominal LTV-MPC tracking",
        "parameters": p, "controller_dt_s": DT, "plant_dt_s": H, "tau_v_s": TAU_V, "tau_gamma_dot_s": TAU_W,
        "horizon_steps": args.horizon, "horizon_s": args.horizon * DT,
        "reference_mode": "original_time",
        "control_limits": control_limits,
        "control_limits_sha256": shared.sha(HERE / "ltv_control_limits.json"),
        "rate_limit_source": "Original R-CILQR src/ilqr.cpp forward-pass clipping uses [-1.5,1.5] rad/s; initial_cpp_parameters gamma_dot limits are soft barrier thresholds, recorded separately as planning_gamma_dot_min/max. Numerical control envelope, not identified hardware capability.",
        "progress_rule": "Original timestamps and all 401 poses retained; poses linearly interpolated in original time and the 400 applied speed/rate inputs held on their original 0.1-s intervals. No retiming, smoothing or measurement-based phase alignment.",
        "terminal": "Simulation stops at the original final timestamp (40 s). A stationary terminal hold is used only beyond 40 s inside the MPC prediction horizon.",
        "obstacle_clock": "All obstacle centers use scene initial pose + speed*actual_simulation_time. MPC horizon uses t_actual+k*0.05 and nonlinear checks use actual .01 times, never original-reference progress time.",
        "qp": "6-state Cartesian LTV model, 2 lagged actuator commands, finite horizon, OSQP. Full affine defect retained. Hard articulation bound and original planning numerical rate/speed envelopes, actual acceleration and commanded-speed-slew bounds. Linearized separating halfplanes for two joint-extended body rectangles and all nearby obstacles. Nonlinear horizon rollout and dense first-command check; at failure no unverified candidate is executed.",
        "weights": {"longitudinal_position":60., "lateral_position":1200., "heading":100., "gamma":15., "v":4., "gamma_dot":2., "command":[.2,.5], "terminal_multiplier":8.},
        "qp_numerics": "All variables are deviations from the time-varying reference. Objective multiplied uniformly by0.001; OSQP internal scaling20, scaled termination, adaptive rho interval25, polishing, abs/rel tolerance1e-5, maxiter30000. Hard physical/numeric bounds are unchanged.",
        "safety_geometry": "Front local longitudinal [-lf,+box_length/2], rear [-box_length/2,+lr], width=3m, rear heading=theta+gamma. Both rectangles include the joint. Obstacles 5x3m rectangles at constant-speed predictions in actual time. Linearized margin0.075m; issued-command nonlinear .01s check requires0.05m. No continuous guarantee is implied by .01s logs.",
        "error_definition": "Actual minus original reference at the same absolute time; front/rear lateral project onto the original front/rear heading normals. All .01s samples over the original 40 s are included.",
        "not_claimed": ["Tire/multibody dynamics or real-vehicle validation", "Online planner or solver-time delay is included", "QP status alone establishes safety"],
        "script_sha256":shared.sha(__file__), "shared_parameters_script_sha256":shared.sha(HERE/'tracking_experiment.py'),
        "python":sys.version, "numpy":np.__version__, "scipy":scipy.__version__, "osqp":osqp.__version__}
    save_json(folder / "protocol.json", protocol)
    print(json.dumps({k: summary[k] for k in ("duration_s", "stop_reason", "front_lateral_error_m_rmse", "rear_lateral_error_m_rmse", "min_extended_body_clearance_m", "final_goal_distance_m", "final_speed_mps", "engineering_precision_target_met", "sampled_safety_target_met")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
