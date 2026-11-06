"""基于时变线性二次型调节器 (TVLQR) 的铰接式车辆轨迹跟踪与对照实验框架。

本模块提供跟踪控制实验的公共支撑函数库与时变 LQR (TVLQR) 基准对比实现：
1. 动力学与运动学模型：
   - 6自由度状态空间 x = [p_x, p_y, 	heta, \\gamma, v, \\omega]^T；
   - 考虑执行机构一阶惯性滞后响应模型，真实系统采用 0.01 s 步长 RK4 积分推进。
2. 控制器设计：
   - 在直行标称工况下求解离散代数黎卡提方程 (DARE)，获得终值代价矩阵 P；
   - 沿时变连续参考轨迹反向迭代离散黎卡提差分方程，生成有限时域 TVLQR 时变增益矩阵序列 K_k；
   - 控制周期为 0.05 s，反馈量在参考航向局部切向/法向坐标系下计算。
3. 实验工况：
   - 包含标称工况 (Nominal) 与扰动工况 (Perturbed，包含横向偏差、航向角偏差及参数滞后时变失配)。
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import numpy as np
import scipy
from scipy.linalg import expm, solve_discrete_are
from scipy.spatial import cKDTree
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAPS = {'B201': 5.0, 'B207': 6.0, 'B301': 15.0}
METHODS = ('cilqr', 'alilqr', 'sqp')
PLANT_DT = 0.01
CONTROL_DT = 0.05
Q = np.diag([1.0, 6.0, 8.0, 2.0, 0.5, 0.4])
R = np.diag([2.0, 12.0])
CASES = ({'name': 'nominal', 'lateral_offset_m': 0.0, 'heading_offset_rad': 0.0, 'gamma_offset_rad': 0.0, 'tau_v_s': 0.2, 'tau_w_s': 0.15}, {'name': 'perturbed', 'lateral_offset_m': 0.25, 'heading_offset_rad': 0.02, 'gamma_offset_rad': 0.02, 'tau_v_s': 0.35, 'tau_w_s': 0.25})

def read_json(path):
    """读取 JSON 格式文件并反序列化为字典对象。

    参数:
        path (str | Path): 文件路径。

    返回:
        dict | list: 解析后的 JSON 对象。
    """
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    """计算指定二进制文件的 SHA-256 哈希校验码。

    参数:
        path (str | Path): 文件路径。

    返回:
        str: 64位十六进制哈希字符串。
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    """将数据对象以 UTF-8 编码格式化写入 JSON 文件。

    参数:
        path (str | Path): 目标保存路径。
        value (Any): 待序列化的数据。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def write_csv(path, rows, columns=None):
    """将字典行列表以 UTF-8-SIG 格式保存为 CSV 数据文件。

    参数:
        path (str | Path): 目标 CSV 文件路径。
        rows (list[dict]): 数据字典列表。
        columns (list[str], optional): 指定表头字段名列表。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def wrap(x):
    """将输入角度归一化至主值区间 [-pi, pi]。

    参数:
        x (float | np.ndarray): 输入角度值 [单位: rad]。

    返回:
        float | np.ndarray: 截断至主值区间的角度值 [单位: rad]。
    """
    return np.arctan2(np.sin(x), np.cos(x))

def direction(theta):
    """根据航向角计算平面二维单位方向向量。

    参数:
        theta (float): 航向角 [单位: rad]。

    返回:
        np.ndarray: 二维单位方向向量 [cos(theta), sin(theta)]^T [无量纲]。
    """
    return np.array([math.cos(theta), math.sin(theta)])

def rear_axle(state, p):
    """根据前轴中心位姿与车辆轴距几何计算后轴中心二维坐标。

    几何关系:
        p_rear = p_front - l_f * [cos(	heta), sin(	heta)]^T - l_r * [cos(	heta + \\gamma), sin(	heta + \\gamma)]^T

    参数:
        state (np.ndarray): 状态向量，前4维为 [x, y, 	heta, \\gamma]。
        p (dict): 包含前后轴距 lf, lr 的参数字典 [单位: m]。

    返回:
        np.ndarray: 后轴中心二维坐标 [p_rx, p_ry]^T [单位: m]。
    """
    return state[:2] - p['lf'] * direction(state[2]) - p['lr'] * direction(state[2] + state[3])

def plant_rhs(state, command, p, tau_v=0.2, tau_w=0.15):
    """计算带一阶执行器滞后的铰接车辆非线性运动学状态导数右端项 (ODE)。

    状态方程:
        \\dot{p}_x = v * cos(	heta)
        \\dot{p}_y = v * sin(	heta)
        \\dot{	heta} = -(v * sin(\\gamma) + l_r * \\omega) / (l_r + l_f * cos(\\gamma))
        \\dot{\\gamma} = \\omega
        \\dot{v} = clip((u_v - v) / 	au_v, acc_min, acc_max)
        \\dot{\\omega} = (u_\\omega - \\omega) / 	au_\\omega

    参数:
        state (np.ndarray): 6维状态向量 [x, y, 	heta, \\gamma, v, \\omega]^T [m, m, rad, rad, m/s, rad/s]。
        command (np.ndarray): 2维控制指令 [u_v, u_\\omega]^T [m/s, rad/s]。
        p (dict): 车辆几何与动力学限幅参数字典。
        tau_v (float): 纵向速度响应时间常数 [单位: s]，默认 0.20 s。
        tau_w (float): 铰接转向响应时间常数 [单位: s]，默认 0.15 s。

    返回:
        np.ndarray: 状态关于时间的一阶导数 dx/dt，形状 (6,)。

    异常:
        FloatingPointError: 当运动学几何分母低于 0.1 m 时抛出奇异点异常。
    """
    _, _, theta, gamma, speed, rate = state
    den = p['lr'] + p['lf'] * math.cos(gamma)
    if abs(den) < 0.1:
        raise FloatingPointError('articulated kinematic denominator below 0.1 m')
    return np.array([speed * math.cos(theta), speed * math.sin(theta), -(speed * math.sin(gamma) + p['lr'] * rate) / den, rate, np.clip((command[0] - speed) / tau_v, p['acc_min'], p['acc_max']), (command[1] - rate) / tau_w])

def rk4(state, command, dt, p, tau_v, tau_w):
    """采用经典四阶龙格-库塔法 (RK4) 对非线性状态微分方程进行离散推进。

    参数:
        state (np.ndarray): 初始状态向量，形状 (6,)。
        command (np.ndarray): 控制输入指令，形状 (2,)。
        dt (float): 推进步长 [单位: s]。
        p (dict): 车辆模型参数字典。
        tau_v (float): 车速执行器时间常数 [单位: s]。
        tau_w (float): 铰接角执行器时间常数 [单位: s]。

    返回:
        np.ndarray: 下一时步状态向量，形状 (6,)。
    """
    f = lambda z: plant_rhs(z, command, p, tau_v, tau_w)
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

def design_lqr(p, speed):
    """在名义稳态直行工况下设计无限时域离散 LQR，并求解终值黎卡提矩阵 P 与反馈增益 K。

    误差状态定义: [e_long, e_lat, e_theta, e_gamma, e_v, e_omega]^T。
    在平衡点 (v = speed, gamma = 0, omega = 0) 进行连续时间雅可比线性化与零阶离散化，
    通过求解离散代数黎卡提方程 (DARE) 获得稳态增益 K 与终值代价矩阵 P。

    参数:
        p (dict): 车辆几何与参数字典。
        speed (float): 设计巡航速度 [单位: m/s]。

    返回:
        tuple[np.ndarray, dict]:
            - K (np.ndarray): 离散 LQR 稳态反馈增益矩阵，形状 (2, 6)。
            - info (dict): 包含谱半径、黎卡提方程残差与数值雅可比校验结果的元数据字典。

    异常:
        RuntimeError: 闭环系统谱半径 >= 1 (不稳定) 或有限差分雅可比误差超限时抛出。
    """
    A = np.zeros((6, 6))
    B = np.zeros((6, 2))
    A[0, 4] = 1.0
    A[1, 2] = speed
    A[2, 3] = -speed / (p['lf'] + p['lr'])
    A[2, 5] = -p['lr'] / (p['lf'] + p['lr'])
    A[3, 5] = 1.0
    A[4, 4], A[5, 5] = (-1 / 0.2, -1 / 0.15)
    B[4, 0], B[5, 1] = (1 / 0.2, 1 / 0.15)
    block = np.zeros((8, 8))
    block[:6, :6], block[:6, 6:] = (A, B)
    disc = expm(block * CONTROL_DT)
    Ad, Bd = (disc[:6, :6], disc[:6, 6:])
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    residual = P - (Q + Ad.T @ P @ Ad - Ad.T @ P @ Bd @ K)
    radius = float(max(abs(np.linalg.eigvals(Ad - Bd @ K))))
    x, u, eps = (np.array([0.0, 0.0, 0.0, 0.0, speed, 0.0]), np.array([speed, 0.0]), 1e-05)
    numerical_A = np.column_stack([(plant_rhs(x + eps * np.eye(6)[j], u, p) - plant_rhs(x - eps * np.eye(6)[j], u, p)) / (2 * eps) for j in range(6)])
    numerical_B = np.column_stack([(plant_rhs(x, u + eps * np.eye(2)[j], p) - plant_rhs(x, u - eps * np.eye(2)[j], p)) / (2 * eps) for j in range(2)])
    jacobian_error = float(max(abs(numerical_A - A).max(), abs(numerical_B - B).max()))
    if not radius < 1 or jacobian_error > 1e-07:
        raise RuntimeError('LQR design stability/Jacobian check failed')
    return (K, {'kind': 'nominal_speed_terminal_cost_LQR_per_map', 'design_speed_mps': speed, 'design_tau_v_s': 0.2, 'design_tau_gamma_dot_s': 0.15, 'Q_diagonal': np.diag(Q).tolist(), 'R_diagonal': np.diag(R).tolist(), 'gain': K.tolist(), 'discrete_A': Ad.tolist(), 'discrete_B': Bd.tolist(), 'terminal_P': P.tolist(), 'closed_loop_spectral_radius': radius, 'riccati_max_abs_residual': float(abs(residual).max()), 'independent_finite_difference_jacobian_max_error': jacobian_error})

def load_parameters(input_root, mapname):
    """从配置快照中加载指定场景地图的车辆几何尺寸、物理边界与初始状态参数。

    参数:
        input_root (str | Path): 输入快照根目录路径。
        mapname (str): 地图场景标识符 (如 "B201", "B207", "B301")。

    返回:
        dict: 聚合后的车辆参数字典。
    """
    folder = Path(input_root) / mapname / 'cilqr'
    cfg = read_json(folder / 'effective_config.json')
    meta = read_json(folder / 'metadata.json')['initial_cpp_parameters']
    keys = ('lf', 'lr', 'gamma_min', 'gamma_max', 'gamma_dot_min', 'gamma_dot_max', 'acc_min', 'acc_max', 'trace_safe_width_left', 'trace_safe_width_right', 'obs_length', 'obs_width')
    p = {k: float(meta[k]) for k in keys}
    p.update({'v_min': -5.0, 'v_max': 30.0, 'width': float(cfg['main']['vehicle']['width']), 'box_length': float(cfg['main']['vehicle']['box_length']), 'initial_speed': MAPS[mapname], 'initial_gamma_dot': 0.0})
    return p

def original_euler(x, u, p, dt=0.1):
    """原版 C++ 模型离散前向欧拉运动学推演单步函数，用于回放一致性比对。

    参数:
        x (np.ndarray): 状态数组，形状 (N, 4) [x, y, 	heta, \\gamma]。
        u (np.ndarray): 控制输入数组，形状 (N, 2) [v, \\dot{\\gamma}]。
        p (dict): 包含前后轴距 lf, lr 的参数字典。
        dt (float): 欧拉推进步长 [单位: s]，默认 0.1 s。

    返回:
        np.ndarray: 欧拉推演后的下一时刻状态数组，形状 (N, 4)。
    """
    den = p['lr'] + p['lf'] * np.cos(x[:, 3])
    out = x.copy()
    out[:, 0] += dt * u[:, 0] * np.cos(x[:, 2])
    out[:, 1] += dt * u[:, 0] * np.sin(x[:, 2])
    out[:, 2] -= dt * (u[:, 0] * np.sin(x[:, 3]) + p['lr'] * u[:, 1]) / den
    out[:, 3] += dt * u[:, 1]
    return out

def violation(values, low, high):
    """计算数组元素超出指定区间 [low, high] 的绝对超限量。

    参数:
        values (float | np.ndarray): 待检验数值或数组。
        low (float): 下界限值。
        high (float): 上界限值。

    返回:
        np.ndarray: 非负超限量数组，位于区间内时为 0.0。
    """
    return np.maximum(np.maximum(low - np.asarray(values), np.asarray(values) - high), 0.0)

def load_frozen_reference(folder, p, output):
    """加载上层规划冻结基准轨迹，并执行完整的离线欧拉回放与物理约束核验。

    参数:
        folder (str | Path): 规划结果保存目录 (包含 trajectories.npz 与 cycles.jsonl)。
        p (dict): 车辆参数字典。
        output (str | Path): 导出的标准参考 CSV 路径。

    返回:
        tuple[np.ndarray, dict]:
            - reference (np.ndarray): 整合时间戳、状态与控制序列的数组，形状 (M, 7)。
            - audit (dict): 包含 SHA256 哈希、回放残差与约束违背统计的核验报告字典。

    异常:
        ValueError: 当存在空洞、非连续周期或欧拉回放残差 > 1e-8 时抛出。
    """
    folder = Path(folder)
    with np.load(folder / 'trajectories.npz', allow_pickle=False) as packed:
        states = packed['actual_states'].copy()
        controls_all = packed['controls'].copy()
    cycles = [json.loads(s) for s in (folder / 'cycles.jsonl').read_text(encoding='utf-8').splitlines() if s]
    executed = [(i, c['summary']) for i, c in enumerate(cycles) if c['summary']['executed']]
    controls = np.array([[c['velocity'], c['gamma_dot']] for _, c in executed])
    if len(states) != len(controls) + 1 or len(executed) != len(cycles):
        raise ValueError('This protocol requires contiguous, fully executed source cycles')
    np.testing.assert_allclose(controls, controls_all[[i for i, _ in executed], 0], atol=1e-10, rtol=1e-10)
    times = np.r_[0.0, [c['time_s'] for _, c in executed]]
    np.testing.assert_allclose(times, np.arange(len(states)) * 0.1, atol=1e-10, rtol=0)
    if not np.isfinite(states).all() or not np.isfinite(controls).all():
        raise ValueError('Nonfinite source reference; not replaced or filtered')
    err = original_euler(states[:-1], controls, p) - states[1:]
    err[:, 2] = wrap(err[:, 2])
    max_replay_error = float(abs(err).max())
    if max_replay_error > 1e-08:
        raise ValueError('Executed state/control replay disagrees with original model')
    states[:, 2] = np.unwrap(states[:, 2])
    reference = np.column_stack([times, states, np.vstack([controls, controls[-1]])])
    write_csv(output, [dict(zip(('t', 'x', 'y', 'theta', 'gamma', 'v', 'gamma_dot'), row)) for row in reference])
    dv = np.diff(controls[:, 0]) / 0.1
    audit = {'source_npz_sha256': sha(folder / 'trajectories.npz'), 'source_cycles_jsonl_sha256': sha(folder / 'cycles.jsonl'), 'source_metadata_sha256': sha(folder / 'metadata.json'), 'source_effective_config_sha256': sha(folder / 'effective_config.json'), 'reference_csv': str(Path(output).resolve()), 'reference_sha256': sha(output), 'state_rows': len(states), 'executed_controls': len(controls), 'duration_s': float(times[-1]), 'original_euler_replay_max_error': max_replay_error, 'reference_gamma_rate_violation_samples': int(np.count_nonzero(violation(controls[:, 1], p['gamma_dot_min'], p['gamma_dot_max']) > 1e-09)), 'reference_gamma_rate_max_violation_radps': float(violation(controls[:, 1], p['gamma_dot_min'], p['gamma_dot_max']).max()), 'reference_speed_slew_violation_samples': int(np.count_nonzero(violation(dv, p['acc_min'], p['acc_max']) > 1e-09)), 'reference_speed_slew_max_violation_mps2': float(violation(dv, p['acc_min'], p['acc_max']).max()), 'reference_articulation_max_violation_rad': float(violation(states[:, 3], p['gamma_min'], p['gamma_max']).max()), 'reference_speed_min_mps': float(controls[:, 0].min()), 'reference_speed_max_mps': float(controls[:, 0].max())}
    return (reference, audit)

def read_reference(path):
    """读取并解析参考轨迹 CSV 文件，对航向角进行连续相位展开 (unwrap)。

    参数:
        path (str | Path): 参考 CSV 文件路径。

    返回:
        np.ndarray: 解析后的参考矩阵，每行包含 [t, x, y, theta, gamma, v, gamma_dot]。
    """
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    ref = np.array([[float(row[k]) for k in ('t', 'x', 'y', 'theta', 'gamma', 'v', 'gamma_dot')] for row in rows])
    if len(ref) < 2 or not np.isfinite(ref).all() or np.any(np.diff(ref[:, 0]) <= 0):
        raise ValueError('Reference must be finite and have strictly increasing timestamps')
    ref[:, 0] -= ref[0, 0]
    ref[:, 3] = np.unwrap(ref[:, 3])
    return ref

def sample_reference(ref, time):
    """根据指定仿真时间线性插值采样参考位姿状态与前馈控制量。

    参数:
        ref (np.ndarray): 全局参考轨迹矩阵，形状 (M, 7)。
        time (float): 绝对仿真时间 [单位: s]。

    返回:
        tuple[np.ndarray, np.ndarray]:
            - z (np.ndarray): 线性插值后的位姿状态 [x, y, 	heta, \\gamma]^T，形状 (4,)。
            - u (np.ndarray): 当前区间的前馈控制指令 [v, \\dot{\\gamma}]^T，形状 (2,)。
    """
    k = min(max(int(np.searchsorted(ref[:, 0], time + 1e-10, side='right') - 1), 0), len(ref) - 2)
    z = np.array([np.interp(time, ref[:, 0], ref[:, j]) for j in range(1, 5)])
    return (z, ref[k, 5:7].copy())

def design_tvlqr(reference, p, nominal_design):
    """基于离散时间有限时域时变黎卡提差分方程，沿整条参考轨迹逆向推导时变反馈增益序列 K_k。

    局部误差坐标系定义于参考航向角的旋转坐标系下 (类似于 Frenet 误差框架)，包含旋转系科氏项。
    终端代价矩阵 P_N 继承自直行名义无限时域黎卡提方程的解。

    参数:
        reference (np.ndarray): 离线参考轨迹矩阵。
        p (dict): 车辆几何与物理参数字典。
        nominal_design (dict): 包含直行终值 P 矩阵的名义设计字典。

    返回:
        tuple[np.ndarray, dict]:
            - gains (np.ndarray): 时变 LQR 反馈增益三维张量序列，形状 (N_steps, 2, 6)。
            - meta (dict): 包含增益哈希与最小黎卡提特征值的元数据字典。

    异常:
        RuntimeError: 黎卡提差分迭代过程中矩阵正定性丧失 (min_eigen <= 0) 时抛出。
    """
    count = int(math.ceil(float(reference[-1, 0]) / CONTROL_DT - 1e-10))
    As, Bs = ([], [])
    for k in range(count):
        time = k * CONTROL_DT
        ref, u = sample_reference(reference, time)
        gamma, speed, rate = (ref[3], u[0], u[1])
        den = p['lr'] + p['lf'] * math.cos(gamma)
        num = speed * math.sin(gamma) + p['lr'] * rate
        j = min(max(int(np.searchsorted(reference[:, 0], time + 1e-10, side='right') - 1), 0), len(reference) - 2)
        omega = (reference[j + 1, 3] - reference[j, 3]) / (reference[j + 1, 0] - reference[j, 0])
        A, B = (np.zeros((6, 6)), np.zeros((6, 2)))
        A[0, 1], A[1, 0] = (omega, -omega)
        A[0, 4], A[1, 2] = (1.0, speed)
        A[2, 3] = -(speed * math.cos(gamma) * den + num * p['lf'] * math.sin(gamma)) / den ** 2
        A[2, 4], A[2, 5], A[3, 5] = (-math.sin(gamma) / den, -p['lr'] / den, 1.0)
        A[4, 4], A[5, 5] = (-1 / 0.2, -1 / 0.15)
        B[4, 0], B[5, 1] = (1 / 0.2, 1 / 0.15)
        block = np.zeros((8, 8))
        block[:6, :6], block[:6, 6:] = (A, B)
        local_dt = min(CONTROL_DT, float(reference[-1, 0]) - time)
        discrete = expm(block * local_dt)
        As.append(discrete[:6, :6])
        Bs.append(discrete[:6, 6:])
    P = np.asarray(nominal_design['terminal_P'], float)
    gains = np.empty((count, 2, 6))
    min_eigen = float(np.linalg.eigvalsh(P).min())
    for k in range(count - 1, -1, -1):
        A, B = (As[k], Bs[k])
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        gains[k] = K
        P = Q + A.T @ P @ A - A.T @ P @ B @ K
        P = (P + P.T) / 2
        min_eigen = min(min_eigen, float(np.linalg.eigvalsh(P).min()))
    if not np.isfinite(gains).all() or min_eigen <= 0:
        raise RuntimeError('Time-varying Riccati sequence invalid')
    return (gains, {'kind': 'finite_horizon_time_varying_LQR', 'control_steps': count, 'design_tau_v_s': 0.2, 'design_tau_gamma_dot_s': 0.15, 'Q_diagonal': np.diag(Q).tolist(), 'R_diagonal': np.diag(R).tolist(), 'terminal_cost': 'same-map nominal-speed infinite-horizon P', 'frame': 'reference-front-heading local coordinates, including rotating-frame terms', 'feedback_only': True, 'reference_defect_affine_feedforward_compensation': False, 'min_riccati_eigenvalue': min_eigen, 'gain_array_sha256': hashlib.sha256(gains.astype('<f8').tobytes()).hexdigest()})

def box(center, heading, length, width):
    """根据中心坐标、航向角与长宽尺寸构建平面定向包络矩形顶点。

    参数:
        center (np.ndarray): 矩形几何几何中心 [x, y]^T [单位: m]。
        heading (float): 局部纵向偏角 [单位: rad]。
        length (float): 矩形全长 [单位: m]。
        width (float): 矩形全宽 [单位: m]。

    返回:
        np.ndarray: 包含4个顶点坐标的数组，形状 (4, 2) [单位: m]。
    """
    e = direction(heading)
    n = np.array([-e[1], e[0]])
    return center + np.array([length * e / 2 + width * n / 2, length * e / 2 - width * n / 2, -length * e / 2 - width * n / 2, -length * e / 2 + width * n / 2])

def boxes_overlap(a, b):
    """利用分离轴定理 (SAT) 判定两个定向矩形是否发生碰撞相交。

    参数:
        a (np.ndarray): 矩形 A 顶点，形状 (4, 2)。
        b (np.ndarray): 矩形 B 顶点，形状 (4, 2)。

    返回:
        bool: True 表示发生重叠相交，False 表示两多边形分离安全。
    """
    for shape in (a, b):
        for edge in (shape[1] - shape[0], shape[2] - shape[1]):
            axis = np.array([-edge[1], edge[0]])
            axis /= np.linalg.norm(axis)
            pa, pb = (a @ axis, b @ axis)
            if pa.max() < pb.min() - 1e-10 or pb.max() < pa.min() - 1e-10:
                return False
    return True

class Geometry:
    """提供路径走廊包络与动态障碍物碰撞抽样评估的几何诊断工具类。"""

    def __init__(self, scene, params):
        """初始化几何评估环境。

        参数:
            scene (dict): 包含静态路径 path 与障碍物参数 initial_obstacles 的场景配置。
            params (dict): 车辆几何尺寸参数字典。
        """
        self.path = np.asarray(scene['path'], dtype=float)
        self.tree = cKDTree(self.path[:, :2])
        self.obs = np.asarray(scene['initial_obstacles'], dtype=float).reshape(-1, 4)
        self.p = params

    def evaluate(self, state, time):
        """评估当前时刻车辆状态相对于道路走廊边界的横向偏差及动态障碍物碰撞情况。

        参数:
            state (np.ndarray): 车辆状态向量，形状 (6,)。
            time (float): 当前绝对仿真时间 [单位: s]。

        返回:
            dict: 包含前后轴横向位置偏差、走廊违背量及碰撞标识的评估结果字典。
        """
        p = self.p
        rear = rear_axle(state, p)
        bodies = [box(state[:2], state[2], p['box_length'], p['width']), box(rear, state[2] + state[3], p['box_length'], p['width'])]
        points = np.vstack([state[:2], rear, *bodies])
        _, near = self.tree.query(points)
        matched = self.path[np.minimum(near + 1, len(self.path) - 1)]
        delta = points - matched[:, :2]
        lateral = -delta[:, 0] * np.sin(matched[:, 2]) + delta[:, 1] * np.cos(matched[:, 2])
        corridor = violation(lateral, -p['trace_safe_width_right'], p['trace_safe_width_left'])
        hits = 0
        for obstacle in self.obs:
            center = obstacle[:2] + time * obstacle[3] * direction(obstacle[2])
            ob = box(center, obstacle[2], p['obs_length'], p['obs_width'])
            hits += int(any((boxes_overlap(body, ob) for body in bodies)))
        return {'front_global_path_lateral_m': float(lateral[0]), 'rear_global_path_lateral_m': float(lateral[1]), 'front_corridor_violation_m': float(corridor[0]), 'rear_corridor_violation_m': float(corridor[1]), 'body_corner_corridor_violation_m': float(corridor[2:].max()), 'sampled_obstacle_collision': int(hits > 0)}

def simulate(reference, params, gain, case, output, geometry=None):
    """执行完整的轨迹跟踪闭环动态推演仿真，并生成高精度统计报告。

    在控制周期 (0.05 s) 下依据反馈增益计算控制指令，在对象推进步长 (0.01 s) 下利用 RK4 积分，
    并对执行机构的加速度限幅及控制量饱和进行逐步约束。

    参数:
        reference (np.ndarray): 全局参考轨迹矩阵。
        params (dict): 车辆几何与动力学限幅字典。
        gain (np.ndarray): 控制器反馈增益矩阵或增益张量序列。
        case (dict): 当前实验工况字典 (包含初始扰动量与滞后时间常数)。
        output (str | Path): 仿真结果 CSV 文件保存路径。
        geometry (Geometry, optional): 几何走廊与避障评估器。

    返回:
        dict: 聚合统计字典，包含各轴横向误差 RMSE、最大超调、物理约束违背量及输出哈希。
    """
    p = params
    first, ff = sample_reference(reference, 0.0)
    state = np.r_[first, p['initial_speed'], p['initial_gamma_dot']]
    state[:2] += case['lateral_offset_m'] * np.array([-math.sin(first[2]), math.cos(first[2])])
    state[2] += case['heading_offset_rad']
    state[3] += case['gamma_offset_rad']
    command = state[4:6].copy()
    raw_command = command.copy()
    previous_command = command.copy()
    duration = float(reference[-1, 0])
    n = int(round(duration / PLANT_DT))
    if abs(n * PLANT_DT - duration) > 1e-08:
        raise ValueError('Full reference duration must align with plant sampling')
    rows, reason = ([], 'fixed_reference_time_complete')
    speed_slew, saturated_v, saturated_w = (0.0, False, False)
    for i in range(n + 1):
        time = i * PLANT_DT
        ref, ff = sample_reference(reference, time)
        e = direction(ref[2])
        normal = np.array([-e[1], e[0]])
        delta = state[:2] - ref[:2]
        error = np.array([delta @ e, delta @ normal, wrap(state[2] - ref[2]), wrap(state[3] - ref[3]), state[4] - ff[0], state[5] - ff[1]])
        update = i % 5 == 0 and i < n
        if update:
            local_gain = gain[min(i // 5, len(gain) - 1)] if gain.ndim == 3 else gain
            raw_command = ff - local_gain @ error
            command = raw_command.copy()
            command[0] = np.clip(command[0], p['v_min'], p['v_max'])
            command[0] = np.clip(command[0], previous_command[0] + p['acc_min'] * CONTROL_DT, previous_command[0] + p['acc_max'] * CONTROL_DT)
            command[1] = np.clip(command[1], p['gamma_dot_min'], p['gamma_dot_max'])
            speed_slew = float((command[0] - previous_command[0]) / CONTROL_DT)
            saturated_v, saturated_w = (bool(abs(command[j] - raw_command[j]) > 1e-09) for j in range(2))
            previous_command = command.copy()
        rear_delta = rear_axle(state, p) - rear_axle(ref, p)
        rear_normal = np.array([-math.sin(ref[2] + ref[3]), math.cos(ref[2] + ref[3])])
        acc = float(np.clip((command[0] - state[4]) / case['tau_v_s'], p['acc_min'], p['acc_max']))
        row = dict(zip(('x', 'y', 'theta', 'gamma', 'v', 'gamma_dot'), map(float, state)))
        row = {'t': time, **row, **dict(zip(('ref_x', 'ref_y', 'ref_theta', 'ref_gamma'), map(float, ref))), 'ref_v': float(ff[0]), 'ref_gamma_dot': float(ff[1]), 'v_command': float(command[0]), 'gamma_dot_command': float(command[1]), 'v_command_raw': float(raw_command[0]), 'gamma_dot_command_raw': float(raw_command[1]), 'control_update': int(update), 'speed_command_saturated': int(saturated_v), 'gamma_rate_command_saturated': int(saturated_w), 'speed_command_slew_mps2': speed_slew, 'acceleration_mps2': acc, 'front_longitudinal_error_m': float(error[0]), 'front_lateral_error_m': float(error[1]), 'rear_lateral_error_m': float(rear_delta @ rear_normal), 'front_position_error_m': float(np.linalg.norm(delta)), 'rear_position_error_m': float(np.linalg.norm(rear_delta)), 'heading_error_rad': float(error[2]), 'rear_heading_error_rad': float(wrap(state[2] + state[3] - ref[2] - ref[3])), 'gamma_error_rad': float(error[3]), 'speed_error_mps': float(error[4]), 'gamma_rate_error_radps': float(error[5]), 'speed_violation_mps': float(violation(state[4], p['v_min'], p['v_max'])), 'gamma_violation_rad': float(violation(state[3], p['gamma_min'], p['gamma_max'])), 'gamma_rate_violation_radps': float(violation(state[5], p['gamma_dot_min'], p['gamma_dot_max'])), 'acceleration_violation_mps2': float(violation(acc, p['acc_min'], p['acc_max']))}
        if geometry is not None:
            row.update(geometry.evaluate(state, time))
        rows.append(row)
        if i == n:
            break
        try:
            candidate = rk4(state, command, PLANT_DT, p, case['tau_v_s'], case['tau_w_s'])
            if not np.isfinite(candidate).all():
                raise FloatingPointError('nonfinite independent plant state')
            state = candidate
        except FloatingPointError as exc:
            reason = str(exc)
            break
    write_csv(output, rows)
    result = {'case': case['name'], 'case_parameters': case, 'samples': len(rows), 'duration_s': rows[-1]['t'], 'reference_duration_s': duration, 'full_reference_completed': bool(len(rows) == n + 1), 'stop_reason': reason, 'finite': bool(all((np.isfinite(list(row.values())).all() for row in rows))), 'output_csv': str(Path(output).resolve()), 'output_sha256': sha(output)}
    errors = [k for k in rows[0] if '_error_' in k]
    for key in errors:
        values = np.array([row[key] for row in rows])
        base, unit = key.rsplit('_', 1)
        result[base + '_rmse_' + unit] = float(np.sqrt(np.mean(values ** 2)))
        result[base + '_max_abs_' + unit] = float(abs(values).max())
        result[base + '_final_abs_' + unit] = float(abs(values[-1]))
    for key in [k for k in rows[0] if '_violation_' in k]:
        values = np.array([row[key] for row in rows])
        result[key + '_max'] = float(values.max())
        result[key + '_samples'] = int(np.count_nonzero(values > 1e-09))
    for key in ('v', 'gamma', 'gamma_dot', 'v_command', 'gamma_dot_command', 'v_command_raw', 'gamma_dot_command_raw', 'speed_command_slew_mps2', 'acceleration_mps2'):
        values = np.array([row[key] for row in rows])
        result[key + '_min'] = float(values.min())
        result[key + '_max'] = float(values.max())
        result[key + '_max_abs'] = float(abs(values).max())
    for key in ('speed_command_saturated', 'gamma_rate_command_saturated'):
        result[key + '_updates'] = sum((row[key] for row in rows if row['control_update']))
    result['controller_updates'] = sum((row['control_update'] for row in rows))
    if geometry is not None:
        result['sampled_obstacle_collision_samples'] = sum((row['sampled_obstacle_collision'] for row in rows))
        result['continuous_collision_guarantee'] = False
    return result

def self_checks(p, gain):
    """执行几何包络相交算法、后轴无侧滑运动学导数与直行平衡态推演的自检断言。

    参数:
        p (dict): 车辆几何尺寸与动力学参数字典。
        gain (np.ndarray): 控制器增益。

    返回:
        dict: 包含后轴无侧滑残差与自检通过标识的字典。
    """
    a = box(np.array([0.0, 0.0]), 0.0, 5.0, 3.0)
    assert boxes_overlap(a, box(np.array([1.0, 0.0]), 0.2, 5.0, 3.0))
    assert not boxes_overlap(a, box(np.array([20.0, 0.0]), 0.2, 5.0, 3.0))
    s = np.array([2.0, 3.0, 0.4, 0.2, 5.0, 0.1])
    ds = plant_rhs(s, s[4:], p)
    h = 1e-06
    rv = (rear_axle(s + h * ds, p) - rear_axle(s - h * ds, p)) / (2 * h)
    rear_lateral = float(rv @ np.array([-math.sin(s[2] + s[3]), math.cos(s[2] + s[3])]))
    assert abs(rear_lateral) < 1e-07
    s = np.array([0.0, 0.0, 0.0, 0.0, p['initial_speed'], 0.0])
    for _ in range(100):
        s = rk4(s, np.array([p['initial_speed'], 0.0]), PLANT_DT, p, 0.2, 0.15)
    np.testing.assert_allclose(s, [p['initial_speed'], 0.0, 0.0, 0.0, p['initial_speed'], 0.0], atol=1e-10)
    return {'rear_no_slip_derivative_residual_mps': rear_lateral, 'straight_equilibrium_rk4_check': True, 'oriented_box_checks': True}

def main():
    """执行跨地图场景、多规划算法基准与多工况设置的轨迹跟踪评测主流程。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-root', type=Path, default=ROOT / '输入快照/full')
    ap.add_argument('--output-root', type=Path, default=ROOT)
    ap.add_argument('--maps', nargs='+', choices=list(MAPS), default=['B207'])
    ap.add_argument('--cases', nargs='+', choices=[c['name'] for c in CASES], default=['nominal'])
    ap.add_argument('--include-graph', action='store_true', help="Also track each selected map's existing independent graph reference")
    args = ap.parse_args()
    results, references, designs, checks = ([], {}, {}, {})
    graph_results = []
    chosen_cases = [c for c in CASES if c['name'] in args.cases]
    log = []
    for mapname in args.maps:
        speed = MAPS[mapname]
        p = load_parameters(args.input_root, mapname)
        scene_path = args.input_root / mapname / 'scene/scene.json'
        scene = read_json(scene_path)
        geometry = Geometry(scene, p)
        gain, design = design_lqr(p, speed)
        designs[mapname] = {'parameters': p, 'controller': design, 'scene_sha256': sha(scene_path)}
        checks[mapname] = self_checks(p, gain)
        for method in METHODS:
            folder = args.output_root / '原始数据/跟踪' / mapname / method
            ref, audit = load_frozen_reference(args.input_root / mapname / method, p, folder / 'reference.csv')
            references[mapname + '/' + method] = audit
            tvgain, tvdesign = design_tvlqr(ref, p, design)
            np.save(folder / 'controller_gains.npy', tvgain, allow_pickle=False)
            write_json(folder / 'controller_design.json', tvdesign)
            for case in chosen_cases:
                result = simulate(ref, p, tvgain, case, folder / (case['name'] + '.csv'), geometry)
                result.update({'map': mapname, 'method': method, 'reference_sha256': audit['reference_sha256']})
                results.append(result)
                write_json(folder / (case['name'] + '_summary.json'), result)
                log.append(f"{mapname}/{method}/{case['name']}: complete={result['full_reference_completed']} front_RMSE={result['front_lateral_error_rmse_m']:.6g} rear_RMSE={result['rear_lateral_error_rmse_m']:.6g} sampled_collisions={result['sampled_obstacle_collision_samples']}")
        if args.include_graph:
            graph_folder = args.output_root / '原始数据/图搜索' / mapname
            graph_refpath = graph_folder / 'reference.csv'
            graph_ref = read_reference(graph_refpath)
            graph_gain, graph_design = design_tvlqr(graph_ref, p, design)
            np.save(graph_folder / 'tracking_controller_gains.npy', graph_gain, allow_pickle=False)
            write_json(graph_folder / 'tracking_controller_design.json', graph_design)
            local_results = []
            for case in chosen_cases:
                result = simulate(graph_ref, p, graph_gain, case, graph_folder / ('tracking_' + case['name'] + '.csv'), geometry)
                result.update({'map': mapname, 'method': 'hybrid_astar', 'reference_csv': str(graph_refpath.resolve()), 'reference_sha256': sha(graph_refpath)})
                graph_results.append(result)
                local_results.append(result)
                log.append(f"{mapname}/hybrid_astar/{case['name']}: complete={result['full_reference_completed']} front_RMSE={result['front_lateral_error_rmse_m']:.6g} rear_RMSE={result['rear_lateral_error_rmse_m']:.6g} sampled_collisions={result['sampled_obstacle_collision_samples']}")
            write_json(graph_folder / 'tracking_summary.json', {'controller_design': graph_design, 'parameters': p, 'scope': 'separate_global_graph_reference_tracking_not_a_per_cycle_planning_time_comparison', 'results': local_results})
    protocol = {'scope': 'independent_fixed_full_reference_TVLQR_kinematic_tracking_with_assumed_actuator_lag', 'input_root': str(args.input_root.resolve()), 'script_sha256': sha(__file__), 'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'source_planning_horizon_N': 80, 'source_planning_dt_s': 0.1, 'plant_integrator': 'RK4', 'plant_dt_s': PLANT_DT, 'controller_dt_s': CONTROL_DT, 'maps': args.maps, 'cases': chosen_cases, 'designs': designs, 'formal_scope': 'User-selected B207 nominal case only by default; no other maps, perturbation cases, or preliminary controller results enter this summary.', 'controller': "Finite-horizon time-varying LQR along each entire frozen reference, in the rotating reference-heading frame. Same Q/R, nominal actuator lags and terminal-cost rule for every method and case. The terminal P is designed at the map's 5/6/15 m/s nominal speed. Gains are computed offline once per reference and reused for nominal/perturbed. No method-specific tuning or reference-defect feedforward correction.", 'controller_precheck': 'A straight fixed-gain controller precheck showed loss of tracking away from its design condition. It was replaced with the same TVLQR design rule for all methods. No comparison or best-result selection across controllers is used.', 'state_order': ['front_x', 'front_y', 'front_heading', 'articulation', 'speed', 'articulation_rate'], 'reference': 'Frozen actual_states plus actually executed first controls. Position/gamma and unwrapped heading linearly interpolated; controls zero-order held. Final control duplicated for CSV only. No replanning, resetting, smoothing, terminal extension, or delayed input injection.', 'initialization': 'All methods in a map start at the same map speed (5/6/15 m/s) and zero articulation rate; nominal has no additional pose offset. Nominal does not imply that initial actuator values equal the first reference input.', 'limits': "One map's original R initial_cpp_parameters supplies articulation/rate/acceleration and corridor limits to all methods. Speed command numerical envelope [-5,30] is original R's hard line-search envelope, not an identified physical actuator limit. Command speed slew and plant acceleration use [-4,3] m/s^2. Rate commands clipped to nominal metadata limits; gamma state is never clipped, and violations remain recorded.", 'errors': 'At each 0.01 s sample, actual minus fixed time-matched reference. Front lateral is projection on normal to reference front heading. Rear reference/actual position uses p_rear=p_front-lf*e(theta)-lr*e(theta+gamma), and rear lateral projects on reference rear normal. Position error is Euclidean axle distance. Angular differences wrapped to [-pi,pi]. RMSE uses all samples including endpoints without warm-up exclusion.', 'geometry': 'Sampled diagnostic only: front/rear axle-centered rectangles use original box_length and width; moving obstacle rectangles use scene initial poses, constant speeds and original obs_length/obs_width. SAT detects overlap. Corridor diagnostics use nearest frozen global path vertex plus one and its heading, original left/right half-widths, and axle centers/eight body corners. They are not raster road-boundary certificates. No collision/lane event terminates or filters a run.', 'completion': 'full_reference_completed means the full fixed time interval was simulated, not arrival, safe success, or closed-loop stability. Runs stop early only on nonfinite state/kinematic singularity and remain in all summaries.', 'limitations': ['No tire slip, tire forces, mass/inertia, or multibody dynamics.', 'Lag constants and initial perturbations are assumed scenarios, not measured vehicle parameters.', 'No online optimizer or planner wall-time delay is included.', 'No method-specific gains, filtering, source edits, or exclusion of bad references.']}
    summary_dir = args.output_root / '汇总结果'
    write_json(summary_dir / 'tracking_protocol.json', protocol)
    write_json(summary_dir / 'tracking_reference_audit.json', references)
    write_json(summary_dir / 'tracking_validation.json', checks)
    write_json(summary_dir / 'tracking_summary.json', {'protocol': str((summary_dir / 'tracking_protocol.json').resolve()), 'results': results})
    if graph_results:
        write_json(summary_dir / 'graph_tracking_summary.json', {'results': graph_results, 'comparison_scope': 'Independent complete global graph reference; different reference duration. No comparison of global search time with per-cycle optimizer time is implied.'})
    flat = [{k: v for k, v in result.items() if not isinstance(v, (dict, list))} for result in results]
    first = ['map', 'method', 'case', 'full_reference_completed', 'front_lateral_error_rmse_m', 'rear_lateral_error_rmse_m']
    write_csv(summary_dir / 'tracking_summary.csv', flat, first + [k for k in flat[0] if k not in first])
    (summary_dir / 'tracking_run.log').write_text('\n'.join(log) + '\n', encoding='utf-8')
    print(json.dumps({'runs': len(results), 'graph_runs': len(graph_results), 'full_reference_completed': sum((r['full_reference_completed'] for r in results)), 'summary': str((summary_dir / 'tracking_summary.csv').resolve())}, ensure_ascii=False))
if __name__ == '__main__':
    main()