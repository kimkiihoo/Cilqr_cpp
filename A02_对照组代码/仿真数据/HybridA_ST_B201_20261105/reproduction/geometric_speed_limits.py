"""基于铰接车辆运动学空间常微分方程的沿程几何限速与反馈转向控制模块。

数学推导与物理原理:
1. 空间铰接角常微分方程 (Spatial Articulation ODE):
   在非线性铰接车辆模型中，航向角速度满足:
       \\dot{\\theta} = -(v * sin(\\gamma) + l_r * \\dot{\\gamma}) / (l_r + l_f * cos(\\gamma))
   在光滑空间路径上，设定航向变化满足纯几何约束 \\dot{\\theta} = \\kappa(s) * v，
   且铰接角变化率满足 \\dot{\\gamma} = \\gamma'(s) * v (其中 \\gamma'(s) = d\\gamma / ds 为关于弧长的一阶空间导数)。
   当前向车速 v > 0 时，两边消去车速 v，得到不依赖时间基准的纯几何空间常微分方程:
       \\gamma'(s) = -(\\kappa(s) * (l_r + l_f * cos(\\gamma)) + sin(\\gamma)) / l_r
   通过初值问题 (IVP) 沿几何路径积分求解，即可获得整条路径各弧长处的理论铰接角 \\gamma(s) 与空间导数 \\gamma'(s)。

2. 物理铰接角速率界限反推沿程速度上限:
   实际转向液压机构存在物理角速度极限 [\\dot{\\gamma}_{min}, \\dot{\\gamma}_{max}]。
   根据 \\dot{\\gamma} = \\gamma'(s) * v，推导路径各点允许的最大行驶车速剖面:
       当 \\gamma'(s) > 0 时: v_{limit}(s) = min(v_{config}, \\dot{\\gamma}_{max} / \\gamma'(s))
       当 \\gamma'(s) < 0 时: v_{limit}(s) = min(v_{config}, \\dot{\\gamma}_{min} / \\gamma'(s))

3. 几何前馈 + 状态反馈逆解转向控制律:
   \\dot{\\psi}_{des} = v * (\\kappa(s) - k_{lat} * e_{lat}) - k_{head} * e_{head}
   \\dot{\\gamma} = -(\\dot{\\psi}_{des} * (l_r + l_f * cos(\\gamma)) + v * sin(\\gamma)) / l_r
"""
from __future__ import annotations
import math
import time
import numpy as np
from scipy.integrate import solve_ivp
from st_speed_search import path_arclength

def build_geometric_speed_profile(path, metadata, initial_gamma=0.0, sample_ds=0.02):
    """沿给定的离散几何路径求解空间铰接角微分方程，并推导满足物理角速度极限的空间限速曲线。

参数:
    path (np.ndarray): 离散几何路径，形状 (N, 3)，每行包含 [x, y, theta] [m, m, rad]。
    metadata (dict): 包含前后轴距 lf, lr [m]、铰接角范围 gamma_min, gamma_max [rad]、
        角速度界限 gamma_dot_min, gamma_dot_max [rad/s] 及标称配置车速 configured_speed [m/s] 的元数据字典。
    initial_gamma (float): 起始位置的初始铰接角 [单位: rad]，默认 0.0 rad。
    sample_ds (float): 求解器积分采样步长 [单位: m]，默认 0.02 m。

返回:
    dict: 包含稠密弧长坐标 s [m]、曲率 curvature [1/m]、理论铰接角 gamma [rad]、
        空间导数 gamma_prime [rad/m]、速度上限 speed_limit [m/s] 及可行性指标的字典。

异常:
    ValueError: 参数非有限或非物理有效时抛出。
    RuntimeError: 空间 ODE 数值积分未收敛时抛出。
"""
    started = time.perf_counter()
    path = np.asarray(path, dtype=float)
    arc = path_arclength(path)
    m = metadata
    required = ('lf', 'lr', 'gamma_min', 'gamma_max', 'gamma_dot_min', 'gamma_dot_max', 'configured_speed')
    if not all((math.isfinite(m[k]) for k in required)) or not math.isfinite(initial_gamma):
        raise ValueError('Vehicle parameters and initial articulation must be finite')
    if min(m['lf'], m['lr'], m['configured_speed'], sample_ds) <= 0:
        raise ValueError('Vehicle lengths, configured speed and sample_ds must be positive')
    if m['gamma_dot_min'] >= 0 or m['gamma_dot_max'] <= 0:
        raise ValueError('Articulation-rate bounds must straddle zero')
    heading = np.unwrap(path[:, 2])
    curvature = np.diff(heading) / np.diff(arc)
    dense = np.unique(np.r_[np.linspace(0.0, arc[-1], int(np.ceil(arc[-1] / sample_ds)) + 1), arc, np.maximum(0.0, arc[1:-1] - 1e-08), np.minimum(arc[-1], arc[1:-1] + 1e-08)])

    def curvature_at(s):
        index = np.clip(np.searchsorted(arc, s, side='right') - 1, 0, len(curvature) - 1)
        return curvature[index]

    def derivative(s, gamma):
        return -(curvature_at(s) * (m['lr'] + m['lf'] * np.cos(gamma)) + np.sin(gamma)) / m['lr']
    integrated = solve_ivp(derivative, (0.0, arc[-1]), [initial_gamma], t_eval=dense, max_step=sample_ds, rtol=1e-08, atol=1e-10)
    if not integrated.success or not np.isfinite(integrated.y).all():
        raise RuntimeError('Geometric articulation integration failed: ' + integrated.message)
    gamma = integrated.y[0]
    gamma_prime = derivative(dense, gamma)
    cap = np.full(len(dense), m['configured_speed'], dtype=float)
    positive, negative = (gamma_prime > 1e-12, gamma_prime < -1e-12)
    cap[positive] = np.minimum(cap[positive], m['gamma_dot_max'] / gamma_prime[positive])
    cap[negative] = np.minimum(cap[negative], m['gamma_dot_min'] / gamma_prime[negative])
    violation = np.maximum(gamma - m['gamma_max'], m['gamma_min'] - gamma)
    return {'s': dense, 'heading': np.interp(dense, arc, heading), 'curvature': curvature_at(dense), 'gamma': gamma, 'gamma_prime': gamma_prime, 'speed_limit': cap, 'speed_limits': np.column_stack((dense, cap)), 'geometric_articulation_feasible': bool(np.max(violation) <= 1e-08), 'max_articulation_violation_rad': float(max(0.0, np.max(violation))), 'minimum_speed_limit_mps': float(np.min(cap)), 'rate_only_travel_time_lower_bound_s': float(np.trapezoid(1.0 / cap, dense)), 'wall_ms': (time.perf_counter() - started) * 1000.0, 'definition': 'Original-model spatial articulation ODE; original effective rate bounds; no safety factor'}

def path_feedback_rate(state, speed, path, stations, profile, metadata, heading_gain=1.0, lateral_gain=0.05):
    """计算几何曲率前馈与航向/横向误差反馈融合的铰接角速度控制指令 \\dot{\\gamma}。

控制律架构:
- 航向与横向误差纠偏生成期望横摆角速度: \\dot{\\psi}_{des} = v * (\\kappa - k_{lat} * e_{lat}) - k_{head} * e_{head}
- 利用非线性铰接动力学解析逆解获得理论角速度指令: \\dot{\\gamma} = -(\\dot{\\psi}_{des} * (l_r + l_f * cos(\\gamma)) + v * sin(\\gamma)) / l_r
- 严格施加物理角速率硬限幅与控制周期步长内的铰接角防越界保护。

参数:
    state (np.ndarray): 当前车辆状态 [x, y, theta, gamma] [m, m, rad, rad]。
    speed (float): 当前纵向行驶速度 v [单位: m/s]。
    path (np.ndarray): 几何参考路径路点数组，形状 (N, 3)。
    stations (np.ndarray): 各参考路点累积弧长数组，形状 (N,) [单位: m]。
    profile (dict): 由 build_geometric_speed_profile 生成的空间剖面字典。
    metadata (dict): 车辆几何与控制周期元数据字典。
    heading_gain (float): 航向角误差反馈比例增益 k_head [单位: 1/s]，默认 1.0。
    lateral_gain (float): 横向位置误差反馈比例增益 k_lat [单位: 1/m]，默认 0.05。

返回:
    float: 饱和截断后的铰接角变化率控制指令 \\dot{\\gamma} [单位: rad/s]。
"""
    if speed <= 1e-12:
        return 0.0
    xy = np.asarray(state[:2], dtype=float)
    delta = path[1:, :2] - path[:-1, :2]
    fraction = np.clip(np.sum((xy - path[:-1, :2]) * delta, axis=1) / np.sum(delta * delta, axis=1), 0.0, 1.0)
    projection = path[:-1, :2] + fraction[:, None] * delta
    index = int(np.argmin(np.sum((xy - projection) ** 2, axis=1)))
    station = stations[index] + fraction[index] * (stations[index + 1] - stations[index])
    theta = float(np.interp(station, profile['s'], profile['heading']))
    curvature = float(np.interp(station, profile['s'], profile['curvature']))
    heading_error = math.atan2(math.sin(state[2] - theta), math.cos(state[2] - theta))
    error = xy - projection[index]
    lateral = -math.sin(theta) * error[0] + math.cos(theta) * error[1]
    yaw = speed * (curvature - lateral_gain * lateral) - heading_gain * heading_error
    m = metadata
    rate = -(yaw * (m['lr'] + m['lf'] * math.cos(state[3])) + speed * math.sin(state[3])) / m['lr']
    low = max(m['gamma_dot_min'], (m['gamma_min'] - state[3]) / m['dt'])
    high = min(m['gamma_dot_max'], (m['gamma_max'] - state[3]) / m['dt'])
    return float(np.clip(rate, low, high))