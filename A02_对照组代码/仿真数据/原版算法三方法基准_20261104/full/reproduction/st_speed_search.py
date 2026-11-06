"""基于时空状态图 (s, t, v) 与 A* 启发式搜索的动态避障速度规划器。

算法原理与实现架构:
1. 搜索状态空间 (3维状态):
   - s : 沿固定几何路径的累积弧长位置 (Arc Length) [单位: m]
   - t : 绝对规划时间坐标 (Time) [单位: s]，按步长 dt 离散化
   - v : 车辆纵向行驶线速度 (Speed) [单位: m/s]，按步长 dv 离散化

2. 状态转移与动力学限制:
   - 转移控制量为纵向加速度 a \\in [a_{min}, a_{max}] [单位: m/s^2]；
   - 下一时步速度: v' = v + a * dt，且需满足 0 <= v' <= min(v_{max}, v_{limit}(s))；
   - 时步内弧长增量: \\Delta s = 0.5 * (v + v') * dt；
   - 包含零速等待动作 (v = 0, v' = 0, \\Delta s = 0)，支持在遇到移动障碍阻挡时原位停车礼让。

3. 时变动态避障碰撞检验:
   - 沿状态转移区间以稠密微步长 collision_dt 对障碍物运动轨迹与自车椭圆安全缓冲区进行包络残差检验；
   - 动态障碍物按绝对时间沿固定航向以恒定速度匀速推演。

4. 启发式函数 (Heuristic):
   - 基于最大可行加速度与终点停车约束，估计从当前状态 (s, v) 到达终点 (S, 0) 所需的最小时间代价与加权距离代价。
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import heapq
import math
import time
import numpy as np

@dataclass(frozen=True)
class STSettings:
    """时空图 (s, t, v) A* 速度规划器的超参数配置数据类。"""
    dt: float = 0.5
    dv: float = 0.5
    max_time: float = 40.0
    v_max: float = 5.0
    a_min: float = -4.0
    a_max: float = 3.0
    initial_speed: float = 0.0
    goal_speed: float = 0.0
    collision_dt: float = 0.05
    collision_margin_m: float = 0.0
    max_expansions: int = 1000000
    max_seconds: float = 120.0

    def validate(self):
        for name, value in asdict(self).items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
        for name in ('dt', 'dv', 'max_time', 'v_max', 'collision_dt', 'max_seconds'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.a_min >= 0 or self.a_max <= 0:
            raise ValueError('a_min must be negative and a_max positive')
        if self.collision_dt > self.dt or self.collision_margin_m < 0:
            raise ValueError('collision_dt must not exceed dt; collision margin must be nonnegative')
        if self.max_expansions != int(self.max_expansions) or self.max_expansions < 1:
            raise ValueError('max_expansions must be a positive integer')
        for name in ('initial_speed', 'goal_speed'):
            value = getattr(self, name)
            if not 0 <= value <= self.v_max:
                raise ValueError(f'{name} must be between zero and v_max')
            if not math.isclose(value / self.dv, round(value / self.dv), abs_tol=1e-09):
                raise ValueError(f'{name} must lie on the speed grid')
        if self.goal_speed != 0:
            raise ValueError('The finite-window terminal hold requires zero goal_speed')
        if not math.isclose(self.max_time / self.dt, round(self.max_time / self.dt), abs_tol=1e-09):
            raise ValueError('max_time must be a multiple of dt')
        if self.dv / self.dt > min(-self.a_min, self.a_max) + 1e-12:
            raise ValueError('Speed grid is too coarse to accelerate and decelerate')

def path_arclength(path):
    """计算二维平面离散几何折线路点的累积欧几里得弧长序列。

参数:
    path (np.ndarray): 离散几何路径数组，形状 (N, 3)，每行包含 [x, y, theta] [单位: m, m, rad]。

返回:
    np.ndarray: 单调递增的累积弧长数组，形状 (N,) [单位: m]，起始值为 0.0。

异常:
    ValueError: 路径点少于2个或存在重复连续点时抛出。
"""
    values = np.asarray(path, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2 or (not np.isfinite(values).all()):
        raise ValueError('path must be a finite N x 3 array, N >= 2')
    distances = np.linalg.norm(np.diff(values[:, :2], axis=0), axis=1)
    if np.any(distances <= 1e-10):
        raise ValueError('Path must have distinct consecutive positions')
    return np.r_[0.0, np.cumsum(distances)]

def sample_path(path, s):
    """根据指定弧长位置在全局几何折线上线性插值获取二维坐标与连续展开航向角。

参数:
    path (np.ndarray): 全局几何路径数组，形状 (N, 3)。
    s (float | np.ndarray): 待采样的目标弧长位置 [单位: m]。

返回:
    np.ndarray: 插值得到的几何位姿数组 [x, y, theta]，形状 (..., 3) [单位: m, m, rad]。
"""
    values = np.asarray(path, dtype=float)
    arc = path_arclength(values)
    positions = np.asarray(s, dtype=float)
    if not np.isfinite(positions).all():
        raise ValueError('Arc positions must be finite')
    heading = np.interp(positions, arc, np.unwrap(values[:, 2]))
    return np.stack((np.interp(positions, arc, values[:, 0]), np.interp(positions, arc, values[:, 1]), np.arctan2(np.sin(heading), np.cos(heading))), axis=-1)

def obstacle_residual(positions, times, initial_obstacles, metadata, margin_m=0.0):
    """计算车辆包络点在指定绝对仿真时间下相对于环境中所有动态障碍物的椭圆代数距离残差。

椭圆残差定义:
    r = 1 - (ox / A)^2 - (oy / B)^2
    其中 (ox, oy) 为车辆相对于障碍物中心的旋转局部坐标，A, B 分别为包含车辆等效半径与安全缓冲区的长短半轴。
    当 r <= 0 时，车辆处于椭圆安全区域外 (安全无碰撞)；当 r > 0 时表示侵入障碍物安全区。

参数:
    positions (np.ndarray): 车辆二维位置坐标，形状 (..., 2) [单位: m]。
    times (np.ndarray): 对应的绝对物理仿真时间 [单位: s]。
    initial_obstacles (np.ndarray): 环境中所有动态障碍物的初始状态 [x, y, heading, speed]。
    metadata (dict): 包含障碍物长宽、自车半径及缓冲区大小的字典。
    margin_m (float): 额外的附加几何安全裕度 [单位: m]，默认 0.0 m。

返回:
    np.ndarray: 对应所有障碍物的最大残差值数组 (<= 0 表示安全)。
"""
    positions = np.asarray(positions, dtype=float)
    ts = np.asarray(times, dtype=float)
    initial = np.asarray(initial_obstacles, dtype=float).reshape(-1, 4)
    shape = np.broadcast_shapes(positions.shape[:-1], ts.shape)
    residual = np.full(shape, -np.inf)
    A = metadata['obs_length'] / 2 + metadata['ego_rad'] / 2 + metadata['safe_a_buffer'] + margin_m
    B = metadata['obs_width'] / 2 + metadata['ego_rad'] / 2 + metadata['safe_b_buffer'] + margin_m
    if not np.isfinite([A, B]).all() or min(A, B) <= 0:
        raise ValueError('Ellipse axes must be finite and positive')
    for ox, oy, heading, speed in initial:
        c, sn = (math.cos(heading), math.sin(heading))
        dx = positions[..., 0] - ox - ts * speed * c
        dy = positions[..., 1] - oy - ts * speed * sn
        residual = np.maximum(residual, 1.0 - ((c * dx + sn * dy) / A) ** 2 - ((-sn * dx + c * dy) / B) ** 2)
    return residual

def sample_speed_plan(plan, times):
    """在规划生成的离散速度曲线 (s, t, v, a) 中按给定时间连续插值当前状态。

参数:
    plan (dict): 由 plan_speed 规划输出的轨迹字典。
    t (float): 目标查询时间 [单位: s]。

返回:
    dict: 包含当前弧长 s [m]、车速 v [m/s] 与加速度 a [m/s^2] 的状态字典。
"""
    if not plan['success']:
        raise ValueError('A failed plan cannot be sampled')
    query = np.asarray(times, dtype=float)
    if not np.isfinite(query).all() or np.any(query < 0):
        raise ValueError('Requested times must be finite and nonnegative')
    if np.any(query > plan['settings']['max_time'] + 1e-09):
        raise ValueError('Requested time exceeds the checked planning window')
    knots = np.asarray(plan['times'], dtype=float)
    distance = np.asarray(plan['s'], dtype=float)
    velocity = np.asarray(plan['speeds'], dtype=float)
    if len(knots) == 1:
        return {'s': np.full(query.shape, distance[0]), 'speed': np.zeros(query.shape), 'acceleration': np.zeros(query.shape)}
    index = np.clip(np.searchsorted(knots, query, side='right') - 1, 0, len(knots) - 2)
    acceleration = (velocity[index + 1] - velocity[index]) / (knots[index + 1] - knots[index])
    elapsed = np.maximum(0.0, query - knots[index])
    distance_at_t = distance[index] + velocity[index] * elapsed + 0.5 * acceleration * elapsed ** 2
    speed_at_t = velocity[index] + acceleration * elapsed
    ended = query >= knots[-1] - 1e-12
    return {'s': np.where(ended, distance[-1], distance_at_t), 'speed': np.where(ended, velocity[-1], speed_at_t), 'acceleration': np.where(ended, 0.0, acceleration)}

def plan_speed(path, initial_obstacles, metadata, settings=None, speed_limits=None, allow_partial=False):
    """执行基于 (s, t, v) 时空图与 A* 优先队列搜索的全局动态避障速度规划。

参数:
    path (np.ndarray): 输入的第一阶段平滑几何参考路径，形状 (N, 3)。
    initial_obstacles (np.ndarray): 动态障碍物初始状态列表 [x, y, heading, speed]。
    metadata (dict): 车辆几何尺寸与安全边界元数据字典。
    settings (STSettings): 时空图网格分辨率、时域窗口与动力学限幅配置对象。
    speed_limits (np.ndarray, optional): 沿弧长的曲率几何限速边界数组，形状 (M, 2) [s, v_cap]。

返回:
    dict: 包含规划时间网格 t [s]、规划弧长轨迹 s [m]、期望速度序列 v [m/s]、
        加速度序列 a [m/s^2] 及搜索节点统计信息的规划结果字典。
"""
    started = time.perf_counter()
    cfg = settings or STSettings()
    cfg.validate()
    path = np.asarray(path, dtype=float)
    arc = path_arclength(path)
    obstacles = np.asarray(initial_obstacles, dtype=float)
    if obstacles.size == 0:
        obstacles = np.empty((0, 4))
    if obstacles.ndim != 2 or obstacles.shape[1] != 4 or (not np.isfinite(obstacles).all()):
        raise ValueError('initial_obstacles must be a finite M x 4 array')
    obstacle_residual(path[0], 0.0, obstacles, metadata, cfg.collision_margin_m)
    ds = cfg.dv * cfg.dt / 2.0
    speed_count = int(math.floor(cfg.v_max / cfg.dv + 1e-10)) + 1
    max_steps = int(round(cfg.max_time / cfg.dt))
    initial_index = int(round(cfg.initial_speed / cfg.dv))
    goal_index = int(math.floor(arc[-1] / ds + 1e-10))
    goal_index -= (goal_index - initial_index) % 2
    if goal_index < 0:
        raise ValueError('Path is shorter than the first reachable stopped station')
    s_grid = np.arange(goal_index + 1) * ds
    limits = None
    limit_min_table = None
    if speed_limits is not None:
        limits = np.column_stack(speed_limits) if isinstance(speed_limits, tuple) else np.asarray(speed_limits, dtype=float)
        if limits.ndim != 2 or limits.shape[1] != 2 or len(limits) < 2 or (not np.isfinite(limits).all()) or np.any(np.diff(limits[:, 0]) <= 0) or (limits[0, 0] > 0) or (limits[-1, 0] < arc[-1] - 1e-09) or np.any(limits[:, 1] < 0):
            raise ValueError('speed_limits must be finite ordered [s,v_max] rows spanning the path, with nonnegative speeds')
        if cfg.initial_speed > np.interp(0.0, limits[:, 0], limits[:, 1]) + 1e-10:
            raise ValueError('initial_speed exceeds the local path speed limit')
        levels = int(math.floor(math.log2(len(limits)))) + 1
        limit_min_table = np.full((levels, len(limits)), np.inf)
        limit_min_table[0] = limits[:, 1]
        for level in range(1, levels):
            width = 1 << level
            count = len(limits) - width + 1
            half = width // 2
            limit_min_table[level, :count] = np.minimum(limit_min_table[level - 1, :count], limit_min_table[level - 1, half:half + count])
    substeps = max(1, int(math.ceil(cfg.dt / cfg.collision_dt)))
    tau = np.linspace(0.0, cfg.dt, substeps + 1)
    geometry_cache = {}
    edge_cache = {}
    edge_collision_cache = {}
    holds = {}
    counts = {'expanded_nodes': 0, 'generated_nodes': 1, 'collision_rejected_edges': 0, 'speed_limit_rejected_edges': 0, 'cached_edge_masks': 0, 'peak_open_nodes': 1}
    transitions = {}
    for iv in range(speed_count):
        transitions[iv] = [j for j in range(speed_count) if cfg.a_min - 1e-10 <= (j - iv) * cfg.dv / cfg.dt <= cfg.a_max + 1e-10]

    def edge_free(kt, iv, jv):
        key = (kt, iv, jv)
        if key in edge_cache:
            return edge_cache[key]
        pair = (iv, jv)
        if pair not in geometry_cache:
            a = (jv - iv) * cfg.dv / cfg.dt
            offset = iv * cfg.dv * tau + 0.5 * a * tau ** 2
            s = s_grid[None, :] + offset[:, None]
            xy = np.stack((np.interp(s, arc, path[:, 0]), np.interp(s, arc, path[:, 1])), axis=-1)
            permitted = np.ones(len(s_grid), dtype=bool)
            if limits is not None:
                caps = np.interp(s, limits[:, 0], limits[:, 1])
                minimum_cap = np.minimum(caps[:-1], caps[1:])
                lo = np.searchsorted(limits[:, 0], s[:-1], side='left')
                hi = np.searchsorted(limits[:, 0], s[1:], side='right') - 1
                count = hi - lo + 1
                has_knots = count > 0
                level = np.floor(np.log2(np.maximum(count, 1))).astype(int)
                safe_lo = np.clip(lo, 0, len(limits) - 1)
                safe_hi = np.clip(hi - (1 << level) + 1, 0, len(limits) - 1)
                knot_minimum = np.minimum(limit_min_table[level, safe_lo], limit_min_table[level, safe_hi])
                minimum_cap = np.minimum(minimum_cap, np.where(has_knots, knot_minimum, np.inf))
                speed = iv * cfg.dv + a * tau
                maximum_speed = np.maximum(speed[:-1], speed[1:])[:, None]
                permitted = np.all(maximum_speed <= minimum_cap + 1e-10, axis=0)
            geometry_cache[pair] = (xy, permitted)
        residual = obstacle_residual(geometry_cache[pair][0], kt * cfg.dt + tau[:, None], obstacles, metadata, cfg.collision_margin_m)
        collision_free = np.all(residual <= 0.0, axis=0)
        valid = collision_free & geometry_cache[pair][1]
        valid &= np.arange(goal_index + 1) + iv + jv <= goal_index
        edge_cache[key] = valid
        edge_collision_cache[key] = collision_free
        counts['cached_edge_masks'] += 1
        return valid

    def terminal_hold_free(kt, si):
        key = (kt, si)
        if key not in holds:
            remaining = cfg.max_time - kt * cfg.dt
            n = max(1, int(math.ceil(remaining / cfg.collision_dt)))
            ts = np.linspace(kt * cfg.dt, cfg.max_time, n + 1)
            xy = np.array([np.interp(si * ds, arc, path[:, 0]), np.interp(si * ds, arc, path[:, 1])])
            holds[key] = bool(np.all(obstacle_residual(xy, ts, obstacles, metadata, cfg.collision_margin_m) <= 0.0))
        return holds[key]

    def finish(success, reason, end=None, reached_goal=False, optimal_progress_proven=False):
        sequence = []
        while end is not None:
            sequence.append(end)
            end = parent[end]
        sequence.reverse()
        counts['solver_wall_ms'] = (time.perf_counter() - started) * 1000.0
        counts['path_length_m'] = float(arc[-1])
        counts['s_resolution_m'] = ds
        counts['path_speed_limits_used'] = limits is not None
        counts['minimum_path_speed_limit_mps'] = float(np.min(limits[:, 1])) if limits is not None else None
        counts['collision_sample_dt_s'] = cfg.dt / substeps
        counts['search_objective'] = 'earliest feasible stopped arrival; otherwise maximum full-window progress when partial plans are allowed, on this finite lattice'
        result = {'success': success, 'reason': reason, 'settings': asdict(cfg), 'stats': counts, 'times': np.array([q[0] * cfg.dt for q in sequence]), 's': np.array([q[1] * ds for q in sequence]), 'speeds': np.array([q[2] * cfg.dv for q in sequence]), 'goal_distance_remaining_m': float(arc[-1] - sequence[-1][1] * ds) if success else None, 'reached_goal': reached_goal, 'optimal_progress_proven': optimal_progress_proven, 'arrival_time_s': sequence[-1][0] * cfg.dt if reached_goal else None, 'terminal_time_s': sequence[-1][0] * cfg.dt if success else None, 'checked_until_s': cfg.max_time if success else None}
        return result
    start = (0, 0, initial_index)
    parent = {start: None}
    if np.any(obstacle_residual(path[0], 0.0, obstacles, metadata, cfg.collision_margin_m) > 0.0):
        return finish(False, 'initial_position_blocked')
    queue = [(goal_index * ds / cfg.v_max, 0, 0, initial_index, start)]
    best_partial = None
    while queue:
        _, _, _, _, node = heapq.heappop(queue)
        kt, si, iv = node
        if si == goal_index and iv == 0 and terminal_hold_free(kt, si):
            return finish(True, 'goal_reached_and_terminal_hold_checked', node, reached_goal=True)
        if kt >= max_steps:
            if best_partial is None or (si, -iv) > (best_partial[1], -best_partial[2]):
                best_partial = node
            continue
        counts['expanded_nodes'] += 1
        if counts['expanded_nodes'] > cfg.max_expansions:
            if allow_partial and best_partial is not None:
                return finish(True, 'safe_partial_plan_expansion_limit', best_partial)
            return finish(False, 'expansion_limit')
        if counts['expanded_nodes'] % 256 == 0 and time.perf_counter() - started > cfg.max_seconds:
            if allow_partial and best_partial is not None:
                return finish(True, 'safe_partial_plan_time_limit', best_partial)
            return finish(False, 'time_limit')
        for jv in reversed(transitions[iv]):
            sj = si + iv + jv
            if sj > goal_index:
                continue
            nxt = (kt + 1, sj, jv)
            if nxt in parent:
                continue
            if not edge_free(kt, iv, jv)[si]:
                if not edge_collision_cache[kt, iv, jv][si]:
                    counts['collision_rejected_edges'] += 1
                else:
                    counts['speed_limit_rejected_edges'] += 1
                continue
            lower_bound = (kt + 1) * cfg.dt + (goal_index - sj) * ds / cfg.v_max
            if not allow_partial and lower_bound > cfg.max_time + 1e-10:
                continue
            parent[nxt] = node
            counts['generated_nodes'] += 1
            heapq.heappush(queue, (round(lower_bound, 12), -sj, kt + 1, -jv, nxt))
        counts['peak_open_nodes'] = max(counts['peak_open_nodes'], len(queue))
    if allow_partial and best_partial is not None:
        return finish(True, 'safe_partial_plan', best_partial, optimal_progress_proven=True)
    return finish(False, 'no_feasible_schedule_within_window')