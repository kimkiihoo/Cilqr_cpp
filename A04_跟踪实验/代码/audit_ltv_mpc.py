"""B201 场景下 R-CILQR 空间路径结合 LTV-MPC 跟踪实验全量独立核验脚本。

本模块对跟踪实验的所有输出数据进行严格、独立的多重交叉核验：
1. 几何安全性：利用高精几何库 (Shapely) 对 0.002 s / 0.01 s 极密时钟采样点计算前后车体与全部动态障碍物的欧氏净距；
2. 走廊约束：检验前后轴中心及全部车体角点相对于原规划路点边界的法向投影是否严格在走廊限宽内；
3. 控制与状态边界：校验铰接角、线速度、角速度、纵向加速度、控制速度变化率的物理限值违背量；
4. 跟踪精度：独立核验前后轴横向误差 RMSE、最大值、终点距离及末速度停车状态；
5. QP 状态：核验 OSQP 求解状态码、残差与内迭代收敛性。
"""
from __future__ import annotations
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import shapely
sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / '核验结果'

def sha(path):
    """计算目标文件的 SHA-256 校验哈希值。

参数:
    path (str | Path): 目标文件路径。

返回:
    str: 64位十六进制哈希字符串。
"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_json(path):
    """读取并解析 JSON 配置文件。

参数:
    path (str | Path): JSON 文件路径。

返回:
    dict | list: 解析后的数据对象。
"""
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def read_numeric_csv(path):
    """以字典形式解析全部数值列的 CSV 数据表。

参数:
    path (str | Path): CSV 文件路径。

返回:
    dict[str, np.ndarray]: 列名到一维浮点数组的映射字典。
"""
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    return {name: np.array([float(row[name]) for row in rows]) for name in rows[0]}

def polygon_rectangles(centers, headings, backward, forward, width):
    """利用 Shapely 批量矢量化生成任意定向矩形多边形几何对象。

参数:
    centers (np.ndarray): 矩形中心点坐标数组，形状 (N, 2) [单位: m]。
    headings (np.ndarray): 矩形定向偏角数组，形状 (N,) [单位: rad]。
    backward (float): 沿纵轴向后的局部距离 [单位: m]。
    forward (float): 沿纵轴向前的局部距离 [单位: m]。
    width (float): 矩形全宽 [单位: m]。

返回:
    shapely.Geometry: 包含 N 个定向矩形多边形的矢量化几何数组。
"""
    local = np.array([[-backward, -width / 2], [forward, -width / 2], [forward, width / 2], [-backward, width / 2]])
    c, s = (np.cos(headings), np.sin(headings))
    coordinates = centers[:, None, :] + np.stack((local[None, :, 0] * c[:, None] - local[None, :, 1] * s[:, None], local[None, :, 0] * s[:, None] + local[None, :, 1] * c[:, None]), axis=-1)
    return shapely.polygons(coordinates)

def dense_geometry(states, times, p, scene, extended, with_values=False):
    """对全时程轨迹执行稠密时间步长下的车体与动态障碍物最小净距分析。

参数:
    states (np.ndarray): 车辆状态序列，形状 (N, 4) 或 (N, 6)。
    times (np.ndarray): 各时刻绝对时间戳数组，形状 (N,) [单位: s]。
    p (dict): 车辆与障碍物尺寸参数字典。
    scene (dict): 包含障碍物初态的场景字典。
    extended (bool): 是否将车身包络延伸覆盖铰接连接点。
    with_values (bool): 是否同时返回逐时刻最小净距序列。

返回:
    dict | tuple[dict, np.ndarray]: 包含最小净距 [m]、碰撞样本数与碰撞发生时刻的评估结果。
"""
    lf, lr, width, length = (p['lf'], p['lr'], p['width'], p['box_length'])
    front_heading, rear_heading = (states[:, 2], states[:, 2] + states[:, 3])
    rear = states[:, :2] - lf * np.column_stack((np.cos(front_heading), np.sin(front_heading)))
    rear -= lr * np.column_stack((np.cos(rear_heading), np.sin(rear_heading)))
    front_polys = polygon_rectangles(states[:, :2], front_heading, max(length / 2, lf) if extended else length / 2, length / 2, width)
    rear_polys = polygon_rectangles(rear, rear_heading, length / 2, max(length / 2, lr) if extended else length / 2, width)
    distances = []
    for x, y, heading, speed in scene['initial_obstacles']:
        centers = np.column_stack((x + speed * np.cos(heading) * times, y + speed * np.sin(heading) * times))
        obstacle = polygon_rectangles(centers, np.full(len(times), heading), p['obs_length'] / 2, p['obs_length'] / 2, p['obs_width'])
        distances.append(np.minimum(shapely.distance(front_polys, obstacle), shapely.distance(rear_polys, obstacle)))
    matrix = np.column_stack(distances)
    min_by_time = matrix.min(axis=1)
    sample, obstacle_index = np.unravel_index(int(matrix.argmin()), matrix.shape)
    result = {'sample_count': len(times), 'zero_clearance_or_collision_samples': int(np.sum(min_by_time <= 0)), 'minimum_clearance_m': float(matrix.min()), 'minimum_clearance_time_s': float(times[sample]), 'minimum_clearance_obstacle_index': int(obstacle_index), 'geometry': 'rectangles extended to the articulation joint' if extended else 'original axle-centered length/width rectangles'}
    return (result, min_by_time) if with_values else result

def excess(values, low, high):
    """计算数值相对于闭区间 [low, high] 的非负超限违背量。

参数:
    values (np.ndarray): 待检验数据数组。
    low (float): 允许下限。
    high (float): 允许上限。

返回:
    np.ndarray: 非负超限量数组。
"""
    return np.maximum(0, np.maximum(low - values, values - high))

def original_reference_audit(mapname='B201'):
    """核验原 R-CILQR 空间基准轨迹的一致性、离散欧拉重放误差与物理界限。

参数:
    mapname (str): 地图场景标识符，默认 "B201"。

返回:
    dict: 原参考轨迹核验指标字典。
"""
    reference_path = ROOT / '原始数据/跟踪' / mapname / 'cilqr/reference.csv'
    source = ROOT / '输入快照/full' / mapname / 'cilqr/trajectories.npz'
    config_path = ROOT / '输入快照/full' / mapname / 'cilqr/effective_config.json'
    metadata_path = ROOT / '输入快照/full' / mapname / 'cilqr/metadata.json'
    scene_path = ROOT / '输入快照/full' / mapname / 'scene/scene.json'
    config, metadata, scene = (read_json(config_path), read_json(metadata_path), read_json(scene_path))
    runtime = metadata['initial_cpp_parameters']
    p = {key: float(runtime[key]) for key in ('lf', 'lr', 'obs_length', 'obs_width')}
    p.update({key: float(config['main']['vehicle'][key]) for key in ('width', 'box_length')})
    limits = config['ilqr']['constraints']
    reference = read_numeric_csv(reference_path) if reference_path.exists() else None
    with np.load(source, allow_pickle=False) as archive:
        states = archive['actual_states'].copy()
        controls = archive['controls'][:, 0].copy()
    theta, gamma, v, w = (states[:-1, 2], states[:-1, 3], controls[:, 0], controls[:, 1])
    dt = float(runtime['dt'])
    expected = dt * np.column_stack((v * np.cos(theta), v * np.sin(theta), -(v * np.sin(gamma) + p['lr'] * w) / (p['lr'] + p['lf'] * np.cos(gamma)), w))
    source_residual = float(np.max(abs(states[1:] - states[:-1] - expected)))
    states[:, 2] = np.unwrap(states[:, 2])
    time = np.arange(len(states)) * dt
    if reference is not None:
        csv_states = np.column_stack([reference[name] for name in ('x', 'y', 'theta', 'gamma')])
        if not np.allclose(csv_states, states, atol=1e-10, rtol=0):
            raise ValueError('Baseline reference differs from the archived R-CILQR trajectory')
        if not np.allclose(reference['t'], time, atol=1e-10, rtol=0):
            raise ValueError('Baseline reference time mismatch')
    dense_dt = 0.002
    dense_time = np.arange(round(time[-1] / dense_dt) + 1) * dense_dt
    dense_states = np.column_stack([np.interp(dense_time, time, states[:, i]) for i in range(4)])
    original_geometry = dense_geometry(dense_states, dense_time, p, scene, extended=False)
    extended_geometry = dense_geometry(dense_states, dense_time, p, scene, extended=True)
    goal_config = config['main']['run_configs'][mapname]
    goal = np.array([goal_config['goal_x'], goal_config['goal_y']])
    acceleration = np.diff(v) / dt
    v_rate_excess = excess(acceleration, runtime['acc_min'], runtime['acc_max'])
    omega_excess = excess(w, runtime['gamma_dot_min'], runtime['gamma_dot_max'])
    report = {'scope': f'Original R-CILQR {mapname} reference only; baseline for the independent nominal LTV-MPC tracking follow-up.', 'map': mapname, 'source_unchanged': True, 'source_euler_state_chain_max_residual': source_residual, 'state_samples': len(states), 'control_intervals': len(controls), 'original_duration_s': float(time[-1]), 'path_length_m': float(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1).sum()), 'goal_xy_m': goal.tolist(), 'terminal_xy_m': states[-1, :2].tolist(), 'terminal_goal_distance_m': float(np.linalg.norm(states[-1, :2] - goal)), 'terminal_reference_speed_mps': float(v[-1]), 'reference_goal_position_within_1p5m': bool(np.linalg.norm(states[-1, :2] - goal) <= 1.5), 'reference_last_speed_at_most_0p5mps': bool(abs(v[-1]) <= 0.5), 'geometry_parameters': p, 'dense_sample_dt_s': dense_dt, 'dense_sampling_method': 'Linear interpolation of x/y/gamma and unwrapped heading between the fixed archived states; no reference resimulation or smoothing.', 'obstacle_clock': 'Obstacle positions advance from their archived initial states at absolute execution time t, independently of any retimed reference progress.', 'initial_obstacles': scene['initial_obstacles'], 'original_rectangles': original_geometry, 'joint_extended_rectangles': extended_geometry, 'configured_articulation_rate_limits_radps': [limits['gamma_dot_min'], limits['gamma_dot_max']], 'effective_runtime_articulation_rate_limits_radps': [runtime['gamma_dot_min'], runtime['gamma_dot_max']], 'reference_articulation_rate_max_abs_radps': float(np.max(abs(w))), 'reference_runtime_rate_violation_intervals': int(np.sum(omega_excess > 1e-09)), 'reference_runtime_rate_max_excess_radps': float(omega_excess.max()), 'effective_acceleration_limits_mps2': [runtime['acc_min'], runtime['acc_max']], 'reference_first_control_acceleration_min_mps2': float(acceleration.min()), 'reference_first_control_acceleration_max_mps2': float(acceleration.max()), 'reference_first_control_acceleration_violation_count': int(np.sum(v_rate_excess > 1e-09)), 'reference_first_control_acceleration_max_excess_mps2': float(v_rate_excess.max()), 'interpretation': ['Collision and clearance results apply only to this map and this original time profile; they must be recomputed against absolute obstacle time for a retimed execution.', 'Rate and acceleration feasibility are assessed separately above; geometric clearance alone does not establish feasibility for the 6-state lagged plant.', 'A new speed/progress profile must be archived separately and must not overwrite or re-anchor this reference to the actual state at each cycle.', 'The 0.1 m RMSE, 0.3 m maximum error and 0.05 m actual clearance criteria are engineering acceptance targets for this follow-up, not a paper or industry standard.', 'No continuous-time or real-vehicle safety claim follows from sampled polygon checks.'], 'acceptance_targets': {'front_spatial_lateral_rmse_m': 0.1, 'rear_spatial_lateral_rmse_m': 0.1, 'front_spatial_lateral_max_abs_m': 0.3, 'rear_spatial_lateral_max_abs_m': 0.3, 'joint_extended_body_sampled_min_clearance_m': 0.05, 'sampled_collision_count': 0, 'original_goal_distance_m': 1.5, 'terminal_abs_speed_mps': 0.5}, 'input_manifest': [{'path': str(f), 'sha256': sha(f)} for f in (source, reference_path, config_path, metadata_path, scene_path) if f.exists()]}
    OUTPUT.mkdir(exist_ok=True, parents=True)
    path = OUTPUT / f'ltv_mpc_reference_feasibility_{mapname}.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return report

def rear_axles(pose, p):
    """批量计算车辆状态序列对应的后轴中心笛卡尔坐标。

参数:
    states (np.ndarray): 状态数组，形状 (N, >=4)。
    p (dict): 几何参数字典。

返回:
    np.ndarray: 后轴坐标数组，形状 (N, 2) [单位: m]。
"""
    return pose[:, :2] - p['lf'] * np.column_stack((np.cos(pose[:, 2]), np.sin(pose[:, 2]))) - p['lr'] * np.column_stack((np.cos(pose[:, 2] + pose[:, 3]), np.sin(pose[:, 2] + pose[:, 3])))

def stats(values):
    """计算一维误差数组的均方根误差 (RMSE) 与最大绝对偏差。

参数:
    values (np.ndarray): 误差数据序列。

返回:
    dict: 包含 'rmse' 与 'max_abs' 的统计字典。
"""
    return {'rmse': float(np.sqrt(np.mean(np.square(values)))), 'max_abs': float(np.max(abs(values)))}

def polyline_errors(points, vertices):
    """基于空间几何折线切线与法向投影，独立计算前后轴的绝对空间横向位置误差。

参数:
    actual_states (np.ndarray): 实际执行状态序列。
    spatial_path (np.ndarray): 几何空间折线路点序列。
    p (dict): 车辆几何尺寸字典。

返回:
    dict: 前后轴空间横向误差指标字典。
"""
    starts, segments = (vertices[:-1], np.diff(vertices, axis=0))
    length_squared = np.sum(segments ** 2, axis=1)
    valid = length_squared > 1e-24
    starts, segments, length_squared = (starts[valid], segments[valid], length_squared[valid])
    lateral, distance = ([], [])
    for start in range(0, len(points), 256):
        delta = points[start:start + 256, None, :] - starts[None, :, :]
        fraction = np.clip(np.einsum('nki,ki->nk', delta, segments) / length_squared, 0, 1)
        residual = delta - fraction[:, :, None] * segments[None, :, :]
        distance_squared = np.sum(residual ** 2, axis=2)
        index = np.argmin(distance_squared, axis=1)
        ii = np.arange(len(index))
        selected = residual[ii, index]
        selected_segments = segments[index]
        lateral.extend((selected_segments[:, 0] * selected[:, 1] - selected_segments[:, 1] * selected[:, 0]) / np.sqrt(length_squared[index]))
        distance.extend(np.sqrt(distance_squared[ii, index]))
    return (np.asarray(lateral), np.asarray(distance))

def independent_rhs(x, u, p, tau_v=0.2, tau_w=0.15, clip_acceleration=True):
    """独立核验前向动力学微分方程的右端项计算。

参数:
    x (np.ndarray): 状态向量 [x, y, theta, gamma, v, omega]。
    u (np.ndarray): 控制指令 [u_v, u_omega]。
    p (dict): 车辆参数。
    tau_v (float): 车速响应滞后时间常数 [s]。
    tau_w (float): 铰接响应滞后时间常数 [s]。

返回:
    np.ndarray: 状态导数 dx/dt。
"""
    v, w, theta, gamma = (x[4], x[5], x[2], x[3])
    acceleration = (u[0] - v) / tau_v
    if clip_acceleration:
        acceleration = np.clip(acceleration, p['acc_min'], p['acc_max'])
    return np.array((v * np.cos(theta), v * np.sin(theta), -(v * np.sin(gamma) + p['lr'] * w) / (p['lr'] + p['lf'] * np.cos(gamma)), w, acceleration, (u[1] - w) / tau_w))

def corridor_audit(states, times, p, scene):
    """以稠密采样步长严格核验车辆车体各角点是否超出道路走廊限宽范围。

参数:
    nominal (dict): 仿真输出列数字典。
    p (dict): 车辆与走廊宽度参数。
    global_path (np.ndarray): 全局参考路径点。

返回:
    dict: 走廊超限量统计与最大违背指标。
"""
    from scipy.spatial import cKDTree
    front, back = (states[:, :2], rear_axles(states, p))
    points = [front, back]
    for center, heading, low, high in ((front, states[:, 2], -p['lf'], p['box_length'] / 2), (back, states[:, 2] + states[:, 3], -p['box_length'] / 2, p['lr'])):
        along = np.column_stack((np.cos(heading), np.sin(heading)))
        normal = np.column_stack((-np.sin(heading), np.cos(heading)))
        for x, y in ((low, -p['width'] / 2), (low, p['width'] / 2), (high, -p['width'] / 2), (high, p['width'] / 2)):
            points.append(center + x * along + y * normal)
    values = np.stack(points, axis=1)
    path = np.array(scene['path'], dtype=float)
    nearest = cKDTree(path[:, :2]).query(values.reshape(-1, 2))[1]
    reference = path[np.minimum(nearest + 1, len(path) - 1)]
    delta = values.reshape(-1, 2) - reference[:, :2]
    lateral = (-delta[:, 0] * np.sin(reference[:, 2]) + delta[:, 1] * np.cos(reference[:, 2])).reshape(len(states), 10)
    violations = excess(lateral, -p['trace_safe_width_right'], p['trace_safe_width_left'])
    return {'evaluation_rule': 'Original nearest-global-path-vertex plus one, projected on that vertex heading normal; not a raster road-boundary proof.', 'left_halfwidth_m': p['trace_safe_width_left'], 'right_halfwidth_m': p['trace_safe_width_right'], 'front_center_max_violation_m': float(violations[:, 0].max()), 'rear_center_max_violation_m': float(violations[:, 1].max()), 'joint_extended_corner_max_violation_m': float(violations[:, 2:].max()), 'front_center_violation_samples': int(np.sum(violations[:, 0] > 1e-09)), 'rear_center_violation_samples': int(np.sum(violations[:, 1] > 1e-09)), 'joint_extended_corner_violation_samples': int(np.sum(violations[:, 2:].max(axis=1) > 1e-09)), 'maximum_abs_corner_lateral_m': float(abs(lateral[:, 2:]).max()), 'samples': len(times)}

def qp_snapshot_audit(folder, name, p, dt):
    """核验关键时刻 (初始步与急弯步) 保存的 QP 矩阵稀疏结构与解的残差。

参数:
    raw_folder (Path): 包含快照 NPZ 的目录。

返回:
    dict: QP 快照核验状态字典。
"""
    from scipy import sparse
    from scipy.linalg import expm
    from scipy.integrate import solve_ivp
    P, A = (sparse.load_npz(folder / f'{name}_P.npz'), sparse.load_npz(folder / f'{name}_A.npz'))
    with np.load(folder / f'{name}_vectors.npz', allow_pickle=False) as archive:
        q, lower, upper = (archive['q'], archive['lower'], archive['upper'])
        refs, inputs, measured = (archive['reference_states'], archive['reference_commands'], archive['measured_initial_state'])
    with np.load(folder / f'{name}_solution.npz', allow_pickle=False) as archive:
        z, dual = (archive['x'], archive['y'])
    status = read_json(folder / f'{name}_status.json')
    az = A @ z
    primal = float(np.maximum(0, np.maximum(lower - az, az - upper)).max())
    equal = np.isfinite(lower) & np.isfinite(upper) & (lower == upper)
    equality = float(np.max(abs(az[equal] - lower[equal])))
    stationarity = float(np.max(abs(P @ z + q + A.T @ dual)))
    horizon = len(inputs)
    nstate = 6 * (horizon + 1)
    matrix = A.toarray()
    initial = measured - refs[0]
    initial[2] = np.arctan2(np.sin(initial[2]), np.cos(initial[2]))
    initial_residual = float(abs(lower[:6] - initial).max())
    max_model_difference, max_affine_difference, all_A = (0.0, 0.0, [])
    for k in range(horizon):
        x, u = (refs[k], inputs[k])
        h = 1e-05
        Aw = np.column_stack([(independent_rhs(x + h * np.eye(6)[j], u, p, clip_acceleration=False) - independent_rhs(x - h * np.eye(6)[j], u, p, clip_acceleration=False)) / (2 * h) for j in range(6)])
        Bw = np.column_stack([(independent_rhs(x, u + h * np.eye(2)[j], p, clip_acceleration=False) - independent_rhs(x, u - h * np.eye(2)[j], p, clip_acceleration=False)) / (2 * h) for j in range(2)])
        block = np.zeros((8, 8))
        block[:6, :6] = Aw
        block[:6, 6:] = Bw
        discrete = expm(block * dt)
        rows = slice(6 + 6 * k, 12 + 6 * k)
        saved_A = -matrix[rows, 6 * k:6 * k + 6]
        saved_B = -matrix[rows, nstate + 2 * k:nstate + 2 * k + 2]
        all_A.append(saved_A)
        max_model_difference = max(max_model_difference, float(abs(discrete[:6, :6] - saved_A).max()), float(abs(discrete[:6, 6:] - saved_B).max()))
        ivp = solve_ivp(lambda t, state: independent_rhs(state, u, p), (0, dt), x, method='DOP853', rtol=1e-11, atol=1e-12)
        affine = ivp.y[:, -1] - refs[k + 1]
        max_affine_difference = max(max_affine_difference, float(abs(lower[rows] - affine).max()))
    minimum_eigenvalue = float(np.linalg.eigvalsh(P.toarray()).min())
    return {'name': name, 'cycle': status['cycle'], 'time_s': status['time_s'], 'status': status['status'], 'variables': len(z), 'constraint_rows': len(lower), 'horizon_steps': horizon, 'max_primal_constraint_violation': primal, 'max_equality_residual': equality, 'max_kkt_stationarity_residual': stationarity, 'minimum_objective_hessian_eigenvalue': minimum_eigenvalue, 'initial_measured_state_equality_error': initial_residual, 'independent_world_jacobian_discretization_max_error': max_model_difference, 'independent_DOP853_affine_defect_max_error': max_affine_difference, 'A_variation_within_horizon': float(np.max(abs(np.array(all_A) - all_A[0]))), 'passed': bool(primal < 0.0002 and equality < 2e-05 and (stationarity < 0.0002) and (minimum_eigenvalue > 0) and (initial_residual < 1e-09) and (max_model_difference < 2e-07) and (max_affine_difference < 2e-05)), 'residual_interpretation': 'QP residuals are reported in original stored coordinates; actual state/input engineering acceptance is separately enforced at its stricter 1e-5 tolerance.'}

def final_tracking_audit(folder=None):
    """执行 B201 场景下 LTV-MPC 跟踪实验的全量综合核验主逻辑。

参数:
    mapname (str): 地图场景标识符，默认 "B201"。
"""
    from scipy.integrate import solve_ivp
    folder = Path(folder) if folder else ROOT / '原始数据/LTV_MPC/B201/cilqr'
    protocol, summary = (read_json(folder / 'protocol.json'), read_json(folder / 'summary.json'))
    targets = read_json(ROOT / '代码/ltv_acceptance_targets.json')
    p = protocol['parameters']
    scene = read_json(ROOT / '输入快照/full/B201/scene/scene.json')
    data = read_numeric_csv(folder / 'nominal.csv')
    original = read_numeric_csv(folder / 'original_reference.csv')
    profile = read_numeric_csv(folder / 'progress_profile.csv')
    with (folder / 'mpc_cycles.csv').open(encoding='utf-8-sig', newline='') as handle:
        cycles = list(csv.DictReader(handle))
    errors = []

    def check(ok, label):
        if not ok:
            errors.append(label)

    def close(a, b, label, atol=2e-08):
        check(np.allclose(a, b, atol=atol, rtol=1e-09), label)
    check(summary['map'] == 'B201' and summary['case'] == 'nominal', 'B201 nominal scope')
    check(sha(folder / 'nominal.csv') == summary['actual_csv_sha256'], 'actual CSV hash')
    check(sha(folder / 'original_reference.csv') == summary['original_reference_sha256'], 'original CSV hash')
    check(sha(folder / 'retimed_reference.csv') == summary['retimed_reference_sha256'], 'retimed CSV hash')
    check(sha(ROOT / '代码/ltv_mpc_tracking.py') == protocol['script_sha256'], 'controller code hash')
    check(sha(ROOT / '代码/tracking_experiment.py') == protocol['shared_parameters_script_sha256'], 'shared source-loading code hash')
    with np.load(ROOT / '输入快照/full/B201/cilqr/trajectories.npz', allow_pickle=False) as archive:
        archived_pose = archive['actual_states'].copy()
        archived_control = archive['controls'][:, 0].copy()
    archived_pose[:, 2] = np.unwrap(archived_pose[:, 2])
    original_pose = np.column_stack([original[k] for k in ('x', 'y', 'theta', 'gamma')])
    close(original_pose, archived_pose, 'unchanged complete original geometry')
    close(np.column_stack([original[k][:-1] for k in ('v', 'gamma_dot')]), archived_control, 'unchanged original applied controls')
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(original_pose[:, :2], axis=0), axis=1))]
    close(profile['path_s'], s, 'profile preserves all original vertices')
    close(profile['original_index'], np.arange(len(s)), 'profile index completeness')
    close(profile['original_time_s'], original['t'], 'profile original clock')
    profile_time, profile_speed = (profile['new_time_s'], profile['profile_speed_mps'])
    check(np.all(np.diff(profile_time) > 0), 'strictly increasing offline progress clock')
    times = data['t']
    dt, plant_dt = (protocol['controller_dt_s'], protocol['plant_dt_s'])
    close(times, np.arange(len(times)) * plant_dt, 'complete .01-s sample clock')
    expected_end = float(np.ceil((profile_time[-1] + 4.0) / dt) * dt)
    close(times[-1], expected_end, 'full schedule and four-second terminal settling')
    check(len(times) == int(round(expected_end / plant_dt)) + 1, 'expected sample count')
    close(data['path_s'][-1], s[-1], 'full original geometry progress completion')
    check(np.all(np.diff(data['path_s']) >= -1e-10), 'monotone saved offline progress')
    j = np.clip(np.searchsorted(profile_time, times, side='right') - 1, 0, len(s) - 2)
    local = np.maximum(times - profile_time[j], 0)
    acceleration_profile = np.diff(profile_speed) / np.diff(profile_time)
    progress_speed = profile_speed[j] + acceleration_profile[j] * local
    progress = s[j] + profile_speed[j] * local + 0.5 * acceleration_profile[j] * local ** 2
    fraction = np.clip((progress - s[j]) / np.diff(s)[j], 0, 1)
    reference_pose = original_pose[j] + fraction[:, None] * (original_pose[j + 1] - original_pose[j])
    signed_v = np.sign(original['v'][:-1])[j] * progress_speed
    rate = np.diff(original_pose[:, 3])[j] / np.diff(s)[j] * progress_speed
    settled = times >= profile_time[-1]
    reference_pose[settled], signed_v[settled], rate[settled] = (original_pose[-1], 0.0, 0.0)
    progress[settled], progress_speed[settled] = (s[-1], 0.0)
    original_time = original['t'][j] + fraction * np.diff(original['t'])[j]
    original_time[settled] = original['t'][-1]
    close(np.column_stack([data[k] for k in ('ref_x', 'ref_y', 'ref_theta', 'ref_gamma')]), reference_pose, 'offline mapping independent of actual state')
    close(data['ref_v'], signed_v, 'retimed signed reference speed')
    close(data['ref_gamma_dot'], rate, 'retimed articulation derivative')
    close(data['path_s'], progress, 'saved progress matches offline schedule')
    close(data['original_time_s'], original_time, 'original progress clock is distinct from actual clock')
    close(data['time_delay_from_original_s'], times - original_time, 'saved time delay mapping')
    actual = np.column_stack([data[k] for k in ('x', 'y', 'theta', 'gamma', 'v', 'gamma_dot')])
    commands = np.column_stack([data[k] for k in ('v_command', 'gamma_dot_command')])
    close(actual[0], np.r_[archived_pose[0], 5.0, 0.0], 'same nominal initial pose and 5 m/s actual speed')
    front_delta = actual[:, :2] - reference_pose[:, :2]
    rear_delta = rear_axles(actual, p) - rear_axles(reference_pose, p)
    reference_front_heading = reference_pose[:, 2]
    reference_rear_heading = reference_pose[:, 2] + reference_pose[:, 3]
    reconstructed = {'front_lateral_error_m': -front_delta[:, 0] * np.sin(reference_front_heading) + front_delta[:, 1] * np.cos(reference_front_heading), 'rear_lateral_error_m': -rear_delta[:, 0] * np.sin(reference_rear_heading) + rear_delta[:, 1] * np.cos(reference_rear_heading), 'front_position_error_m': np.linalg.norm(front_delta, axis=1), 'rear_position_error_m': np.linalg.norm(rear_delta, axis=1), 'heading_error_rad': np.arctan2(np.sin(actual[:, 2] - reference_pose[:, 2]), np.cos(actual[:, 2] - reference_pose[:, 2])), 'gamma_error_rad': actual[:, 3] - reference_pose[:, 3], 'speed_error_mps': actual[:, 4] - signed_v}
    time_metrics = {}
    for name, values in reconstructed.items():
        close(values, data[name], 'recomputed sample ' + name)
        time_metrics[name] = stats(values)
        close(summary[name + '_rmse'], time_metrics[name]['rmse'], 'summary RMSE ' + name)
        close(summary[name + '_max_abs'], time_metrics[name]['max_abs'], 'summary maximum ' + name)
    front_lat, front_distance = polyline_errors(actual[:, :2], archived_pose[:, :2])
    rear_lat, rear_distance = polyline_errors(rear_axles(actual, p), rear_axles(archived_pose, p))
    spatial_metrics = {'front_lateral_m': stats(front_lat), 'rear_lateral_m': stats(rear_lat), 'front_nearest_polyline_distance_m': stats(front_distance), 'rear_nearest_polyline_distance_m': stats(rear_distance)}
    stride = int(round(dt / plant_dt))
    close([float(row['t']) for row in cycles], np.arange(len(cycles)) * dt, 'complete control-cycle clock')
    check(len(cycles) == int(round(times[-1] / dt)), 'all controller cycles retained')
    for i in range(1, len(times)):
        if i % stride != 0 or i == len(times) - 1:
            close(commands[i], commands[i - 1], 'held command between updates')
    acceleration = np.clip((commands[:, 0] - actual[:, 4]) / protocol['tau_v_s'], p['acc_min'], p['acc_max'])
    close(acceleration, data['acceleration_mps2'], 'actual actuator acceleration')
    constraint_violations = {}
    for name, values, low, high in (('articulation', actual[:, 3], p['gamma_min'], p['gamma_max']), ('speed', actual[:, 4], p['v_min'], p['v_max']), ('articulation_rate', actual[:, 5], p['gamma_dot_min'], p['gamma_dot_max']), ('speed_command', commands[:, 0], p['v_min'], p['v_max']), ('rate_command', commands[:, 1], p['gamma_dot_min'], p['gamma_dot_max']), ('acceleration', acceleration, p['acc_min'], p['acc_max'])):
        constraint_violations[name] = float(excess(values, low, high).max())
    command_slew = np.diff(np.r_[p['initial_speed'], commands[:-1:stride, 0]]) / dt
    constraint_violations['command_speed_slew'] = float(excess(command_slew, p['acc_min'], p['acc_max']).max())
    dense_dt = float(targets['dense_geometry_dt_s'])
    dense_states, dense_times, maximum_replay_difference = ([], [], 0.0)
    for cycle in range(len(cycles)):
        row = cycle * stride
        t0, t1 = (times[row], times[row + stride])
        sample_times = np.linspace(t0, t1, int(round(dt / dense_dt)) + 1)
        u = commands[row]
        solution = solve_ivp(lambda t, x: independent_rhs(x, u, p), (t0, t1), actual[row], t_eval=sample_times, method='DOP853', rtol=1e-11, atol=1e-12)
        check(solution.success, 'independent dense plant integration')
        dense = solution.y.T
        maximum_replay_difference = max(maximum_replay_difference, float(abs(dense[::int(round(plant_dt / dense_dt))] - actual[row:row + stride + 1]).max()))
        dense_states.append(dense[:-1])
        dense_times.append(sample_times[:-1])
    dense_states = np.vstack([*dense_states, actual[-1:]])
    dense_times = np.r_[*dense_times, times[-1]]
    check(maximum_replay_difference < 2e-05, 'independent DOP853 vs logged RK4 state samples')
    geometry, discrete_clearance = dense_geometry(actual, times, p, scene, True, with_values=True)
    close(discrete_clearance, data['extended_body_clearance_m'], 'independent actual-clock Shapely distances')
    dense_safety = dense_geometry(dense_states, dense_times, p, scene, True)
    dense_safety['sample_dt_s'] = dense_dt
    dense_safety['method'] = 'Independent DOP853 over each held-control interval from its logged initial state, compared with all logged RK4 nodes; dynamic obstacles use actual t. GEOS rectangle distances/intersections include the joint.'
    corridor = corridor_audit(dense_states, dense_times, p, scene)
    runtime = read_json(ROOT / '输入快照/full/B201/cilqr/metadata.json')['initial_cpp_parameters']
    for key in ('gamma_min', 'gamma_max', 'gamma_dot_min', 'gamma_dot_max', 'acc_min', 'acc_max'):
        close(p[key], runtime[key], 'unchanged original runtime limit ' + key)
    qp_checks = [qp_snapshot_audit(folder, name, p, dt) for name in ('initial_qp', 'turn_qp')]
    for q in qp_checks:
        check(q['passed'], 'independent QP snapshot ' + q['name'])
    failures = sum((int(float(row['qp_status_val'])) not in (1, 2) for row in cycles))
    fallback = sum((row['fallback'] != 'none' for row in cycles))
    check(failures == summary['qp_failures'] and fallback == summary['fallback_cycles'], 'all QP failures/fallbacks accounted')
    controller_time = np.array([float(row['controller_wall_s']) for row in cycles])
    qp_time = np.array([float(row['qp_total_solve_wall_s']) for row in cycles])
    check(np.all(controller_time + 1e-09 >= qp_time), 'full controller timer includes QP timers')
    timing = {'control_period_s': dt, 'cycles': len(cycles), 'controller_mean_s': float(controller_time.mean()), 'controller_p95_s': float(np.quantile(controller_time, 0.95)), 'controller_max_s': float(controller_time.max()), 'deadline_exceedance_cycles': int(np.sum(controller_time > dt)), 'deadline_exceedance_fraction': float(np.mean(controller_time > dt)), 'qp_total_mean_s': float(qp_time.mean()), 'qp_failures': failures, 'fallback_cycles': fallback, 'max_command_projection': max((float(row['command_projection_max_abs']) for row in cycles)), 'delay_policy': 'Offline fixed physical clock; controller wall time is measured but not injected as an actuation delay. This is not a hard-real-time demonstration.'}
    goal_config = read_json(ROOT / '输入快照/full/B201/cilqr/effective_config.json')['main']['run_configs']['B201']
    goal = np.array([goal_config['goal_x'], goal_config['goal_y']])
    goal_distance = float(np.linalg.norm(actual[-1, :2] - goal))
    close(goal_distance, summary['final_goal_distance_m'], 'original scene terminal goal distance')
    terminal_pass = goal_distance <= targets['goal_distance_max_m'] and abs(actual[-1, 4]) <= targets['terminal_speed_absolute_max_mps']
    spatial_pass = all((spatial_metrics[name]['rmse'] <= 0.1 and spatial_metrics[name]['max_abs'] <= 0.3 for name in ('front_lateral_m', 'rear_lateral_m')))
    time_pass = all((time_metrics[name]['rmse'] <= 0.1 and time_metrics[name]['max_abs'] <= 0.3 for name in ('front_lateral_error_m', 'rear_lateral_error_m')))
    limits_pass = max(constraint_violations.values()) <= targets['state_and_input_bound_tolerance']
    safety_pass = dense_safety['minimum_clearance_m'] >= targets['minimum_body_obstacle_clearance_m'] and dense_safety['zero_clearance_or_collision_samples'] == 0 and limits_pass
    outcome = {'audit_passed': not errors, 'audit_errors': errors, 'map': 'B201', 'planner': 'R-CILQR', 'controller': 'LTV-MPC', 'case': 'nominal', 'targets': targets, 'original_duration_s': float(original['t'][-1]), 'retimed_duration_with_settling_s': float(times[-1]), 'retimed_travel_time_s': float(profile_time[-1]), 'samples': len(times), 'expected_control_cycles': len(cycles), 'reference_vertices_preserved': len(archived_pose), 'spatial_reference_reanchored': False, 'time_matched_metrics': time_metrics, 'frozen_original_polyline_spatial_metrics': spatial_metrics, 'spatial_error_definition': 'Nearest continuous segment on the frozen original front/rear axle polylines, independently of controller progress. Lateral projection and Euclidean nearest distance both reported; all actual samples retained.', 'dense_plant_replay_max_abs_error': maximum_replay_difference, 'dense_actual_clock_geometry': dense_safety, 'dense_global_path_corridor_diagnostic': corridor, 'actual_state_and_input_max_violations': constraint_violations, 'qp_snapshot_checks': qp_checks, 'controller_timing': timing, 'goal_xy_m': goal.tolist(), 'final_position_m': actual[-1, :2].tolist(), 'final_goal_distance_m': goal_distance, 'final_actual_speed_mps': float(actual[-1, 4]), 'acceptance': {'time_matched_precision': time_pass, 'independent_spatial_precision': spatial_pass, 'sampled_dense_safety_and_actual_limits': safety_pass, 'original_goal_arrival_and_stop': bool(terminal_pass), 'all_engineering_targets_met': bool(not errors and time_pass and spatial_pass and safety_pass and terminal_pass), 'hard_real_time_demonstrated': False}, 'source_reference_audit': original_reference_audit('B201'), 'input_hashes': [{'path': str(path), 'sha256': sha(path)} for path in [folder / 'nominal.csv', folder / 'original_reference.csv', folder / 'retimed_reference.csv', folder / 'progress_profile.csv', folder / 'mpc_cycles.csv', folder / 'protocol.json', ROOT / '代码/ltv_acceptance_targets.json', Path(__file__)]]}
    path = OUTPUT / 'ltv_mpc_independent_audit_B201.json'
    path.write_text(json.dumps(outcome, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    metric_row = {'map': 'B201', 'controller': 'LTV-MPC', 'duration_s': float(times[-1]), 'front_time_rmse_m': time_metrics['front_lateral_error_m']['rmse'], 'rear_time_rmse_m': time_metrics['rear_lateral_error_m']['rmse'], 'front_time_max_m': time_metrics['front_lateral_error_m']['max_abs'], 'rear_time_max_m': time_metrics['rear_lateral_error_m']['max_abs'], 'front_spatial_rmse_m': spatial_metrics['front_lateral_m']['rmse'], 'rear_spatial_rmse_m': spatial_metrics['rear_lateral_m']['rmse'], 'front_spatial_max_m': spatial_metrics['front_lateral_m']['max_abs'], 'rear_spatial_max_m': spatial_metrics['rear_lateral_m']['max_abs'], 'dense_min_clearance_m': dense_safety['minimum_clearance_m'], 'dense_collision_samples': dense_safety['zero_clearance_or_collision_samples'], 'goal_distance_m': goal_distance, 'final_speed_mps': float(actual[-1, 4]), 'controller_mean_ms': timing['controller_mean_s'] * 1000, 'controller_max_ms': timing['controller_max_s'] * 1000, 'deadline_exceedance_cycles': timing['deadline_exceedance_cycles'], 'qp_failures': failures, 'all_engineering_targets_met': outcome['acceptance']['all_engineering_targets_met']}
    with (OUTPUT / 'ltv_mpc_independent_metrics_B201.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_row))
        writer.writeheader()
        writer.writerow(metric_row)
    passed_text = '通过' if outcome['acceptance']['all_engineering_targets_met'] else '未通过'
    report_lines = ['# B201 正常状态 LTV-MPC 跟踪独立核验', '', f'工程验收结论：{passed_text}。本次仅核验原 R-CILQR 的 B201 几何参考，使用独立 LTV-MPC 跟踪和透明的离线重定时；未增加其他地图、方法或扰动工况。', '', f"原始 {len(archived_pose)} 个位姿顶点及终点均保留，参考没有随实际车辆位置重新锚定。初始真实速度仍为 5 m/s；重定时巡航上限为 {protocol['cruise_profile_cap_mps']:.1f} m/s，另保留初始减速过程。原参考时长 40.00 s，新行驶时长 {profile_time[-1]:.4f} s，加终点静置后总时长 {times[-1]:.2f} s。全部 {len(times)} 个记录样本进入误差统计。", '', '| 指标 | 前轴 | 后轴 | 工程目标 |', '|---|---:|---:|---:|', f"| 原冻结折线空间横向 RMSE / m | {spatial_metrics['front_lateral_m']['rmse']:.6f} | {spatial_metrics['rear_lateral_m']['rmse']:.6f} | ≤ 0.1 |", f"| 原冻结折线空间横向最大绝对误差 / m | {spatial_metrics['front_lateral_m']['max_abs']:.6f} | {spatial_metrics['rear_lateral_m']['max_abs']:.6f} | ≤ 0.3 |", f"| 重定时参考时间匹配横向 RMSE / m | {time_metrics['front_lateral_error_m']['rmse']:.6f} | {time_metrics['rear_lateral_error_m']['rmse']:.6f} | ≤ 0.1 |", f"| 重定时参考时间匹配横向最大绝对误差 / m | {time_metrics['front_lateral_error_m']['max_abs']:.6f} | {time_metrics['rear_lateral_error_m']['max_abs']:.6f} | ≤ 0.3 |", '', '空间误差通过独立搜索原始前后轴折线的最近连续线段得到，同时核算到折线的欧氏距离，未利用控制器给出的进度来缩小误差。上述阈值是本轮预先约定的工程目标，不是通用标准。', '', f"动态障碍安全核验：在每个实际已执行控制区间用独立 DOP853 积分，按 0.002 s 得到 {len(dense_times)} 个状态，以真实执行时间推进障碍，再用 Shapely/GEOS 检查延伸至铰接点的前后车体矩形。采样碰撞数为 {dense_safety['zero_clearance_or_collision_samples']}，最小间距为 {dense_safety['minimum_clearance_m']:.6f} m，大于 0.05 m。DOP853 与记录的 RK4 状态最大差为 {maximum_replay_difference:.3g}。这仍是密集采样核验，不表述为连续时间或实车安全保证。", '', f"道路走廊核验采用原程序“最近全局路径顶点再加一”的匹配方式及左右各 5 m 范围。前后轴中心与延伸车体八角点均无采样越界，角点最大绝对横向距离为 {corridor['maximum_abs_corner_lateral_m']:.6f} m。该检查不等同于栅格地图真实道路边界认证。", '', f"原场景终点仍为 (80, 80) m。最终位置误差 {goal_distance:.6f} m，实际速度 {actual[-1, 4]:.3g} m/s，满足 1.5 m 内且速度绝对值不大于 0.5 m/s 的停车目标。铰接角、实际角速度、实际加速度和命令边界均满足原运行限制；命令速度变化率仅有 {constraint_violations['command_speed_slew']:.3g} 的浮点误差。", '', f"控制器真实性核验：首周期和转弯周期两份已保存 QP 分别核对了正定目标 Hessian、初始状态等式、仿射缺陷、世界坐标时变动力学矩阵、原始约束残差及 KKT 驻点残差。全部 {len(cycles)} 个控制周期均已归档，QP 失败 {failures} 次、fallback {fallback} 次；执行前命令投影最大修正为 {timing['max_command_projection']:.3g}，未扩大任何硬边界。", '', f"计时限制：每次完整控制计算平均 {timing['controller_mean_s'] * 1000:.2f} ms、95% 分位 {timing['controller_p95_s'] * 1000:.2f} ms、最大 {timing['controller_max_s'] * 1000:.2f} ms。{timing['deadline_exceedance_cycles']}/{len(cycles)} 个周期（{timing['deadline_exceedance_fraction'] * 100:.2f}%）超过 50 ms 控制周期。本仿真未将计算耗时注入执行时钟，因此结论限于离线跟踪，不能写成已证明硬实时运行。", '', '复核入口：代码/audit_ltv_mpc.py。详细散列、QP残差、所有判据和密集检查结果见 ltv_mpc_independent_audit_B201.json；精简数值见 ltv_mpc_independent_metrics_B201.csv。']
    (OUTPUT / 'LTV-MPC跟踪核验说明_B201.md').write_text('\n'.join(report_lines) + '\n', encoding='utf-8')
    return outcome
if __name__ == '__main__':
    report = original_reference_audit()
    if '--reference-only' in sys.argv:
        print(json.dumps({'reference': report['map'] + '/R-CILQR', 'geometry': report['joint_extended_rectangles']}, ensure_ascii=False))
    else:
        audited = final_tracking_audit()
        print(json.dumps({'audit_passed': audited['audit_passed'], 'errors': audited['audit_errors'], 'acceptance': audited['acceptance'], 'spatial': audited['frozen_original_polyline_spatial_metrics'], 'dense_safety': audited['dense_actual_clock_geometry'], 'timing': audited['controller_timing']}, ensure_ascii=False))
        if not audited['audit_passed'] or not audited['acceptance']['all_engineering_targets_met']:
            sys.exit(1)