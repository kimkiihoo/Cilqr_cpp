"""B201 场景下 Hybrid A* + ST* (时空图速度规划) 两阶段算法全流程闭环仿真运行主程序。

算法两阶段流程:
1. 第一阶段 (几何路径): 调用原生 Hybrid A* 与 QP 样条平滑算法生成空间离散路径；
2. 第二阶段 (时空速度规划): 基于 (s, t, v) 状态空间执行全局 A* 速度规划，满足加速度限幅与沿程几何限速；
3. 闭环控制推进: 采用几何前馈结合航向/横向反馈律计算铰接角速度，调用原模型动态链接库推进 40 s 状态。
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import time
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
import numpy as np
from original_backend import OriginalBackend
from hybrid_path_stage import prepare_path
from st_speed_search import STSettings, plan_speed, sample_speed_plan
from geometric_speed_limits import build_geometric_speed_profile, path_feedback_rate
HERE = Path(__file__).resolve().parent

def project_root():
    """获取当前项目根目录的绝对路径。"""
    for folder in HERE.parents:
        if (folder / 'AGENTS.md').is_file() and ((folder / 'A02_对照组代码').is_dir() or (folder / '03_实验数据').is_dir()):
            return folder
    raise FileNotFoundError('Cannot locate the B201 baseline; use a copy inside the project.')

def default_baseline():
    """获取默认的冻结基准测试数据根目录。"""
    root = project_root()
    for relative in ('A02_对照组代码/仿真数据/原版算法三方法基准_20261104/full', '03_实验数据/原版算法三方法基准_20261104/full'):
        candidate = root / relative
        if (candidate / 'B201/scene/scene.json').is_file():
            return candidate
    raise FileNotFoundError('Cannot locate the frozen B201 baseline')

def read_json(path):
    """读取并解析 JSON 文件。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    """保存格式化 JSON 数据。"""

    def default(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, Path):
            return str(v)
        raise TypeError(type(v).__name__)
    Path(path).write_text(json.dumps(value, default=default, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def write_csv(path, rows):
    """保存字典行数据为 CSV 表格。"""
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def sha(path):
    """计算文件 SHA-256 哈希字符串。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def path_stations(path):
    """计算离散几何路径的累积弧长数组 s [单位: m]。"""
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]

def sample_profile(result, times):
    """在速度剖面中线性插值给定时间对应的弧长进度 s 与目标速度 v。"""
    sampled = sample_speed_plan(result, times)
    return (sampled['s'], sampled['speed'])

def obstacle_residual(positions, times, initial, meta):
    """计算当前位姿下所有动态障碍物的椭圆碰撞残差 (<= 0 表示安全)。"""
    positions, times = (np.asarray(positions), np.asarray(times))
    if not len(initial):
        return np.full(len(times), -np.inf)
    centers = initial[None, :, :2] + times[:, None, None] * (initial[:, 3, None] * np.c_[np.cos(initial[:, 2]), np.sin(initial[:, 2])])[None]
    delta = positions[:, None, :2] - centers
    c, sn = (np.cos(initial[:, 2]), np.sin(initial[:, 2]))
    longitudinal = delta[:, :, 0] * c + delta[:, :, 1] * sn
    lateral = -delta[:, :, 0] * sn + delta[:, :, 1] * c
    a = meta['obs_length'] / 2 + meta['ego_rad'] / 2 + meta['safe_a_buffer']
    b = meta['obs_width'] / 2 + meta['ego_rad'] / 2 + meta['safe_b_buffer']
    return np.max(1 - (longitudinal / a) ** 2 - (lateral / b) ** 2, axis=1)

def pursuit_rate(state, speed, path, stations, meta, settings):
    """根据当前跟踪误差与几何曲率计算铰接角速度控制指令 \\dot{\\gamma} [单位: rad/s]。"""
    if speed < 1e-10:
        return 0.0
    nearest = int(np.argmin(np.linalg.norm(path[:, :2] - state[:2], axis=1)))
    lookahead = np.clip(settings['kv'] * max(1.0, speed) + settings['ld0'], settings['ld_min'], settings['ld_max'])
    target = min(int(np.searchsorted(stations, stations[nearest] + lookahead)), len(path) - 1)
    delta = path[target, :2] - state[:2]
    error = math.atan2(delta[1], delta[0]) - state[2]
    error = math.atan2(math.sin(error), math.cos(error))
    rate = -(settings['kp'] * error * (meta['lr'] + meta['lf'] * math.cos(state[3])) + speed * math.sin(state[3])) / meta['lr']
    lower = max(0.8 * meta['gamma_dot_min'], (meta['gamma_min'] - state[3]) / meta['dt'])
    upper = min(0.8 * meta['gamma_dot_max'], (meta['gamma_max'] - state[3]) / meta['dt'])
    return float(np.clip(rate, lower, upper))

def execution_metrics(states, controls, path, initial, meta, times):
    """计算实际执行轨迹的终点距离、横向偏差 RMSE 与约束违背指标。"""
    nearest = np.argmin(np.linalg.norm(states[:, None, :2] - path[None, :, :2], axis=2), axis=1)
    refs = path[np.minimum(nearest + 1, len(path) - 1)]
    delta = states[:, :2] - refs[:, :2]
    lateral = -delta[:, 0] * np.sin(refs[:, 2]) + delta[:, 1] * np.cos(refs[:, 2])
    lane = np.maximum(lateral - meta['trace_safe_width_left'], -lateral - meta['trace_safe_width_right'])
    obs = obstacle_residual(states[:, :2], times, initial, meta)
    acceleration = np.diff(controls[:, 0]) / meta['dt']
    return {'executed_frames': len(controls), 'final_goal_distance_m': float(np.linalg.norm(states[-1, :2] - path[-1, :2])), 'max_abs_lateral_error_m': float(np.max(np.abs(lateral))), 'max_obstacle_residual': float(np.max(obs)) if len(initial) else None, 'obstacle_violation_samples': int(np.count_nonzero(obs > 1e-09)), 'max_lane_violation_m': float(max(0.0, np.max(lane))), 'max_articulation_violation_rad': float(max(0.0, np.max(states[:, 3] - meta['gamma_max']), np.max(meta['gamma_min'] - states[:, 3]))), 'max_rate_violation_radps': float(max(0.0, np.max(controls[:, 1] - meta['gamma_dot_max']), np.max(meta['gamma_dot_min'] - controls[:, 1]))), 'max_internal_acceleration_violation_mps2': float(max(0.0, np.max(acceleration - meta['acc_max'], initial=0.0), np.max(meta['acc_min'] - acceleration, initial=0.0)))}

def benchmark_rows(baseline, scene, meta, new_summary):
    """生成与原三方法基准格式完全对齐的周期状态与控制量对比数据行。"""
    rows = []
    path = np.asarray(scene['path'], float)
    initial = np.asarray(scene['initial_obstacles'], float)
    for folder, label in (('cilqr', 'R-CILQR'), ('alilqr', 'AL-iLQR'), ('sqp', 'SQP-NMPC (original R penalty)')):
        source = baseline / 'B201' / folder
        with (source / 'cycles.csv').open(encoding='utf-8-sig', newline='') as stream:
            records = list(csv.DictReader(stream))
        z = np.load(source / 'trajectories.npz')
        states = z['actual_states']
        controls = np.array([[float(r['velocity']), float(r['gamma_dot'])] for r in records])
        times = np.arange(len(states)) * meta['dt']
        timing = np.array([float(r['solver_wall_ms']) for r in records])
        metrics = execution_metrics(states, controls, path, initial, meta, times)
        rows.append({'map': 'B201', 'method': label, 'planning_mode': 'receding_horizon_8s', 'planning_calls': len(records), 'planning_total_ms': float(timing.sum()), 'planning_mean_call_ms': float(timing.mean()), 'geometry_preparation_ms': None, **metrics})
    rows.append({'map': 'B201', 'method': 'Hybrid A* + ST*', 'planning_mode': 'global_speed_graph_40s', 'planning_calls': 1, 'planning_total_ms': new_summary['st_search_wall_ms'], 'planning_mean_call_ms': None, 'geometry_preparation_ms': new_summary['path_stage_wall_ms'], **new_summary['execution']})
    return rows

def main():
    """B201 场景 Hybrid A* + ST* 两阶段算法仿真推进与数据记录主函数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=default_baseline())
    parser.add_argument('--output', type=Path, required=True, help='New output directory; existing directories are refused')
    parser.add_argument('--reuse-path', action='store_true', help='Reuse the frozen path; no path-generation time is claimed')
    args = parser.parse_args()
    baseline, output = (args.baseline.resolve(), args.output.resolve())
    scene = read_json(baseline / 'B201/scene/scene.json')
    configs = read_json(baseline / 'B201/cilqr/effective_config.json')
    configs = {key: configs[key] for key in ('main', 'ilqr', 'hybrid_astar')}
    if scene['map'] != 'B201' or configs['main']['selected_map'] != 'B201':
        raise ValueError('This experiment only supports B201')
    if output.exists():
        raise FileExistsError('Refusing to overwrite existing results: ' + str(output))
    output.mkdir(parents=True)
    source_dir = output / 'reproduction'
    source_dir.mkdir()
    source_names = ('run_hybrid_astar_st.py', 'hybrid_path_stage.py', 'st_speed_search.py', 'geometric_speed_limits.py', 'original_backend.py', 'project_paths.py', 'test_st_speed_search.py', 'audit_hybrid_st_b201.py', 'plot_hybrid_st_b201.py')
    for name in source_names:
        shutil.copy2(HERE / name, source_dir / name)
    write_json(output / 'scene.json', scene)
    write_json(output / 'effective_config.json', configs)
    print('B201: preparing Hybrid A* + smoothing path', flush=True)
    path_result = prepare_path(baseline, output, regenerate=not args.reuse_path)
    path = np.asarray(path_result['path'], float)
    write_json(output / 'path_stage.json', path_result['path_stage'])
    dll = baseline / 'reproduction/build/original_model_bridge.dll'
    os.environ['ORIGINAL_MODEL_BRIDGE_DLL'] = str(dll)
    with OriginalBackend(path, configs) as backend:
        meta = backend.metadata
        if meta['N'] != 80 or meta['dt'] != 0.1 or meta['configured_speed'] != 5:
            raise ValueError('Expected frozen B201 v5/N80/dt0.1 settings')
        settings = STSettings(dt=0.5, dv=0.25, max_time=40.0, v_max=meta['configured_speed'], a_min=meta['acc_min'], a_max=meta['acc_max'], initial_speed=0.0, goal_speed=0.0, collision_dt=0.05, max_expansions=1000000)
        initial = np.asarray(scene['initial_obstacles'], float).reshape(-1, 4)
        protocol = {'map': 'B201', 'method': 'Hybrid A* + ST*', 'method_definition': 'Hybrid A* and existing path smoothing, followed by independent space-time graph speed search', 'baseline_directory': str(baseline), 'episode_s': 40.0, 'execution_dt_s': 0.1, 'planning_mode': 'one global ST search with constant-velocity obstacle predictions over 40 s', 'stage_timing': 'Path preparation includes process startup and exports; ST timing includes graph preparation/search. Follower and diagnostics are separate.', 'comparison_timing': 'Do not compare one global search directly with one 8 s horizon MPC call. Existing methods are read from their archived runs.', 'geometry': 'Original reference-point inflated ellipse and nearest+1 lane residual; not a whole-body collision claim', 'initial_speed_mps': 0.0, 'initial_speed_note': 'Original four-state model has no velocity state; ST adds an explicit zero initial graph speed.', 'path_speed_limits': 'Original-model spatial articulation ODE; speed limited by effective articulation-rate bounds', 'follower': {'type': 'path curvature feedforward, heading/lateral feedback, original yaw equation inversion', 'heading_gain': 1.0, 'lateral_gain': 0.05, 'rate_limits': 'original effective rate limits and next-step articulation bound; zero rate when stopped'}, 'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform(), 'sources': {name: sha(source_dir / name) for name in source_names}, 'input_hashes': {'scene': sha(baseline / 'B201/scene/scene.json'), 'config': sha(baseline / 'B201/cilqr/effective_config.json'), 'original_model_dll': sha(dll)}, 'model_parameters': meta}
        write_json(output / 'protocol.json', protocol)
        print('B201: searching independent ST speed graph', flush=True)
        started = time.perf_counter()
        geometric_profile = build_geometric_speed_profile(path, meta, initial_gamma=scene['start'][3])
        if not geometric_profile['geometric_articulation_feasible']:
            write_json(output / 'summary.json', {'planning_success': False, 'reason': 'geometric_path_articulation_infeasible'})
            raise SystemExit('The frozen path exceeds the original articulation-angle limits')
        result = plan_speed(path, initial, meta, settings, speed_limits=geometric_profile['speed_limits'], allow_partial=True)
        search_ms = (time.perf_counter() - started) * 1000.0
        np.savez_compressed(output / 'geometric_speed_profile.npz', **{key: value for key, value in geometric_profile.items() if isinstance(value, np.ndarray)})
        write_json(output / 'geometric_speed_profile.json', {key: value for key, value in geometric_profile.items() if not isinstance(value, np.ndarray)})
        write_json(output / 'st_search.json', result)
        if not result['success']:
            write_json(output / 'summary.json', {'planning_success': False, 'reason': result['reason'], 'st_search_wall_ms': search_ms, 'execution': None})
            raise SystemExit('ST search failed: ' + result['reason'])
        times = np.arange(401) * meta['dt']
        station, instantaneous_speed = sample_profile(result, times)
        stations = path_stations(path)
        positions = np.c_[np.interp(station, stations, path[:, 0]), np.interp(station, stations, path[:, 1])]
        heading = np.interp(station, stations, np.unwrap(path[:, 2]))
        speed_controls = np.diff(station) / meta['dt']
        rows, states, controls = ([], [np.asarray(scene['start'], float)], [])
        follower_total_ms = 0.0
        for k in range(400):
            before = states[-1]
            started = time.perf_counter()
            rate = path_feedback_rate(before, speed_controls[k], path, stations, geometric_profile, meta, heading_gain=1.0, lateral_gain=0.05)
            follower_ms = (time.perf_counter() - started) * 1000
            control = np.array([speed_controls[k], rate])
            after = backend.step(before, control)
            states.append(after)
            controls.append(control)
            follower_total_ms += follower_ms
            rows.append({'frame_index': k, 'plan_time_s': times[k], 'time_s': times[k + 1], 'velocity': control[0], 'gamma_dot': control[1], 'x': after[0], 'y': after[1], 'theta': after[2], 'gamma': after[3], 'planned_s': station[k + 1], 'planned_x': positions[k + 1, 0], 'planned_y': positions[k + 1, 1], 'follower_wall_ms': follower_ms})
        states, controls = (np.asarray(states), np.asarray(controls))
        execution = execution_metrics(states, controls, path, initial, meta, times)
        execution['initial_acceleration_mps2'] = float(controls[0, 0] / meta['dt'])
        dense_times = np.linspace(0.0, 40.0, 4001)
        dense_s, _ = sample_profile(result, dense_times)
        dense_xy = np.c_[np.interp(dense_s, stations, path[:, 0]), np.interp(dense_s, stations, path[:, 1])]
        dense_residual = obstacle_residual(dense_xy, dense_times, initial, meta)
        actual_dense = np.c_[np.interp(dense_times, times, states[:, 0]), np.interp(dense_times, times, states[:, 1])]
        execution_dense_residual = obstacle_residual(actual_dense, dense_times, initial, meta)
        execution['dense_obstacle_max_residual'] = float(np.max(execution_dense_residual))
        execution['dense_obstacle_violation_samples'] = int(np.count_nonzero(execution_dense_residual > 1e-09))
        execution['dense_collision_check_dt_s'] = 0.01
        summary = {'map': 'B201', 'method': 'Hybrid A* + ST*', 'planning_success': True, 'reason': result['reason'], 'reached_graph_goal': result['reached_goal'], 'arrival_time_s': result['arrival_time_s'], 'terminal_time_s': float(result['times'][-1]), 'planned_remaining_distance_m': float(stations[-1] - result['s'][-1]), 'planned_max_obstacle_residual_10ms': float(np.max(dense_residual)), 'planned_obstacle_violation_samples_10ms': int(np.count_nonzero(dense_residual > 1e-09)), 'path_stage_wall_ms': path_result['path_stage']['stage_wall_ms'], 'st_search_wall_ms': search_ms, 'geometric_speed_limit_wall_ms': geometric_profile['wall_ms'], 'graph_search_wall_ms': result['stats']['solver_wall_ms'], 'follower_total_wall_ms': follower_total_ms, 'execution': execution}
        write_csv(output / 'cycles.csv', rows)
        write_csv(output / 'planned_trajectory.csv', [{'time_s': t, 's_m': s, 'speed_mps': v, 'x': xy[0], 'y': xy[1], 'heading': h} for t, s, v, xy, h in zip(times, station, instantaneous_speed, positions, heading)])
        np.savez_compressed(output / 'trajectories.npz', times=times, path=path, planned_s=station, planned_positions=positions, planned_speeds=instantaneous_speed, actual_states=states, controls=controls, initial_obstacles=initial)
        write_json(output / 'summary.json', summary)
        comparison = benchmark_rows(baseline, scene, meta, summary)
        shared_keys = list(comparison[0])
        write_csv(output / 'B201_four_method_comparison.csv', [{k: row[k] for k in shared_keys} for row in comparison])
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
if __name__ == '__main__':
    main()