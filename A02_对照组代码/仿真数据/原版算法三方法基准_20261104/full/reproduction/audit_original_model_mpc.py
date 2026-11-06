"""原模型与各求解器全量数值、动力学状态转移方程与物理限值独立核验脚本。"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import numpy as np

def read(path):
    """读取 JSON 文件内容。"""
    return json.loads(path.read_text(encoding='utf-8'))

def lines(path):
    """按行读取文本文件。"""
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s]

def sha(path):
    """计算哈希校验值。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()

def num(x):
    """转换为双精度浮点数组。"""
    return {'NaN': math.nan, '+Infinity': math.inf, '-Infinity': -math.inf}.get(x, x) if isinstance(x, str) else x

def equivalent(a, b):
    """数值全等比对。"""
    if b is None:
        return a is None or a == ''
    if isinstance(b, bool):
        return str(a).lower() == str(b).lower()
    try:
        av, bv = (float(num(a)), float(num(b)))
        return bool(np.isclose(av, bv, atol=1e-08, rtol=1e-09, equal_nan=True))
    except (ValueError, TypeError):
        return a == b

def dynamics(x, u, m):
    """独立核验前向动力学欧拉状态推进。"""
    px, py, theta, gamma = x
    speed, rate = u
    yaw = theta - m['dt'] * (speed * math.sin(gamma) + m['lr'] * rate) / (m['lr'] + m['lf'] * math.cos(gamma))
    return np.array([px + m['dt'] * speed * math.cos(theta), py + m['dt'] * speed * math.sin(theta), math.atan2(math.sin(yaw), math.cos(yaw)), gamma + m['dt'] * rate])

def state_error(a, b):
    """计算状态误差。"""
    d = np.asarray(a) - np.asarray(b)
    d[..., 2] = np.arctan2(np.sin(d[..., 2]), np.cos(d[..., 2]))
    return float(np.max(np.abs(d), initial=0.0))

def obstacle_states(initial, t):
    """计算障碍物状态。"""
    out = initial.copy()
    if len(out):
        out[:, 0] += t * initial[:, 3] * np.cos(initial[:, 2])
        out[:, 1] += t * initial[:, 3] * np.sin(initial[:, 2])
    return out

def barrier(q1, q2, residual):
    """计算控制障碍函数屏障代价。"""
    raw = q1 * np.exp(np.minimum(q2 * residual, 20.0))
    transition = 0.5 * (1.0 + np.tanh((1000000.0 - raw) / 100000.0))
    return raw * transition + 1000000.0 * (1 - transition)

def independent_geometry_cost(path, before, xs, us, obs, m):
    """独立计算几何代价。"""
    start = int(np.argmin(np.linalg.norm(path[:, :2] - before[:2], axis=1)))
    count = int(max(before[3] * m['dt'] * m['N'], 100.0) + 300.0)
    local = path[start:start + count]
    if len(local) < count:
        local = np.concatenate([local, np.repeat(local[-1:], count - len(local), axis=0)])
    nearest = np.argmin(np.linalg.norm(xs[:, None, :2] - local[None, :, :2], axis=2), axis=1)
    refs = local[np.minimum(nearest + 1, len(local) - 1)]
    err = xs - np.column_stack([refs, np.zeros(len(refs))])
    lateral = -err[:, 0] * np.sin(refs[:, 2]) + err[:, 1] * np.cos(refs[:, 2])
    gup, glo = (xs[:, 3] - m['gamma_max'], m['gamma_min'] - xs[:, 3])
    lup, llo = (lateral - m['trace_safe_width_left'], -lateral - m['trace_safe_width_right'])
    uup, ulo = (us[:, 1] - m['gamma_dot_max'], m['gamma_dot_min'] - us[:, 1])
    obstacles = []
    for o in obs:
        ts = np.arange(len(us)) * m['dt']
        dx = xs[:-1, 0] - (o[0] + ts * o[3] * np.cos(o[2]))
        dy = xs[:-1, 1] - (o[1] + ts * o[3] * np.sin(o[2]))
        ax = np.cos(o[2]) * dx + np.sin(o[2]) * dy
        ay = -np.sin(o[2]) * dx + np.cos(o[2]) * dy
        A = m['obs_length'] / 2 + m['ego_rad'] / 2 + m['safe_a_buffer']
        B = m['obs_width'] / 2 + m['ego_rad'] / 2 + m['safe_b_buffer']
        obstacles.append(1 - (ax / A) ** 2 - (ay / B) ** 2)
    residual = []
    for k in range(len(xs)):
        residual.extend([gup[k], glo[k]])
        if m['if_cal_lane_cost']:
            residual.extend([lup[k], llo[k]])
        if m['if_cal_obs_cost'] and k < len(us):
            residual.extend((o[k] for o in obstacles))
    for k in range(len(us)):
        residual.extend([uup[k], ulo[k]])
    qdiag, rdiag = (np.diag(np.asarray(m['Q'])), np.diag(np.asarray(m['R'])))
    du = us - np.array([m['r_effective_speed'], 0.0])
    base = float(np.sum(err * err * qdiag) + m['ref_weight'] * np.sum(lateral * lateral) + np.sum(du * du * rdiag))
    if m['if_cal_speed_rate_cost']:
        base += m['v_rate_weight'] * float(np.sum(np.diff(us[:, 0]) ** 2))
    penalty = 0.0
    if m['if_cal_gamma_barrier']:
        penalty += float(np.sum(barrier(m['gamma_max_q1'], m['gamma_max_q2'], gup)) + np.sum(barrier(m['gamma_min_q1'], m['gamma_min_q2'], glo)))
    if m['if_cal_gamma_dot_barrier']:
        penalty += float(np.sum(barrier(m['gamma_dot_max_q1'], m['gamma_dot_max_q2'], uup)) + np.sum(barrier(m['gamma_dot_min_q1'], m['gamma_dot_min_q2'], ulo)))
    if m['if_cal_lane_cost']:
        penalty += float(m['lane_q1'] * np.sum(np.exp(m['lane_q2'] * lup) + np.exp(m['lane_q2'] * llo)))
    if m['if_cal_obs_cost']:
        penalty += sum((float(m['obs_q1'] * np.sum(np.exp(m['obs_q2'] * o))) for o in obstacles))
    return (np.asarray(residual), refs, base + penalty)

def json_safe(value):
    """数据安全转换。"""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (float, np.floating)) and (not np.isfinite(value)):
        return 'NaN' if np.isnan(value) else '+Infinity' if value > 0 else '-Infinity'
    if isinstance(value, np.generic):
        return value.item()
    return value

def main():
    """执行全量一致性与物理约束独立核验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--audit-output', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    protocol = read(output / 'protocol.json')
    issues, warnings, audited = ([], [], [])

    def check(condition, message):
        if not condition:
            issues.append(message)
    for rel, digest in protocol['reproduction_sha256'].items():
        check(sha(output / 'reproduction' / rel) == digest, 'snapshot hash mismatch: ' + rel)
    from project_paths import project_root, original_dir
    original = original_dir(project_root(Path(__file__).resolve().parent))
    for source in ['ilqr.cpp', 'config_loader.cpp']:
        check(sha(output / 'reproduction/generated' / source) == sha(original / 'src' / source), 'original core source mismatch: ' + source)
    for mapname in protocol['maps']:
        scene_path = output / mapname / 'scene/scene.json'
        scene = read(scene_path)
        check(sha(scene_path) == protocol['inputs'][mapname]['scene_sha256'], mapname + ': frozen scene mismatch')
        path = np.asarray(scene['path'], float)
        initial_obs = np.asarray(scene['initial_obstacles'], float).reshape(-1, 4)
        for method in protocol['methods']:
            label = mapname + '/' + {'r': 'cilqr', 'al': 'alilqr', 'sqp': 'sqp'}[method]
            folder = output / label
            metadata = read(folder / 'metadata.json')
            m0 = metadata['initial_cpp_parameters']
            config = read(folder / 'effective_config.json')
            summary = read(folder / 'summary.json')
            cycles = lines(folder / 'cycles.jsonl')
            attempts = lines(folder / 'attempts.jsonl')
            with (folder / 'cycles.csv').open(encoding='utf-8-sig', newline='') as stream:
                csvrows = list(csv.DictReader(stream))
            trajectory = np.load(folder / 'trajectories.npz', allow_pickle=False)
            check(len(cycles) == len(attempts) == len(csvrows) == summary['recorded_attempts'], label + ': record count mismatch')
            check(m0['N'] == protocol['N'] and m0['dt'] == protocol['dt_s'], label + ': horizon/dt mismatch')
            check(m0['configured_speed'] == protocol['candidate_reference_speeds'][mapname], label + ': reference speed mismatch')
            check(config['ilqr']['cilqr']['N'] == m0['N'] and config['ilqr']['cost_weights']['desire_speed'] == m0['configured_speed'], label + ': config mismatch')
            actual = np.asarray(scene['start'], float)
            check(np.allclose(trajectory['actual_states'][0], actual, atol=1e-08, rtol=1e-10), label + ': aggregate initial state mismatch')
            check(np.allclose(trajectory['initial_obstacles'], initial_obs, atol=1e-08, rtol=1e-10), label + ': aggregate initial obstacles mismatch')
            target = m0['configured_speed']
            exmax = prmax = obsmax = costmax = resmax = 0.0
            predicted_clip_frames = []
            executed = 0
            nonfinite = []
            wall = []
            geometric_violations = hard_violations = successful = failed_executed = 0
            samples = []
            for i, (cycle, attempt, row) in enumerate(zip(cycles, attempts, csvrows)):
                s = cycle['summary']
                m = cycle['cpp_metadata']
                check(s['frame_index'] == s['cycle'] == i == attempt['frame_index'], label + f': index mismatch {i}')
                check(s['optimizer_attempts'] == 1 and attempt['attempt_index'] == 0, label + f': attempt count changed {i}')
                for key, value in s.items():
                    check(equivalent(row.get(key), value), label + f': CSV/JSONL mismatch {i}/{key}')
                check(equivalent(s['plan_time_s'], i * m0['dt']) and equivalent(s['time_s'], (i + 1) * m0['dt']), label + f': clock mismatch {i}')
                check(np.allclose(attempt['state_before'], actual, rtol=1e-10, atol=1e-08, equal_nan=True), label + f': attempt state chain mismatch {i}')
                if method != 'al':
                    distance = np.linalg.norm(actual[:2] - path[-1, :2])
                    if distance < 10:
                        target *= distance / 10
                        if target < 0.1:
                            target = 0.0
                if s['effective_desired_speed'] is not None:
                    check(equivalent(s['effective_desired_speed'], target), label + f': persistent target speed mismatch {i}')
                if not attempt.get('arrays_file'):
                    check(not s['executed'], label + f': executed frame lost arrays {i}')
                    warnings.append(label + f': retained exception at frame {i}: ' + str(attempt.get('exception')))
                    continue
                npz_path = folder / attempt['arrays_file']
                check(cycle['arrays_file'] == attempt['arrays_file'], label + f': arrays link mismatch {i}')
                check(sha(npz_path) == attempt['arrays_sha256'], label + f': NPZ hash mismatch {i}')
                z = np.load(npz_path, allow_pickle=False)
                before, xs, us, after, obs = (z[k] for k in ['state_before', 'planned_states', 'controls', 'state_after', 'obstacles_current'])
                check(before.shape == (4,) and xs.shape == (m0['N'] + 1, 4) and (us.shape == (m0['N'], 2)), label + f': shape mismatch {i}')
                check(np.allclose(before, actual, atol=1e-08, rtol=1e-10, equal_nan=True), label + f': actual state chain mismatch {i}')
                check(np.allclose(np.asarray(attempt['obstacles_current'], float).reshape(-1, 4), obs, atol=1e-08, rtol=1e-10), label + f': attempt/NPZ obstacle mismatch {i}')
                expected_obs = obstacle_states(initial_obs, s['plan_time_s'])
                obs_error = float(np.max(np.abs(obs - expected_obs), initial=0.0))
                obsmax = max(obsmax, obs_error)
                check(obs_error < 1e-08, label + f': absolute obstacle clock mismatch {i}')
                check(np.allclose(z['obstacles_after'], obstacle_states(initial_obs, s['time_s']), atol=1e-08, rtol=1e-10), label + f': obstacle-after clock mismatch {i}')
                finite = bool(np.isfinite(xs).all() and np.isfinite(us).all())
                check(equivalent(s['velocity'], us[0, 0]) and equivalent(s['gamma_dot'], us[0, 1]), label + f': applied first control mismatch {i}')
                if s['executed']:
                    expected = dynamics(before, us[0], m0)
                    e = state_error(after, expected)
                    exmax = max(exmax, e)
                    check(e < 1e-07, label + f': execution dynamics mismatch {i}')
                    check(np.allclose(after, [s[k] for k in ['x', 'y', 'theta', 'gamma']], atol=1e-08, rtol=1e-10), label + f': after-state CSV mismatch {i}')
                    actual = after
                    executed += 1
                    check(np.allclose(trajectory['actual_states'][executed], after, atol=1e-08, rtol=1e-10), label + f': aggregate actual mismatch {i}')
                    if method == 'sqp':
                        check(s['warmstart_committed'] == bool(np.isfinite(us).all()), label + f': SQP warmstart commit policy mismatch {i}')
                        failed_executed += int(s['scipy_success'] is False)
                if finite:
                    check(state_error(xs[0], before) < 1e-07, label + f': planned initial mismatch {i}')
                    pe = max((state_error(xs[k + 1], dynamics(xs[k], us[k], m0)) for k in range(len(us))))
                    prmax = max(prmax, pe)
                    if pe > 1e-07:
                        predicted_clip_frames.append(i)
                        if method == 'sqp':
                            check(False, label + f': SQP prediction violates exact dynamics {i}')
                    residual, refs, cost = independent_geometry_cost(path, before, xs, us, obs, m)
                    maximum = float(max(0.0, np.max(residual, initial=0.0)))
                    re = abs(maximum - num(s['max_original_residual']))
                    resmax = max(resmax, re)
                    check(np.isclose(maximum, num(s['max_original_residual']), atol=1e-07, rtol=1e-08), label + f': residual mismatch {i}')
                    geometric_violations += int(maximum > 0.001)
                    if method in ['r', 'sqp']:
                        ce = abs(cost - num(s['recomputed_cost']))
                        costmax = max(costmax, ce)
                        check(np.isclose(cost, num(s['recomputed_cost']), atol=1e-05, rtol=1e-08), label + f': R penalty cost mismatch {i}: {ce}')
                    if method == 'sqp':
                        check(np.allclose(z['original_residuals'], residual, atol=1e-07, rtol=1e-08), label + f': residual vector mismatch {i}')
                        check(np.allclose(z['matched_references'], refs, atol=1e-08, rtol=1e-10), label + f': reference match mismatch {i}')
                        dv = np.diff(us[:, 0]) / m['dt']
                        hard = float(max(0.0, np.max(-5 - us[:, 0]), np.max(us[:, 0] - 30), np.max(np.abs(us[:, 1]) - 1.5), np.max(m['acc_min'] - dv), np.max(dv - m['acc_max'])))
                        check(np.isclose(hard, num(s['hard_control_max_residual']), atol=1e-07, rtol=1e-08), label + f': hard-control residual mismatch {i}')
                        hard_violations += int(hard > 1e-06)
                        check(equivalent(s['solver_wall_ms'], s['prepare_wall_ms'] + s['initialization_and_minimize_ms']), label + f': SQP timing sum mismatch {i}')
                        successful += int(s['scipy_success'] is True)
                else:
                    nonfinite.append(i)
                check(equivalent(s['solve_time_ms'], s['solver_wall_ms']), label + f': wall alias mismatch {i}')
                if s['solver_wall_ms'] is not None:
                    wall.append(num(s['solver_wall_ms']))
                    check(num(s['solver_wall_ms']) >= 0, label + f': negative timing {i}')
                check(np.allclose(trajectory['planned_states'][len(samples)], xs, atol=1e-08, rtol=1e-10, equal_nan=True), label + f': aggregate plan mismatch {i}')
                check(np.allclose(trajectory['controls'][len(samples)], us, atol=1e-08, rtol=1e-10, equal_nan=True), label + f': aggregate control mismatch {i}')
                samples.append(i)
            check(executed == summary['executed_frames'] == len(trajectory['actual_states']) - 1, label + ': executed count mismatch')
            check(summary['completed_fixed_window'] == (executed == protocol['steps'][mapname]), label + ': completion flag mismatch')
            check(equivalent(summary['mean_solver_wall_ms'], float(np.mean(wall)) if wall else None), label + ': timing mean mismatch')
            check(equivalent(summary['max_solver_wall_ms'], max(wall) if wall else None), label + ': timing maximum mismatch')
            audited.append({'map': mapname, 'method': method, 'attempts': len(cycles), 'executed': executed, 'completed_fixed_window': summary['completed_fixed_window'], 'execution_error_max': exmax, 'prediction_error_max': prmax, 'prediction_clip_diagnostic_frames': predicted_clip_frames, 'obstacle_error_max': obsmax, 'R_penalty_cost_abs_error_max': costmax if method != 'al' else None, 'original_residual_error_max': resmax, 'nonfinite_frames': nonfinite, 'original_constraint_residual_over_1e_3_frames': geometric_violations, 'SQP_hard_residual_over_1e_6_frames': hard_violations if method == 'sqp' else None, 'SQP_success_count': successful if method == 'sqp' else None, 'SQP_failed_but_executed_count': failed_executed if method == 'sqp' else None, 'mean_solver_wall_ms': float(np.mean(wall)) if wall else None, 'max_solver_wall_ms': max(wall) if wall else None, 'solver_wall_over100ms': sum((t > 100 for t in wall)), 'final_state': actual, 'final_goal_distance_m': float(np.linalg.norm(actual[:2] - path[-1, :2]))})
    report = {'record_audit_pass': not issues, 'issues': issues, 'warnings': warnings, 'scope': __doc__, 'limitations': ['AL augmented cost includes internal multipliers not exported; independent cost audit is limited to R and SQP.', 'Original constraint residual includes articulation and angular-rate limits as well as lane/obstacle geometry; it is not a whole-vehicle occupancy-grid collision check.', 'Record audit pass does not imply convergence, successful arrival, historical reproduction, or real-time feasibility.'], 'protocol_sha256': sha(output / 'protocol.json'), 'runs': audited}
    dest = args.audit_output or output / 'independent_record_audit.json'
    if dest.exists():
        raise FileExistsError(dest)
    dest.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(json_safe({'pass': not issues, 'issues': issues, 'warnings': warnings, 'runs': audited, 'output': str(dest)}), ensure_ascii=False))
if __name__ == '__main__':
    main()