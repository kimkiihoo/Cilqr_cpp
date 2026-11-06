"""仿真输出轨迹与历史冻结记录一致性核验与候选重放数据交叉检验脚本。"""
from pathlib import Path
import copy
import hashlib
import json
import math
import sys
import numpy as np
from original_backend import HERE, ROOT, ORIGINAL, OriginalBackend, read_configs
from sqp_nmpc import solve_prepared

def error(actual, expected):
    """计算相对误差。"""
    a, b = (np.asarray(actual), np.asarray(expected))
    return float(np.max(np.abs(a - b), initial=0.0))

def near(actual, expected, label, atol=1e-09, rtol=1e-11):
    """查找邻近路点。"""
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(f'{label}: max error {error(actual, expected)}')
    return error(actual, expected)

def original_step(x, u, m):
    """模型单步推演。"""
    px, py, theta, gamma = x
    speed, rate = u
    yaw = theta - m['dt'] * (speed * math.sin(gamma) + m['lr'] * rate) / (m['lr'] + m['lf'] * math.cos(gamma))
    return np.array([px + m['dt'] * speed * math.cos(theta), py + m['dt'] * speed * math.sin(theta), (yaw + math.pi) % (2 * math.pi) - math.pi, gamma + m['dt'] * rate])

def barrier(q1, q2, residual):
    """屏障函数评价。"""
    raw = q1 * math.exp(min(q2 * residual, 20.0))
    transition = 0.5 * (1.0 + math.tanh((1000000.0 - raw) / 100000.0))
    return raw * transition + 1000000.0 * (1 - transition)

def independently_evaluate(path, initial, obstacles, controls, m):
    """独立评估轨迹残差与代价。"""
    N = len(controls)
    local_start = int(np.argmin(np.sum((path[:, :2] - initial[:2]) ** 2, axis=1)))
    count = int(max(initial[3] * m['dt'] * N, 100) + 300)
    local = path[local_start:local_start + count]
    if len(local) < count:
        local = np.vstack([local, np.repeat(local[-1:], count - len(local), axis=0)])
    states = [initial]
    for u in controls:
        states.append(original_step(states[-1], u, m))
    states = np.asarray(states)
    refs = []
    base_state = 0.0
    penalty_r = 0.0
    penalty_al = 0.0
    residuals = []
    Q, R = (np.asarray(m['Q']), np.asarray(m['R']))
    for k, x in enumerate(states):
        closest = int(np.argmin(np.sum((local[:, :2] - x[:2]) ** 2, axis=1)))
        ref = local[min(closest + 1, len(local) - 1)]
        refs.append(ref)
        e = x - np.array([*ref, 0.0])
        lateral = e[0] * -math.sin(ref[2]) + e[1] * math.cos(ref[2])
        base_state += float(e @ Q @ e) + m['ref_weight'] * lateral * lateral
        gmax, gmin = (x[3] - m['gamma_max'], m['gamma_min'] - x[3])
        residuals.extend([gmax, gmin])
        if m['if_cal_gamma_barrier']:
            penalty_r += barrier(m['gamma_max_q1'], m['gamma_max_q2'], gmax)
            penalty_r += barrier(m['gamma_min_q1'], m['gamma_min_q2'], gmin)
        penalty_al += 5 * (max(0.0, gmax) ** 2 + max(0.0, gmin) ** 2)
        if m['if_cal_lane_cost']:
            left, right = (lateral - m['trace_safe_width_left'], -lateral - m['trace_safe_width_right'])
            residuals.extend([left, right])
            penalty_r += m['lane_q1'] * (math.exp(m['lane_q2'] * left) + math.exp(m['lane_q2'] * right))
            penalty_al += 5 * max(0.0, left, right) ** 2
        obs_res = []
        if m['if_cal_obs_cost'] and k < N:
            for o in obstacles:
                heading = o[2]
                ox = o[0] + k * m['dt'] * o[3] * math.cos(heading)
                oy = o[1] + k * m['dt'] * o[3] * math.sin(heading)
                dx, dy = (x[0] - ox, x[1] - oy)
                a = m['obs_length'] / 2 + m['ego_rad'] / 2 + m['safe_a_buffer']
                b = m['obs_width'] / 2 + m['ego_rad'] / 2 + m['safe_b_buffer']
                aligned_x = math.cos(heading) * dx + math.sin(heading) * dy
                aligned_y = -math.sin(heading) * dx + math.cos(heading) * dy
                c = 1 - (aligned_x / a) ** 2 - (aligned_y / b) ** 2
                obs_res.append(c)
                residuals.append(c)
                penalty_r += m['obs_q1'] * math.exp(m['obs_q2'] * c)
            penalty_al += 5 * max([0.0] + obs_res) ** 2
    base_r, base_al = (base_state, base_state)
    for u in controls:
        er, ea = (u - np.array([m['r_effective_speed'], 0.0]), u - np.array([m['al_effective_speed'], 0.0]))
        base_r += float(er @ R @ er)
        base_al += float(ea @ R @ ea)
        upper, lower = (u[1] - m['gamma_dot_max'], m['gamma_dot_min'] - u[1])
        residuals.extend([upper, lower])
        if m['if_cal_gamma_dot_barrier']:
            penalty_r += barrier(m['gamma_dot_max_q1'], m['gamma_dot_max_q2'], upper)
            penalty_r += barrier(m['gamma_dot_min_q1'], m['gamma_dot_min_q2'], lower)
        penalty_al += 5 * (max(0.0, upper) ** 2 + max(0.0, lower) ** 2)
    if m['if_cal_speed_rate_cost']:
        smooth = m['v_rate_weight'] * float(np.square(np.diff(controls[:, 0])).sum())
        base_r += smooth
        base_al += smooth
    return {'states': states, 'matched_references': refs, 'residuals': residuals, 'r_base_cost': base_r, 'al_base_cost': base_al, 'r_exact_cost': base_r + penalty_r, 'al_augmented_cost': base_al + penalty_al}

def main():
    """执行一致性校验。"""
    out = HERE / 'verification'
    out.mkdir(exist_ok=True)
    source = ORIGINAL / 'src/ilqr.cpp'
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    scene_path = ROOT / '03_实验数据/原地图对照实验_20261103/scenes/B201/scene.json'
    scene = json.loads(scene_path.read_text(encoding='utf-8'))
    path, initial, obstacles = (np.asarray(scene['path']), np.asarray(scene['start']), np.asarray(scene['initial_obstacles']))
    configs = read_configs()
    configs['ilqr']['cilqr']['N'] = 12
    report = {'scope': 'Single-state differential verification and a two-iteration SQP interface smoke test; no map rollout.', 'small_test_N_override': 12, 'cases': [], 'source_sha256_before': before}
    cases = [('ordinary', initial, obstacles), ('near_obstacle', np.array([obstacles[0, 0] - 3.0, obstacles[0, 1], 0.2, 0.1]), obstacles), ('yaw_wrap_and_gamma_limit', np.array([initial[0], initial[1], -3.13, 1.05]), obstacles)]
    for label, state, obs in cases:
        with OriginalBackend(path, configs) as b:
            m = b.prepare(state, obs)
            u = np.column_stack([3.0 + 0.15 * np.arange(b.N), 0.5 * np.sin(np.arange(b.N) / 3)])
            native = b.evaluate(u)
            reference = independently_evaluate(path, state, obs, u, m)
            diffs = {key: near(native[key], reference[key], label + '/' + key) for key in reference}
            diffs['native_al_residual'] = near(native['max_residual'], native['original_al_max_residual'], label + '/native AL residual')
            shifted = u.copy()
            shifted[:, 0] += 5.0
            native2 = b.evaluate(shifted)
            ref2 = independently_evaluate(path, state, obs, shifted, m)
            near(native2['matched_references'], ref2['matched_references'], label + '/changed candidate refs')
            report['cases'].append({'name': label, 'errors': diffs, 'candidate_reference_change': bool(np.any(native2['matched_references'] != native['matched_references']))})
    with OriginalBackend(path, configs) as b:
        near_goal = np.array([path[-1, 0] - 4.0, path[-1, 1], path[-1, 2], 0.0])
        m1 = b.prepare(near_goal, obstacles)
        m2 = b.prepare(near_goal, obstacles)
        near(m1['r_effective_speed'], 6.0, 'R first terminal slowdown')
        near(m2['r_effective_speed'], 2.4, 'R persistent terminal slowdown')
        near(m2['al_effective_speed'], 15.0, 'AL unchanged target')
        report['terminal_targets'] = {'r_first': m1['r_effective_speed'], 'r_second': m2['r_effective_speed'], 'al': m2['al_effective_speed']}
    report['baseline_single_state'] = []
    for label, folder in [('r', 'cilqr'), ('al', 'alilqr')]:
        base = ROOT / f'03_实验数据/原算法原地图复跑_20261103/B201/{folder}'
        effective = json.loads((base / 'effective_config.json').read_text(encoding='utf-8'))
        baseline_cfg = {key: effective[key] for key in ['main', 'ilqr', 'hybrid_astar']}
        saved = json.loads((base / f'outputs/B201/{folder}/data/{folder}_data_0.json').read_text(encoding='utf-8'))
        with OriginalBackend(path, baseline_cfg) as b:
            result = b.solve_original(label, initial, obstacles)
            expected_u = np.column_stack([saved['control_sequence']['velocity'], saved['control_sequence']['gamma']])
            expected_x = np.column_stack([saved['planned_trajectory'][key] for key in ['x', 'y', 'theta', 'gamma']])
            ue = near(result['controls'], expected_u, label + '/archived controls', atol=1e-08, rtol=1e-09)
            xe = near(result['states'], expected_x, label + '/archived prediction', atol=1e-08, rtol=1e-09)
            report['baseline_single_state'].append({'method': label, 'N': b.N, 'control_max_abs_error': ue, 'state_max_abs_error': xe, 'archived_source': str(base)})
    with OriginalBackend(path, configs) as b:
        b.prepare(initial, obstacles)
        smoke = solve_prepared(b, maxiter=2)
        near(smoke['r_exact_cost'], b.cost(smoke['controls']), 'SQP objective round trip')
        assert np.isfinite(smoke['states']).all()
        report['sqp_smoke'] = {key: smoke[key] for key in ['status', 'message', 'iterations', 'objective_calls', 'hard_control_max_residual', 'max_residual', 'r_exact_cost']}
    after = hashlib.sha256(source.read_bytes()).hexdigest()
    assert before == after
    report['source_sha256_after'] = after
    report['all_checks_passed'] = True
    (out / 'consistency_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PASS: ' + str(out / 'consistency_report.json'))
if __name__ == '__main__':
    main()