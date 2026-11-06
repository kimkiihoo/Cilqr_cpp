"""Hybrid A* + ST* 在 B201 场景下仿真输出数据的独立质量与安全性核验脚本。

核验内容:
- 检验几何路径与障碍物是否与冻结基准严格逐值一致；
- 检验 40 s 时域内全状态序列的物理约束、横向偏差、走廊限宽与最小椭圆避障净距。
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np

def read(path):
    """读取指定名称的 JSON 或 CSV 结果文件。"""
    return json.loads(path.read_text(encoding='utf-8'))

def main():
    """执行完整的独立核验流程并导出核验报告 JSON 文件。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    data = np.load(root / 'trajectories.npz')
    graph, summary, protocol = (read(root / f) for f in ('st_search.json', 'summary.json', 'protocol.json'))
    scene = read(root / 'scene.json')
    profile = np.load(root / 'geometric_speed_profile.npz')
    m = protocol['model_parameters']
    states, controls, times = (data['actual_states'], data['controls'], data['times'])
    assert states.shape == (401, 4) and controls.shape == (400, 2)
    assert np.isfinite(states).all() and np.isfinite(controls).all()
    assert np.array_equal(states[0], scene['start'])
    assert np.array_equal(data['path'], scene['path'])
    assert np.array_equal(data['initial_obstacles'], scene['initial_obstacles'])
    assert np.allclose(times, np.arange(401) * 0.1, rtol=0, atol=1e-12)
    expected = states[:-1].copy()
    theta, gamma, velocity, rate = (states[:-1, 2], states[:-1, 3], controls[:, 0], controls[:, 1])
    expected[:, 0] += 0.1 * velocity * np.cos(theta)
    expected[:, 1] += 0.1 * velocity * np.sin(theta)
    yaw = theta - 0.1 * (velocity * np.sin(gamma) + m['lr'] * rate) / (m['lr'] + m['lf'] * np.cos(gamma))
    expected[:, 2] = np.arctan2(np.sin(yaw), np.cos(yaw))
    expected[:, 3] += 0.1 * rate
    model_error = float(np.max(np.abs(states[1:] - expected)))
    assert model_error < 1e-11, model_error
    assert np.allclose(controls[:, 0] * 0.1, np.diff(data['planned_s']), rtol=0, atol=1e-11)
    assert np.min(velocity) >= -1e-09 and np.max(velocity) <= 5 + 1e-09
    assert np.min(states[:, 3]) >= m['gamma_min'] - 1e-09
    assert np.max(states[:, 3]) <= m['gamma_max'] + 1e-09
    assert np.min(rate) >= m['gamma_dot_min'] - 1e-09 and np.max(rate) <= m['gamma_dot_max'] + 1e-09
    acceleration = np.diff(np.r_[0.0, velocity]) / 0.1
    assert np.min(acceleration) >= m['acc_min'] - 1e-08 and np.max(acceleration) <= m['acc_max'] + 1e-08
    knots, station, speed = (np.asarray(graph[key], float) for key in ('times', 's', 'speeds'))
    assert np.all(np.diff(knots) > 0) and np.all(np.diff(station) >= 0)
    assert np.allclose(np.diff(station), 0.5 * (speed[1:] + speed[:-1]) * np.diff(knots), rtol=0, atol=1e-10)
    dense_t = np.linspace(0, 40, 4001)
    index = np.clip(np.searchsorted(knots, dense_t, side='right') - 1, 0, len(knots) - 2)
    dt = np.minimum(dense_t - knots[index], np.diff(knots)[index])
    acc = np.diff(speed)[index] / np.diff(knots)[index]
    dense_s = station[index] + speed[index] * dt + 0.5 * acc * dt ** 2
    dense_speed = speed[index] + acc * dt
    cap = np.interp(dense_s, profile['s'], profile['speed_limit'])
    max_cap_violation = float(np.max(dense_speed - cap))
    assert max_cap_violation <= 1e-08, max_cap_violation
    path = data['path']
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
    planned = np.c_[np.interp(dense_s, arc, path[:, 0]), np.interp(dense_s, arc, path[:, 1])]
    executed = np.c_[np.interp(dense_t, times, states[:, 0]), np.interp(dense_t, times, states[:, 1])]
    maxima = []
    for positions in (planned, executed):
        residuals = []
        for ox, oy, heading, speed_obs in data['initial_obstacles']:
            c, sn = (np.cos(heading), np.sin(heading))
            dx = positions[:, 0] - ox - dense_t * speed_obs * c
            dy = positions[:, 1] - oy - dense_t * speed_obs * sn
            a = m['obs_length'] / 2 + m['ego_rad'] / 2 + m['safe_a_buffer']
            b = m['obs_width'] / 2 + m['ego_rad'] / 2 + m['safe_b_buffer']
            residuals.append(1 - ((c * dx + sn * dy) / a) ** 2 - ((-sn * dx + c * dy) / b) ** 2)
        maxima.append(float(np.max(residuals)))
    assert max(maxima) <= 1e-09, maxima
    assert summary['reached_graph_goal'] == graph['reached_goal']
    assert (summary['arrival_time_s'] is None) == (not graph['reached_goal'])
    if not graph['reached_goal']:
        assert knots[-1] == 40 and summary['reason'].startswith('safe_partial_plan')
    with (root / 'cycles.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 400
    csv_controls = np.array([[float(r['velocity']), float(r['gamma_dot'])] for r in rows])
    assert np.array_equal(csv_controls, controls)
    for name, digest in protocol['sources'].items():
        assert hashlib.sha256((root / 'reproduction' / name).read_bytes()).hexdigest() == digest, name
    report = {'passed': True, 'executed_frames': 400, 'model_max_abs_error': model_error, 'dense_check_dt_s': 0.01, 'dense_speed_limit_max_violation_mps': max(0.0, max_cap_violation), 'dense_planned_obstacle_max_residual': maxima[0], 'dense_executed_obstacle_max_residual': maxima[1], 'graph_goal_reached': graph['reached_goal'], 'terminal_time_s': float(knots[-1]), 'all_source_snapshot_hashes_match': True}
    (root / 'audit_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()