"""候选重放轨迹与原生日志一致性核验工具。"""
import hashlib
import json
import math
import numpy as np
from original_backend import HERE, ROOT, ORIGINAL, OriginalBackend
from verify_consistency import near

def sha(path):
    """计算文件哈希值。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    """核验所有候选回放轨迹的一致性。"""
    base_root = ROOT / '03_实验数据/原版基准核验_20261104'
    cases = [('B201_v5_N80', 'B201', 3), ('B201_v5_N50', 'B201', 1), ('B207_v6_N80', 'B207', 1)]
    before = {str(p.relative_to(ORIGINAL)): sha(p) for p in [ORIGINAL / 'src/ilqr.cpp', ORIGINAL / 'src/config_loader.cpp']}
    report = {'scope': 'Three real baseline candidates, one state per method except B201_v5_N80 with 3 consecutive cycles per method; no map MPC.', 'comparison_tolerance': {'absolute': 1e-08, 'relative': 1e-09}, 'cases': [], 'original_hashes_before': before, 'derived_rate_note': 'The old README draft decimal 0.1568499014 was a hand-written inaccurate approximation; it was never a parameter. All values below come from the unchanged original finalize_parameters function.'}
    for name, mapname, cycles in cases:
        scene_path = base_root / name / mapname / 'scene/scene.json'
        scene = json.loads(scene_path.read_text(encoding='utf-8'))
        for method, folder in [('r', 'cilqr'), ('al', 'alilqr')]:
            source = base_root / name / mapname / folder
            config_path = source / 'effective_config.json'
            effective = json.loads(config_path.read_text(encoding='utf-8'))
            configs = {key: effective[key] for key in ['main', 'ilqr', 'hybrid_astar']}
            record = {'candidate': name, 'map': mapname, 'method': method, 'cycles': [], 'scene_sha256': sha(scene_path), 'effective_config_sha256': sha(config_path)}
            state = np.asarray(scene['start'], dtype=float)
            obs = np.asarray(scene['initial_obstacles'], dtype=float).reshape(-1, 4)
            with OriginalBackend(scene['path'], configs) as backend:
                m = backend.metadata
                for key in ['N', 'dt', 'lf', 'lr', 'gamma_dot_min', 'gamma_dot_max']:
                    near(m[key], effective['runtime'][key], name + '/' + method + '/' + key, atol=0.0, rtol=0.0)
                near(m['configured_speed'], effective['runtime']['desire_speed'], name + '/speed', atol=0.0, rtol=0.0)
                record['runtime'] = m
                for cycle in range(cycles):
                    saved_path = source / f'outputs/{mapname}/{folder}/data/{folder}_data_{cycle}.json'
                    saved = json.loads(saved_path.read_text(encoding='utf-8'))
                    result = backend.solve_original(method, state, obs)
                    expected_u = np.column_stack([saved['control_sequence']['velocity'], saved['control_sequence']['gamma']])
                    expected_x = np.column_stack([saved['planned_trajectory'][key] for key in ['x', 'y', 'theta', 'gamma']])
                    ue = near(result['controls'], expected_u, name + '/' + method + f'/{cycle}/controls', atol=1e-08, rtol=1e-09)
                    xe = near(result['states'], expected_x, name + '/' + method + f'/{cycle}/prediction', atol=1e-08, rtol=1e-09)
                    state = backend.step(state, result['controls'][0])
                    expected_actual = [saved['ego_vehicle'][key] for key in ['x', 'y', 'theta', 'gamma']]
                    se = near(state, expected_actual, name + '/' + method + f'/{cycle}/actual', atol=1e-08, rtol=1e-09)
                    for obstacle in obs:
                        obstacle[0] = obstacle[0] + obstacle[3] * math.cos(obstacle[2]) * m['dt']
                        obstacle[1] = obstacle[1] + obstacle[3] * math.sin(obstacle[2]) * m['dt']
                    expected_obs = np.array([[o['x'], o['y'], o['theta'], o['speed']] for o in saved['obstacles']]).reshape(-1, 4)
                    oe = near(obs, expected_obs, name + '/' + method + f'/{cycle}/obstacles', atol=1e-12, rtol=0.0)
                    target_key = 'r_effective_speed' if method == 'r' else 'al_effective_speed'
                    near(result['metadata'][target_key], m['configured_speed'], name + '/' + method + f'/{cycle}/persistent target', atol=0.0, rtol=0.0)
                    record['cycles'].append({'cycle': cycle, 'archive_sha256': sha(saved_path), 'control_max_abs_error': ue, 'prediction_max_abs_error': xe, 'actual_state_max_abs_error': se, 'obstacle_state_max_abs_error': oe, 'effective_target_speed': result['metadata'][target_key], 'recomputed_native_cost': result['recomputed_native_cost'], 'original_geometry_max_residual': result['original_geometry_max_residual']})
            report['cases'].append(record)
    after = {str(p.relative_to(ORIGINAL)): sha(p) for p in [ORIGINAL / 'src/ilqr.cpp', ORIGINAL / 'src/config_loader.cpp']}
    assert before == after
    report['original_hashes_after'] = after
    report['all_checks_passed'] = True
    target = HERE / 'verification/candidate_replay_report.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PASS: ' + str(target))
if __name__ == '__main__':
    main()