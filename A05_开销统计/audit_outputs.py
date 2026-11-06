"""开销统计评测输出与冻结基准结果的逐周期全量位级数值等价性 (Bitwise Equivalence) 核验脚本。

核验口径:
- 针对指定场景与方法的所有仿真周期；
- 逐帧加载 NPZ 数组中的 state_before, state_after, planned_states, controls, obstacles_current；
- 计算最大绝对数值偏差 max_abs_diff，确保位级完全相等；
- 核验 SQP 求解状态、外层迭代次数与目标函数评价次数的一致性。
"""
from pathlib import Path
import json, sys, csv
import numpy as np
from profile_paths import baseline_directory
OUT = Path(__file__).resolve().parent
BASE = baseline_directory()
RUN = OUT / ('预运行核验' if '--pilot' in sys.argv else 'B201_正式统计')
reports = []
issues = []
for scene in ['B201']:
    for method in ['cilqr', 'alilqr', 'sqp']:
        p = RUN / scene / method
        rows = list(csv.DictReader((p / 'cycles.csv').open(encoding='utf-8-sig')))
        old = list(csv.DictReader((BASE / scene / method / 'cycles.csv').open(encoding='utf-8-sig')))
        expected = 3 if '--pilot' in sys.argv else {'B201': 400, 'B207': 300, 'B301': 200}[scene]
        assert len(rows) == expected, (scene, method, len(rows), expected)
        stats = {'scene': scene, 'method': method, 'cycles': len(rows), 'all_executed': all((r['executed'] == 'True' for r in rows)), 'max_abs_differences': {}, 'bitwise_equal': True}
        for frame, r in enumerate(rows):
            with np.load(p / 'frames' / f'frame_{frame:06d}.npz') as new, np.load(BASE / scene / method / 'frames' / f'frame_{frame:06d}.npz') as ref:
                for key in ['state_before', 'state_after', 'planned_states', 'controls', 'obstacles_current']:
                    diff = float(np.max(np.abs(new[key] - ref[key]), initial=0.0))
                    stats['max_abs_differences'][key] = max(stats['max_abs_differences'].get(key, 0.0), diff)
                    stats['bitwise_equal'] &= bool(np.array_equal(new[key], ref[key]))
            if method == 'sqp':
                for key in ['slsqp_status', 'slsqp_iterations', 'objective_calls']:
                    if r[key] != old[frame][key]:
                        issues.append([scene, method, frame, key, r[key], old[frame][key]])
        attempts = [json.loads(line) for line in (p / 'attempts.jsonl').read_text(encoding='utf-8').splitlines()]
        for a in attempts:
            result = a['result_metadata']
            profile = result['timing_profile']
            wall = result['solver_wall_ms']
            if method != 'sqp':
                its = profile['iteration_timings']
                assert profile['iterations'] == len(its) > 0
                assert profile['outer_iterations'] == max((t['outer'] for t in its))
                for t in its:
                    assert t['wall_ms'] + 1e-06 >= sum((t[k] for k in ['linearization_ms', 'derivatives_ms', 'backward_ms', 'forward_ms', 'cost_ms']))
                parts = sum((profile[k] for k in ['linearization_ms', 'derivatives_ms', 'backward_ms', 'forward_ms', 'cost_ms', 'al_update_ms']))
                assert wall + 1e-06 >= parts
            else:
                assert profile['other_minimize_ms'] >= -1e-06
        reports.append(stats)
report = {'comparisons': reports, 'sqp_count_status_mismatches': issues, 'all_bitwise_equal': all((r['bitwise_equal'] for r in reports)), 'timing_partition_checks': 'passed'}
(RUN / 'output_equivalence_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
if issues or not report['all_bitwise_equal']:
    raise SystemExit(2)