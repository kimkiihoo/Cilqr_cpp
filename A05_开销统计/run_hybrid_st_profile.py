"""Hybrid A* + ST* 两阶段算法计算耗时评测与输出等价性比对主脚本。

计时口径说明:
- 第一阶段 (几何路径准备): 单独记录地图加载、Hybrid A* 启发式搜索与 QP 样条平滑的总耗时；
- 第二阶段 (ST 速度规划): 单独记录空间限速剖面推导与时空状态图 A* 搜索的实际耗时；
- 两阶段总耗时为一次性全局规划开销，不与 MPC 滚动时域单周期求解耗时混为一谈。
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import numpy as np
from profile_paths import baseline_directory, reference_run_directory
HERE = Path(__file__).resolve().parent

def read_json(path):
    """读取 JSON 配置文件。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    """写入 JSON 数据文件。"""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def sha(path):
    """计算文件哈希校验码。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_csv(path, rows):
    """写入 CSV 数据表。"""
    keys = list(dict.fromkeys((k for row in rows for k in row)))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def equivalent_outputs(reference, current):
    """比对当前运行结果与基准结果的逐值一致性。"""
    comparison = {}
    with np.load(reference / 'trajectories.npz') as expected, np.load(current / 'trajectories.npz') as actual:
        if set(expected.files) != set(actual.files):
            raise ValueError('Trajectory archive fields differ from the reference run')
        for key in expected.files:
            same = np.array_equal(expected[key], actual[key])
            comparison[key] = {'bitwise_equal': bool(same), 'max_abs_error': float(np.max(np.abs(expected[key] - actual[key]), initial=0.0)) if expected[key].shape == actual[key].shape else None}
    return {'passed': all((v['bitwise_equal'] for v in comparison.values())), 'arrays': comparison}

def export_combined_summary(result_dir, summary):
    """导出四方法计算耗时与阶段开销汇总表。"""
    target = HERE / 'B201_结果汇总'
    old = read_json(target / 'summary.json')
    rows = []
    for key in ('cilqr', 'alilqr', 'sqp'):
        source = old[key]
        rows.append({'map': 'B201', 'method': source['method'], 'measurement_batch': 'B201_正式统计', 'planning_mode': 'receding_horizon_8s', 'timing_unit': 'one MPC planning cycle', 'measured_calls': source['cycles'], 'mean_mpc_cycle_ms': source['solver_wall_mean_ms'], 'max_mpc_cycle_ms': source['solver_wall_max_ms'], 'sum_mpc_calls_over_40s_ms': source['solver_wall_mean_ms'] * source['cycles'], 'mean_path_preparation_ms': None, 'mean_global_st_search_ms': None, 'mean_two_stage_total_ms': None, 'mean_optimizer_iterations': source['iterations_mean'], 'mean_st_expanded_nodes': None, 'source': str((HERE / 'B201_正式统计/B201' / key).relative_to(HERE))})
    rows.append({'map': 'B201', 'method': 'Hybrid A* + ST*', 'measurement_batch': str(result_dir.relative_to(HERE)) if result_dir.is_relative_to(HERE) else str(result_dir), 'planning_mode': 'one global plan for the 40s window', 'timing_unit': 'one complete two-stage plan', 'measured_calls': summary['repetitions'], 'mean_mpc_cycle_ms': None, 'max_mpc_cycle_ms': None, 'sum_mpc_calls_over_40s_ms': None, 'mean_path_preparation_ms': summary['path_preparation_mean_ms'], 'mean_global_st_search_ms': summary['st_search_mean_ms'], 'mean_speed_limit_preparation_ms': summary.get('speed_limit_preparation_mean_ms'), 'mean_st_graph_search_ms': summary.get('graph_search_mean_ms'), 'mean_other_st_initialization_ms': summary.get('other_st_initialization_mean_ms'), 'mean_two_stage_total_ms': summary['two_stage_mean_ms'], 'mean_optimizer_iterations': None, 'mean_st_expanded_nodes': summary['expanded_nodes_mean'], 'source': str(result_dir / 'summary.json')})
    write_csv(target / '06_四方法开销口径汇总.csv', rows)
    write_csv(target / '07_HybridA_ST阶段开销.csv', summary['measurements'])
    write_json(target / 'hybrid_st_summary.json', summary)
    text = f"B201 新增 Hybrid A* + ST* 开销统计\n\nHybrid A* + ST* 由既有 Hybrid A* 及 QP 路径平滑，与独立时空图搜索速度规划组成。\n两阶段均直接复用最新 B201 实验入口；复跑所得轨迹、速度及原模型执行状态与参考结果逐值一致。\n\n测量次数：{summary['repetitions']} 次完整规划。\n几何路径准备平均耗时：{summary['path_preparation_mean_ms']:.6f} ms。\nST* 速度图搜索平均耗时：{summary['st_search_mean_ms']:.6f} ms。\n两阶段总耗时平均值：{summary['two_stage_mean_ms']:.6f} ms。\nST* 平均展开节点数：{summary['expanded_nodes_mean']:.1f}。\n其中几何可行速度上限准备：{summary.get('speed_limit_preparation_mean_ms', 0.0):.6f} ms；图搜索：{summary.get('graph_search_mean_ms', 0.0):.6f} ms；其余ST初始化及调用开销：{summary.get('other_st_initialization_mean_ms', 0.0):.6f} ms。\n本次40 s窗口结束时，图搜索终点是否到达：{all((r.get('reached_graph_goal', False) for r in summary['measurements']))}。安全的部分路径计划不记作终点到达。\n\n计时范围\n几何路径准备包括已有程序启动、配置与地图读取、Hybrid A*、QP 平滑、场景初始化与文件导出；不包含 Python 文件准备和生成结果核对。\nST* 时间沿用实验入口的完整速度规划阶段测量，包含几何可行速度上限准备、图准备、障碍检查、搜索及结果构造。速度上限准备、图搜索和其余初始化时间单列且不重复相加。\n两阶段总耗时为单列几何路径准备与 ST* 调用的和。跟踪器执行和统计、绘图均不计入这两个阶段。\n\n比较口径\n原 R-CILQR、AL-iLQR、SQP-NMPC 仍使用原 B201_正式统计 的400周期数据，每次预测窗口8 s。\nHybrid A* + ST* 是针对40 s场景的一次全局计划，完整规划次数不等于400次在线重规划。\n06_四方法开销口径汇总.csv 为两种计时口径保留不同列，未把全局搜索时间除以400或冒充每周期耗时。\n图节点展开次数单列，不称为 iLQR 内迭代次数；已有三方法分项和正文文件保持原口径。\n这些是当前实现与机器上的时间测量，源文件哈希、参数、完整输出和一致性核验见本次结果目录。\n\n复现\npy -3.11 -X utf8 run_hybrid_st_profile.py --output <新的结果目录>\n可用 --reference-run 指定新的 B201 Hybrid A* + ST* 参考结果，--baseline-dir 指定冻结三方法基准，--repetitions 指定完整重复次数。\n默认只测1次完整规划，与已有一次完整场景剖析的协议一致；多次重复时不会覆盖第一次或挑选最快结果。\n"
    (target / '08_HybridA_ST开销说明.txt').write_text(text, encoding='utf-8-sig')

def main():
    """执行 Hybrid A* + ST* 耗时评测与核验主函数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path)
    parser.add_argument('--reference-run', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--no-summary-export', action='store_true', help='Do not refresh the combined summary files')
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error('--repetitions must be positive')
    baseline = (args.baseline_dir or baseline_directory()).resolve()
    reference = (args.reference_run or reference_run_directory()).resolve()
    destination = (args.output or HERE / 'B201_HybridA_ST_开销统计').resolve()
    reference_summary = read_json(reference / 'summary.json')
    if not reference_summary['planning_success']:
        raise ValueError('The selected reference run did not produce a valid ST schedule')
    runner = baseline / 'reproduction/run_hybrid_astar_st.py'
    destination.mkdir(parents=True, exist_ok=False)
    source_names = ['run_hybrid_astar_st.py', 'hybrid_path_stage.py', 'st_speed_search.py', 'original_backend.py']
    if (baseline / 'reproduction/geometric_speed_limits.py').is_file():
        source_names.append('geometric_speed_limits.py')
    source_hashes = {name: sha(baseline / 'reproduction' / name) for name in source_names}
    protocol = {'map': 'B201', 'method': 'Hybrid A* + ST*', 'reference_run': str(reference), 'reference_summary_sha256': sha(reference / 'summary.json'), 'reference_plan_sha256': sha(reference / 'st_search.json'), 'baseline_directory': str(baseline), 'repetitions': args.repetitions, 'warmup_calls': 0, 'all_attempts_retained': True, 'runner': str(runner), 'source_sha256': source_hashes, 'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform(), 'timing_mode': 'Independent complete 40-second global plans. Path preparation and ST search measured separately.', 'not_mpc_cycle_time': True}
    write_json(destination / 'protocol.json', protocol)
    shutil.copy2(Path(__file__), destination / 'run_hybrid_st_profile_source.py')
    rows, audits = ([], [])
    environment = os.environ.copy()
    environment['PYTHONUTF8'] = '1'
    for k in range(args.repetitions):
        run = destination / ('run_%03d' % (k + 1))
        command = [sys.executable, '-X', 'utf8', str(runner), '--baseline', str(baseline), '--output', str(run)]
        with (destination / ('run_%03d.log' % (k + 1))).open('w', encoding='utf-8') as log:
            completed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            write_json(destination / 'failed_attempt.json', {'attempt': k + 1, 'returncode': completed.returncode, 'command': command})
            raise RuntimeError('B201 profiling run failed; see ' + str(destination / ('run_%03d.log' % (k + 1))))
        summary = read_json(run / 'summary.json')
        graph = read_json(run / 'st_search.json')
        check = equivalent_outputs(reference, run)
        check['attempt'] = k + 1
        audits.append(check)
        write_json(destination / 'output_equivalence_audit.json', {'passed': all((a['passed'] for a in audits)), 'attempts': audits})
        if not check['passed']:
            raise ValueError('Returned trajectories differ from the selected reference; update the reference explicitly before profiling')
        path_ms, st_ms = (summary['path_stage_wall_ms'], summary['st_search_wall_ms'])
        if path_ms is None or not np.isfinite([path_ms, st_ms]).all() or min(path_ms, st_ms) <= 0:
            raise ValueError('The run must measure both planning stages')
        row = {'attempt': k + 1, 'path_preparation_ms': path_ms, 'st_search_ms': st_ms, 'two_stage_total_ms': path_ms + st_ms, 'expanded_nodes': graph['stats']['expanded_nodes'], 'generated_nodes': graph['stats']['generated_nodes'], 'arrival_time_s': summary['arrival_time_s'], 'success': summary['planning_success'], 'reason': summary['reason'], 'reached_graph_goal': summary.get('reached_graph_goal', False), 'planned_remaining_distance_m': summary.get('planned_remaining_distance_m'), 'output_equivalence_passed': check['passed']}
        for key, value in summary.items():
            if key.endswith('_ms') and key not in ('path_stage_wall_ms', 'st_search_wall_ms', 'follower_total_wall_ms'):
                row[key] = value
        rows.append(row)
        write_csv(destination / 'measurements.csv', rows)
        print(f'B201 complete plan {k + 1}/{args.repetitions}: path {path_ms:.3f} ms; ST {st_ms:.3f} ms', flush=True)
    for name, digest in source_hashes.items():
        if sha(baseline / 'reproduction' / name) != digest:
            raise RuntimeError('Planner source changed during timing: ' + name)
    summary = {'map': 'B201', 'method': 'Hybrid A* + ST*', 'repetitions': len(rows), 'output_equivalence_passed': True, 'path_preparation_mean_ms': float(np.mean([r['path_preparation_ms'] for r in rows])), 'st_search_mean_ms': float(np.mean([r['st_search_ms'] for r in rows])), 'two_stage_mean_ms': float(np.mean([r['two_stage_total_ms'] for r in rows])), 'expanded_nodes_mean': float(np.mean([r['expanded_nodes'] for r in rows])), 'speed_limit_preparation_mean_ms': float(np.mean([r.get('geometric_speed_limit_wall_ms', 0.0) for r in rows])), 'graph_search_mean_ms': float(np.mean([r.get('graph_search_wall_ms', r['st_search_ms']) for r in rows])), 'other_st_initialization_mean_ms': float(np.mean([r['st_search_ms'] - r.get('geometric_speed_limit_wall_ms', 0.0) - r.get('graph_search_wall_ms', r['st_search_ms']) for r in rows])), 'measurements': rows, 'protocol': protocol}
    write_json(destination / 'summary.json', summary)
    if not args.no_summary_export:
        export_combined_summary(destination, summary)
    print(destination, flush=True)
if __name__ == '__main__':
    main()