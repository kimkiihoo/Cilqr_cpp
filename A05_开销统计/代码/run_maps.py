"""原版三方法 (R-CILQR, AL-iLQR, SQP-NMPC) 在全场景下的滚动时域仿真推进主程序。"""
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
import traceback
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'
import numpy as np
import scipy
from original_backend import HERE, ROOT, OriginalBackend
from single_state import serial
from sqp_nmpc import solve_prepared
BASELINES = {'B201': HERE.parent / 'B201_正式统计/B201'}
FULL_STEPS = {'B201': 400, 'B207': 300, 'B301': 200}
SPEEDS = {'B201': 5.0, 'B207': 6.0, 'B301': 15.0}
FOLDERS = {'r': 'cilqr', 'al': 'alilqr', 'sqp': 'sqp'}

def sha(path):
    """计算指定文件的 SHA-256 哈希校验码。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, data):
    """将数据格式化写入 JSON 文件。"""
    path.write_text(json.dumps(serial(data), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def absolute_obstacles(initial, clock_s):
    """根据当前仿真时间推演动态障碍物的全局绝对坐标。"""
    out = initial.copy()
    for i, obstacle in enumerate(initial):
        out[i, 0] = obstacle[0] + clock_s * obstacle[3] * math.cos(obstacle[2])
        out[i, 1] = obstacle[1] + clock_s * obstacle[3] * math.sin(obstacle[2])
    return out

def snapshot(output):
    """保存当前周期的完整状态、预测轨迹、控制量与障碍物快照至 NPZ 文件。"""
    dest = output / 'reproduction'
    dest.mkdir()
    names = ['bridge.cpp', 'CMakeLists.txt', 'original_backend.py', 'sqp_nmpc.py', 'run_maps.py', 'single_state.py', 'sqp_profile.py', 'source_manifest.json']
    for name in names:
        shutil.copy2(HERE / name, dest / name)
    shutil.copytree(HERE / 'generated', dest / 'generated')
    (dest / 'build').mkdir()
    for name in ['original_model_bridge.dll', 'libgcc_s_seh-1.dll', 'libstdc++-6.dll', 'libwinpthread-1.dll']:
        shutil.copy2(HERE / 'build' / name, dest / 'build' / name)
    hashes = {str(p.relative_to(dest)).replace('\\', '/'): sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}
    os.environ['ORIGINAL_MODEL_BRIDGE_DLL'] = str((dest / 'build/original_model_bridge.dll').resolve())
    return hashes
CSV_FIELDS = ['frame_index', 'cycle', 'plan_time_s', 'time_s', 'executed', 'velocity', 'gamma_dot', 'x', 'y', 'theta', 'gamma', 'before_x', 'before_y', 'before_theta', 'before_gamma', 'effective_desired_speed', 'r_effective_speed', 'al_effective_speed', 'solve_time_ms', 'solver_wall_ms', 'native_clock_ms', 'prepare_wall_ms', 'initialization_and_minimize_ms', 'post_evaluation_ms', 'scipy_success', 'slsqp_status', 'slsqp_iterations', 'objective_calls', 'optimizer_attempts', 'max_original_residual', 'hard_control_max_residual', 'recomputed_cost', 'finite_control_sequence', 'warmstart_committed', 'message']

def run_method(output, mapname, method, steps, scene, config, config_source):
    """在指定地图场景下运行指定优化算法的完整闭环仿真推演。"""
    folder = output / mapname / FOLDERS[method]
    folder.mkdir(parents=True)
    (folder / 'frames').mkdir()
    write_json(folder / 'effective_config.json', config)
    state = np.asarray(scene['start'], dtype=float)
    initial_obs = np.asarray(scene['initial_obstacles'], dtype=float).reshape(-1, 4)
    actual_states, plans, controls_log, rows = ([state.copy()], [], [], [])
    stop_reason = 'fixed_window_completed'
    with OriginalBackend(scene['path'], config) as backend, (folder / 'cycles.csv').open('w', newline='', encoding='utf-8') as csvfile, (folder / 'attempts.jsonl').open('w', encoding='utf-8') as attempts, (folder / 'cycles.jsonl').open('w', encoding='utf-8') as details:
        meta0 = backend.metadata
        assert meta0['N'] == 80 and meta0['dt'] == 0.1
        assert meta0['configured_speed'] == SPEEDS[mapname]
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()
        write_json(folder / 'metadata.json', {'map': mapname, 'method': method, 'expected_frames': steps, 'config_source': str(config_source), 'initial_cpp_parameters': meta0, 'sqp_options': {'maxiter': 100, 'ftol': 1e-06} if method == 'sqp' else None, 'original_solver': method in ['r', 'al'], 'native_async_preserved': method == 'al'})
        for frame in range(steps):
            before = state.copy()
            plan_time = frame * meta0['dt']
            obstacles = absolute_obstacles(initial_obs, plan_time)
            after_obstacles = absolute_obstacles(initial_obs, (frame + 1) * meta0['dt'])
            attempt = {'frame_index': frame, 'attempt_index': 0, 'state_before': before, 'plan_time_s': plan_time, 'obstacles_current': obstacles}
            row = {key: None for key in CSV_FIELDS}
            row.update(frame_index=frame, cycle=frame, plan_time_s=plan_time, time_s=(frame + 1) * meta0['dt'], executed=False, optimizer_attempts=1, before_x=before[0], before_y=before[1], before_theta=before[2], before_gamma=before[3], warmstart_committed=False)
            try:
                if method == 'sqp':
                    backend.prepare(state, obstacles, target='r')
                    result = solve_prepared(backend, maxiter=100, ftol=1e-06)
                    row.update(scipy_success=result['scipy_success'], slsqp_status=result['status'], slsqp_iterations=result['iterations'], objective_calls=result['objective_calls'], prepare_wall_ms=result['prepare_wall_ms'], initialization_and_minimize_ms=result['initialization_and_minimize_ms'], max_original_residual=result['max_residual'], hard_control_max_residual=result['hard_control_max_residual'], recomputed_cost=result['r_exact_cost'], message=result['message'])
                    diagnostic_residuals = result['residuals']
                    references = result['matched_references']
                else:
                    result = backend.solve_original(method, state, obstacles)
                    row.update(native_clock_ms=result['native_fields']['solve_time_ms'], max_original_residual=result['original_geometry_max_residual'], recomputed_cost=result['recomputed_native_cost'], message='Original solver returned; AL native convergence placeholders unavailable' if method == 'al' else 'Original solver returned')
                    diagnostic_residuals = np.array([result['original_geometry_max_residual']])
                    references = np.empty((0, 3))
                meta = result['metadata']
                row.update(solver_wall_ms=result['solver_wall_ms'], solve_time_ms=result['solver_wall_ms'], effective_desired_speed=meta['al_effective_speed'] if method == 'al' else meta['r_effective_speed'], r_effective_speed=meta['r_effective_speed'], al_effective_speed=meta['al_effective_speed'])
                candidate = np.asarray(result['controls'])
                predicted = np.asarray(result['states'])
                finite_sequence = bool(np.isfinite(candidate).all())
                row['finite_control_sequence'] = finite_sequence
                row['velocity'], row['gamma_dot'] = candidate[0].tolist()
                attempt.update(returned=True, finite_first_control=bool(np.isfinite(candidate[0]).all()), finite_control_sequence=finite_sequence, result_metadata={k: v for k, v in result.items() if k not in ['states', 'controls', 'matched_references', 'residuals']})
                plans.append(predicted.copy())
                controls_log.append(candidate.copy())
                npz_path = folder / 'frames' / f'frame_{frame:06d}.npz'

                def save_arrays(after):
                    np.savez_compressed(npz_path, state_before=before, planned_states=predicted, controls=candidate, state_after=after, obstacles_current=obstacles, obstacles_after=after_obstacles, original_residuals=diagnostic_residuals, matched_references=references)
                    attempt['arrays_file'] = str(npz_path.relative_to(folder)).replace('\\', '/')
                    attempt['arrays_sha256'] = sha(npz_path)
                save_arrays(np.full(4, np.nan))
                if np.isfinite(candidate[0]).all():
                    evaluation_started = time.perf_counter()
                    after = backend.step(state, candidate[0])
                    if not np.isfinite(after).all():
                        raise FloatingPointError('original model returned nonfinite executed state')
                    state = after
                    row.update(executed=True, x=state[0], y=state[1], theta=state[2], gamma=state[3])
                    actual_states.append(state.copy())
                    save_arrays(after)
                    if method == 'sqp' and finite_sequence:
                        backend.commit(candidate)
                        row['warmstart_committed'] = True
                    row['post_evaluation_ms'] = (time.perf_counter() - evaluation_started) * 1000
                else:
                    stop_reason = 'nonfinite_first_control'
                    after = np.full(4, np.nan)
            except Exception as error:
                stop_reason = 'execution_or_postprocessing_exception' if attempt.get('returned') else 'exception_without_executable_return'
                row['message'] = repr(error)
                attempt.setdefault('returned', False)
                attempt.update(exception=repr(error), traceback=traceback.format_exc())
            rows.append(row)
            writer.writerow(serial(row))
            csvfile.flush()
            attempts.write(json.dumps(serial(attempt), ensure_ascii=False, allow_nan=False) + '\n')
            attempts.flush()
            details.write(json.dumps(serial({'summary': row, 'cpp_metadata': backend.metadata, 'arrays_file': attempt.get('arrays_file')}), ensure_ascii=False, allow_nan=False) + '\n')
            details.flush()
            write_json(folder / 'progress.json', {'recorded_attempts': len(rows), 'executed_frames': len(actual_states) - 1, 'expected_frames': steps, 'last_status': row.get('slsqp_status'), 'stop_reason': stop_reason})
            print(f"{mapname} {method}: {frame + 1}/{steps}, executed={row['executed']}, wall_ms={row['solver_wall_ms']}, status={row['slsqp_status']}, residual={row['max_original_residual']}", flush=True)
            if stop_reason != 'fixed_window_completed':
                break
        np.savez_compressed(folder / 'trajectories.npz', planned_states=np.asarray(plans), controls=np.asarray(controls_log), actual_states=np.asarray(actual_states), initial_obstacles=initial_obs)
        backend.preserve_logs(folder / 'native_logs')
    walls = np.array([r['solver_wall_ms'] for r in rows if r['solver_wall_ms'] is not None], dtype=float)
    summary = {'map': mapname, 'method': method, 'expected_frames': steps, 'recorded_attempts': len(rows), 'executed_frames': len(actual_states) - 1, 'completed_fixed_window': len(actual_states) - 1 == steps, 'stop_reason': stop_reason, 'all_attempts_retained': True, 'scipy_success_count': sum((r['scipy_success'] is True for r in rows)) if method == 'sqp' else None, 'scipy_status_counts': {str(status): sum((r['slsqp_status'] == status for r in rows)) for status in sorted(set((r['slsqp_status'] for r in rows if r['slsqp_status'] is not None)))}, 'mean_solver_wall_ms': float(walls.mean()) if len(walls) else None, 'max_solver_wall_ms': float(walls.max()) if len(walls) else None}
    write_json(folder / 'summary.json', summary)
    return summary

def main():
    """跨地图场景与多算法基准闭环仿真评测主函数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps', nargs='+', choices=list(BASELINES), default=['B201'])
    parser.add_argument('--methods', nargs='+', choices=list(FOLDERS), default=['r', 'al', 'sqp'])
    parser.add_argument('--mode', choices=['pilot', 'full'], default='pilot')
    parser.add_argument('--pilot-cycles', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.pilot_cycles < 1:
        raise ValueError('pilot cycles must be positive')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite ' + str(output))
    output.mkdir(parents=True)
    sources = {}
    for mapname in args.maps:
        base = BASELINES[mapname]
        scene_file = base / 'scene/scene.json'
        scene = json.loads(scene_file.read_text(encoding='utf-8'))
        cfgs = {}
        for method in args.methods:
            config_file = base / ('alilqr' if method == 'al' else 'cilqr') / 'effective_config.json'
            effective = json.loads(config_file.read_text(encoding='utf-8'))
            cfgs[method] = ({key: effective[key] for key in ['main', 'ilqr', 'hybrid_astar']}, config_file)
        sources[mapname] = (scene, cfgs)
        scene_out = output / mapname / 'scene'
        scene_out.mkdir(parents=True)
        shutil.copy2(scene_file, scene_out / 'scene.json')
    hashes = snapshot(output)
    protocol = {'mode': args.mode, 'maps': args.maps, 'methods': args.methods, 'steps': {m: args.pilot_cycles if args.mode == 'pilot' else FULL_STEPS[m] for m in args.maps}, 'N': 80, 'dt_s': 0.1, 'candidate_reference_speeds': {m: SPEEDS[m] for m in args.maps}, 'scope': 'New benchmark using the only handed-over code; candidate speed labels retained; not exact historical-figure restoration.', 'sqp_options': {'maxiter': 100, 'ftol': 1e-06}, 'optimizer_attempts_per_cycle': 1, 'execution': 'Apply returned first control iff finite, even when SLSQP reports failure; no clipping or safety override. Finite entire SQP sequences are committed for original warm-start shifting. Nonfinite first control or exception is retained and stops that run.', 'obstacle_clock': 'Each cycle obstacles are recomputed from frozen initial states and absolute plan_time_s; no cumulative obstacle integration drift.', 'target_speed': 'Each SQP cycle calls the original R preparation once; its member desired speed accumulates the original near-goal decay. AL retains original behavior.', 'timing': {'solver_wall_ms': 'Original R/AL: steady_clock around full native solve, including local preparation, nominal initialization, optimization, and native logging. SQP: sum of steady_clock original-R preparation plus perf_counter nominal/constraint initialization and minimize. Both exclude obstacle prediction, supplementary cost/residual evaluation, execution, commit and serialization.', 'solve_time_ms': 'Exact alias of solver_wall_ms for plot compatibility.', 'native_clock_ms': 'Original returned Solution internal clock field; not the comparison timer, AL placeholder zero.'}, 'constraints': 'Original R and AL preserve different hard/soft constraint enforcement; SQP uses exact R penalty objective and original geometric residual definitions, R fixed control envelope and predicted internal acceleration bounds; it does not emulate the R iteration-dependent first-speed trust interval.', 'threading': 'Methods run sequentially; native AL std::async preserved. OMP/BLAS environment limits do not imply AL single-thread execution.', 'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'platform': platform.platform(), 'cpu': '12th Gen Intel(R) Core(TM) i7-12700H', 'profiling': 'Measurement-only timers; native AL asynchronous workers and native R logging retained; per-iteration costs measured over actual inner-loop bodies. SQP core includes QP, BFGS and line-search bookkeeping.', 'thread_limits': {v: os.environ[v] for v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']}}, 'reproduction_sha256': hashes, 'inputs': {m: {'scene_source': str(BASELINES[m] / 'scene/scene.json'), 'scene_sha256': sha(BASELINES[m] / 'scene/scene.json'), 'config_sources': {method: {'path': str(pair[1]), 'sha256': sha(pair[1])} for method, pair in sources[m][1].items()}} for m in args.maps}}
    write_json(output / 'protocol.json', protocol)
    summaries = []
    for mapname in args.maps:
        scene, cfgs = sources[mapname]
        for method in args.methods:
            summaries.append(run_method(output, mapname, method, protocol['steps'][mapname], scene, *cfgs[method]))
            write_json(output / 'summary.json', summaries)
    print('FINISHED ' + str(output), flush=True)
if __name__ == '__main__':
    main()