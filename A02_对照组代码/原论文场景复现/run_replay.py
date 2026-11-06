"""原始论文交接算法在三张基准地图 (B301, B201, B207) 上的完整复跑驱动脚本。

主要流程:
  1. 调用 adapt_original 构建无交互、可复现的 paper_replay.exe 二进制文件;
  2. 预备并冻结测试场景: 导出全局参考路径 (frozen_global_path.txt) 与动态障碍物初态 (frozen_obstacles.txt);
  3. 分别在相同冻结场景下按顺序执行 C-iLQR 与 AL-iLQR 滚动规划仿真;
  4. 采集各周期 (0.1 s) 车辆状态 [x, y, theta, gamma]、控制输入 [v, omega]、单步耗时、迭代次数与收敛标志;
  5. 校验非有限数值 (NaN/Inf)，生成全量 cycles.csv 与 run_audit.json 审计元数据。

物理量纲与状态定义:
  状态量 x = [x (m), y (m), theta (rad), gamma (rad)]^T (车辆铰接角)
  控制量 u = [velocity (m/s), gamma_dot (rad/s)]^T (铰接角速率)
  仿真步长 dt = 0.1 s, 规划步长 cycle 从 0 开始索引。
"""
from pathlib import Path
import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from adapt_original import adapt, sha
from project_paths import project_root, results_dir


def write_json(path, obj):
    """以 UTF-8 编码将嵌套字典/列表对象格式化写入 JSON 文件。
    
    参数:
      path (Path): 目标 JSON 文件路径。
      obj (Any): 待序列化数据对象。
    """
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def command(args, cwd, log, env):
    """执行外部子进程命令并将标准输出与错误重定向至指定日志文件。
    
    参数:
      args (list): 命令行参数列表。
      cwd (Path): 子进程执行工作目录。
      log (Path): 输出日志文件路径。
      env (dict): 子进程环境变量字典。
      
    返回:
      int: 进程退出码 (0 表示执行成功)。
    """
    with log.open('w', encoding='utf-8') as f:
        p = subprocess.run([str(x) for x in args], cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return p.returncode


def read_points(path):
    """读取冻结场景数据文件中的多维浮点数列表。
    
    参数:
      path (Path): 数据文件路径 (首行为点数整数，后续行为空格分隔的坐标)。
      
    返回:
      list[list[float]]: 浮点数组列表。
    """
    lines = path.read_text().splitlines()
    rows = [[float(v) for v in line.split()] for line in lines[1:]]
    assert len(rows) == int(lines[0])
    return rows


def nonfinite_fields(obj, path=''):
    """递归检测 JSON 结构中是否存在非有限浮点数 (NaN, +Infinity, -Infinity)。
    
    参数:
      obj: 待检测的嵌套数据结构 (dict, list, str, etc.)。
      path (str): 当前字段的层次路径标识。
      
    返回:
      list[str]: 所有包含非有限值的字段路径列表。
    """
    if isinstance(obj, str) and obj in ['NaN', '+Infinity', '-Infinity']:
        return [path]
    if isinstance(obj, dict):
        return sum((nonfinite_fields(v, path + '.' + k) for k, v in obj.items()), [])
    if isinstance(obj, list):
        return sum((nonfinite_fields(v, f'{path}[{i}]') for i, v in enumerate(obj)), [])
    return []


def parse_steps_by_map(entries, maps):
    """解析命令行针对不同地图配置的独立步长覆盖参数 (MAP=COUNT 格式)。
    
    参数:
      entries (list[str]|None): 命令行输入的 MAP=COUNT 字符串列表。
      maps (list[str]): 当前选定的有效地图名称列表。
      
    返回:
      dict[str, int]: 地图名称映射到正整数步数的字典。
    """
    overrides = {}
    for entry in entries or []:
        match = re.fullmatch(r'([^=]+)=([0-9]+)', entry)
        if not match or int(match.group(2)) < 1:
            raise ValueError('Each --steps-by-map item must be MAP=POSITIVE_INTEGER: ' + entry)
        mapname, count = match.group(1), int(match.group(2))
        if mapname not in maps:
            raise ValueError('--steps-by-map names a map not selected by --maps: ' + mapname)
        if mapname in overrides:
            raise ValueError('Duplicate --steps-by-map entry: ' + mapname)
        overrides[mapname] = count
    return overrides


def main():
    """复跑主程序入口: 解析参数、编译 C++ 程序、逐场景执行 C-iLQR 与 AL-iLQR 并汇总数据。"""
    here = Path(__file__).resolve().parent
    root = project_root(here)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['handoff', 'paper_table1'], default='handoff')
    p.add_argument('--output', type=Path)
    p.add_argument('--maps', nargs='+', choices=['B301', 'B201', 'B207'], default=['B301', 'B201', 'B207'])
    step_options = p.add_mutually_exclusive_group()
    step_options.add_argument('--steps', type=int, default=None)
    step_options.add_argument('--steps-by-map', nargs='+', metavar='MAP=COUNT',
                              help='Per-map positive step counts; only maps selected by --maps are accepted')
    p.add_argument('--speed', type=float, default=None)
    p.add_argument('--build-only', action='store_true')
    p.add_argument('--compiler', default='C:/msys64/mingw64/bin/g++.exe')
    p.add_argument('--ninja', default='C:/msys64/mingw64/bin/ninja.exe')
    a = p.parse_args()
    if a.steps is not None and a.steps < 1:
        raise ValueError('steps must be positive')
    try:
        steps_by_map = parse_steps_by_map(a.steps_by_map, a.maps)
    except ValueError as error:
        p.error(str(error))
    if a.speed is not None and a.speed <= 0:
        raise ValueError('speed must be positive')
    original = adapt(root, here / 'source')
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env['PATH'] = str(Path(a.compiler).parent) + os.pathsep + env.get('PATH', '')
    build = here / 'build'
    build.mkdir(exist_ok=True)
    cmake = Path(a.compiler).parent / 'cmake.exe'
    if not cmake.is_file():
        cmake = shutil.which('cmake')
    config_cmd = [cmake, '--fresh', '-S', str(here), '-B', str(build), '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_COMPILER=' + a.compiler, '-DCMAKE_MAKE_PROGRAM=' + a.ninja]
    if command(config_cmd, here, build / 'configure.log', env):
        raise RuntimeError('Configure failed; see build/configure.log')
    if command([cmake, '--build', str(build), '--parallel', '2'], here, build / 'build.log', env):
        raise RuntimeError('Build failed; see build/build.log')
    exe = build / 'paper_replay.exe'
    if a.build_only:
        print(str(exe), flush=True)
        return
    out = (a.output or results_dir(root) / ('原论文场景复跑_' + time.strftime('%Y-%m-%d_%H%M%S'))).resolve()
    if out.exists():
        raise FileExistsError('Refusing to overwrite existing result: ' + str(out))
    out.mkdir(parents=True)
    main_config = json.loads((original / 'config/main.json').read_text(encoding='utf-8'))
    ilqr_config = json.loads((original / 'config/ilqr.json').read_text(encoding='utf-8'))
    ha_config = json.loads((original / 'config/hybrid_astar.json').read_text(encoding='utf-8'))
    if a.mode == 'paper_table1':
        ilqr_config['cilqr']['N'] = 80
    if a.speed is not None:
        ilqr_config['cost_weights']['desire_speed'] = a.speed
    provenance = dict(mode=a.mode, created_local=time.strftime('%Y-%m-%d %H:%M:%S'), python=platform.python_version(), platform=platform.platform(),
        compiler=a.compiler, build_type='Release', sequential_method_runs=True,
        environment_thread_limits={key: env[key] for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
        native_async_preserved=True,
        threading_note='Method runs are dispatched sequentially. Environment limits do not disable the original AL solver std::async tasks; the original native internal parallelism is preserved.',
        order=a.maps, steps_override=a.steps,
        steps_by_map_override=steps_by_map,
        effective_steps={name: a.steps if a.steps is not None else steps_by_map.get(name, main_config['run_configs'][name]['ITER']) for name in a.maps},
        speed_override=a.speed,
        executable_sha256=sha(exe), source_hashes=json.loads((here / 'source/source_hashes.json').read_text(encoding='utf-8')),
        interpretation='Original-map rerun with handed-over optimizer/model; not exact regeneration of historical paper numerical data.',
        paper_differences=['Handed-over N=50; manuscript Table 1 N=80. paper_table1 mode only switches N, not a complete original-paper parameter reconstruction.',
          'Handed-over default desired speed is 15 m/s and fixed run length is 200 steps; original Fig.7 spans approximately 400 steps and plotted speeds largely 0–5.5 m/s. Original run-specific reference speed is not inferred.',
          'This is fixed-length rolling kinematic planning, with no independent plant or goal-completion claim.',
          'Computed gamma_dot limits override values in ilqr.json as in original main.cpp. See effective_runtime.json.',
          'Stochastic fallback global-path samplers use seed 20261030; path and obstacles are frozen before paired solver runs.',
          'Timing retains original clock() boundary and includes the original solver logging; original AL internal std::async parallelism is preserved. Machine-dependent timings cannot reproduce historical paper values.',
          'Original ALILQRSolver::solve updates a class convergence flag but never copies converged/iterations/final_cost into its returned Solution. AL-iLQR false/0/0 export fields are unavailable placeholders, not 200 failed optimizations.'],
        frame_semantics={'cycle': 'zero-based optimization cycle', 'frame_index': 'same zero-based value as cycle in cycles.csv', 'time_s': 'post-application ego/obstacle time (cycle+1)*dt', 'plan_time_s': 'initial time of planned_trajectory, cycle*dt', 'control_sequence.gamma': 'articulation angular rate, rad/s (legacy key name)', 'nonfinite': 'nonfinite numbers are preserved as strings NaN / +Infinity / -Infinity and listed in run_audit.json'}, runs=[])
    write_json(out / 'manifest.json', provenance)
    shutil.copy2(here / 'source/adaptation.diff', out / 'adaptation.diff')
    shutil.copy2(here / 'replay_support.h', out / 'replay_support.h')
    shutil.copy2(exe, out / 'paper_replay.exe')
    for dll in ['libgcc_s_seh-1.dll', 'libstdc++-6.dll', 'libwinpthread-1.dll']:
        candidate = Path(a.compiler).parent / dll
        if candidate.exists():
            shutil.copy2(candidate, out / dll)

    def setup(directory, mapname):
        directory.mkdir(parents=True)
        (directory / 'config').mkdir()
        (directory / 'Maps/bitmap').mkdir(parents=True)
        mc = copy.deepcopy(main_config)
        mc['selected_map'] = mapname
        if a.steps is not None:
            mc['run_configs'][mapname]['ITER'] = a.steps
        elif mapname in steps_by_map:
            mc['run_configs'][mapname]['ITER'] = steps_by_map[mapname]
        write_json(directory / 'config/main.json', mc)
        write_json(directory / 'config/ilqr.json', ilqr_config)
        write_json(directory / 'config/hybrid_astar.json', ha_config)
        mp = original / 'Maps/bitmap' / f'{mapname}_global_map.json'
        shutil.copy2(mp, directory / 'Maps/bitmap' / mp.name)
        return mc

    for mapname in a.maps:
        print('Preparing frozen scene ' + mapname, flush=True)
        prep = out / mapname / 'scene'
        mc = setup(prep, mapname)
        code = command([exe, '--prepare-scene'], prep, prep / 'prepare.log', env)
        if code:
            raise RuntimeError(f'Scene preparation failed ({code}): {prep}')
        points = read_points(prep / 'frozen_global_path.txt')
        obstacles = read_points(prep / 'frozen_obstacles.txt')
        scene = dict(map=mapname, path=points, initial_obstacles=obstacles, path_columns=['x', 'y', 'heading'], obstacle_columns=['x', 'y', 'heading', 'speed'],
                   start=[mc['run_configs'][mapname][k] for k in ['start_x', 'start_y', 'start_theta']] + [0.0],
                   requested_obstacles=mc['run_configs'][mapname]['obstacle_count'], actual_obstacles=len(obstacles),
                   path_sha256=sha(prep / 'frozen_global_path.txt'), obstacles_sha256=sha(prep / 'frozen_obstacles.txt'), map_sha256=sha(prep / 'Maps/bitmap' / f'{mapname}_global_map.json'))
        write_json(prep / 'scene.json', scene)
        for solver in ['cilqr', 'alilqr']:
            directory = out / mapname / solver
            setup(directory, mapname)
            for name in ['frozen_global_path.txt', 'frozen_obstacles.txt']:
                shutil.copy2(prep / name, directory / name)
            print('Running ' + mapname + ' ' + solver, flush=True)
            start = time.perf_counter()
            code = command([exe, '--solver', solver], directory, directory / 'run.log', env)
            elapsed = time.perf_counter() - start
            frames = sorted((directory / 'outputs' / mapname / solver / 'data').glob(f'{solver}_data_*.json'), key=lambda f: int(f.stem.rsplit('_', 1)[1]))
            audit = dict(map=mapname, solver=solver, returncode=code, wall_total_s=elapsed, expected_frames=mc['run_configs'][mapname]['ITER'], frame_count=len(frames), nonfinite=[], converged=0,
                       convergence_fields_available=solver == 'cilqr', convergence_note='Raw exported count only. Original AL-iLQR Solution fields remain default false/0/0 and do not represent convergence outcomes.' if solver == 'alilqr' else 'Original CILQR exported stopping flag; not a feasibility or collision guarantee.',
                       path_identical=sha(directory / 'frozen_global_path.txt') == scene['path_sha256'], obstacles_identical=sha(directory / 'frozen_obstacles.txt') == scene['obstacles_sha256'])
            rows = []
            global_hash = None
            for index, f in enumerate(frames):
                data = json.loads(f.read_text(encoding='utf-8'))
                assert data['cycle'] == index
                ghash = hashlib.sha256(json.dumps(data['global_plan'], sort_keys=True).encode()).hexdigest()
                if global_hash is None:
                    global_hash = ghash
                assert global_hash == ghash
                nf = nonfinite_fields(data)
                if nf:
                    audit['nonfinite'].append(dict(frame=f.name, fields=nf))
                audit['converged'] += int(data['convergence_info']['converged'])
                row = dict(frame_index=index, cycle=index, time_s=data['time_s'], plan_time_s=data['plan_time_s'], **data['ego_vehicle'], **data['convergence_info'])
                row['velocity'] = data['control_sequence']['velocity'][0] if data['control_sequence']['velocity'] else None
                row['gamma_dot'] = data['control_sequence']['gamma'][0] if data['control_sequence']['gamma'] else None
                rows.append(row)
            audit['global_plan_frames_sha256'] = global_hash
            audit['completed_fixed_steps'] = code == 0 and len(frames) == audit['expected_frames']
            assert audit['path_identical'] and audit['obstacles_identical']
            if rows:
                with (directory / 'cycles.csv').open('w', newline='', encoding='utf-8-sig') as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
            effective = dict(main=mc, ilqr=ilqr_config, hybrid_astar=ha_config, runtime=json.loads((directory / 'effective_runtime.json').read_text()), solver=solver)
            write_json(directory / 'effective_config.json', effective)
            write_json(directory / 'run_audit.json', audit)
            provenance['runs'].append(audit)
            write_json(out / 'manifest.json', provenance)
            print(f'Finished {mapname} {solver}: {len(frames)} frames, nonfinite={len(audit["nonfinite"])}, exit={code}', flush=True)
        pair = provenance['runs'][-2:]
        assert pair[0]['global_plan_frames_sha256'] == pair[1]['global_plan_frames_sha256']
    provenance['all_fixed_steps_completed'] = all(r['completed_fixed_steps'] for r in provenance['runs'])
    write_json(out / 'manifest.json', provenance)
    print('Saved ' + str(out), flush=True)


if __name__ == '__main__':
    main()
