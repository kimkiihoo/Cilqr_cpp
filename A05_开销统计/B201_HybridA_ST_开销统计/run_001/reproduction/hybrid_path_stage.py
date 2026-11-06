"""第一阶段几何路径准备模块: 负责调用原生可执行程序生成或校验基准 Hybrid A* 与 QP 平滑路径。

本模块实现两阶段混合规划方法的第一阶段:
- 调用原版 C++ 程序加载 B201 矿区栅格地图，执行 Hybrid A* 路径搜索与 QP 样条平滑；
- 导出或验证无碰撞几何路径点序列与障碍物列表，单独记录该阶段的初始化与生成耗时。
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import numpy as np
from project_paths import project_root, original_dir, replay_executable

def _sha(path):
    """计算指定文件的 SHA-256 哈希校验字符串。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _write_json(path, values):
    """将数据结构以 UTF-8 编码格式化写入 JSON 文件。"""
    Path(path).write_text(json.dumps(values, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def _project_root(baseline_dir):
    """定位项目根目录路径。"""
    return project_root(baseline_dir)

def _read_rows(path, columns):
    """读取文本格式的矩阵数据行。"""
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    count = int(lines[0])
    values = np.asarray([[float(v) for v in line.split()] for line in lines[1:]], dtype=float).reshape(-1, columns)
    if len(values) != count or not np.isfinite(values).all():
        raise ValueError('Invalid generated rows in ' + str(path))
    return values

def prepare_path(baseline_dir: Path, output_dir: Path, regenerate=True) -> dict:
    """加载并校验基准几何路径，或者重新调用可执行程序生成并验证输出的一致性。

参数:
    baseline_dir (Path): 冻结基准结果根目录。
    output_dir (Path): 当前运行输出目录。
    regenerate (bool): 是否重新调用底层可执行程序运行第一阶段。

返回:
    dict: 包含几何路径点数组 path [m, m, rad] 与阶段计时元数据 path_stage 的字典。
"""
    baseline_dir, output_dir = (Path(baseline_dir).resolve(), Path(output_dir).resolve())
    scene_file = baseline_dir / 'B201/scene/scene.json'
    config_file = baseline_dir / 'B201/cilqr/effective_config.json'
    scene = json.loads(scene_file.read_text(encoding='utf-8'))
    configs = json.loads(config_file.read_text(encoding='utf-8'))
    if scene['map'] != 'B201' or configs['main']['selected_map'] != 'B201':
        raise ValueError('This additional benchmark only supports B201')
    frozen_path = np.asarray(scene['path'], dtype=float)
    frozen_obstacles = np.asarray(scene['initial_obstacles'], dtype=float).reshape(-1, 4)
    if frozen_path.ndim != 2 or frozen_path.shape[1] != 3 or len(frozen_path) < 2:
        raise ValueError('Frozen path must contain at least two x/y/heading rows')
    if not np.isfinite(frozen_path).all() or not np.isfinite(frozen_obstacles).all():
        raise ValueError('Frozen scene contains nonfinite values')
    stage = output_dir / 'path_stage'
    stage.mkdir(parents=True, exist_ok=False)
    metadata = {'map': 'B201', 'method': 'existing Hybrid A* + existing QP path smoothing', 'mode': 'regenerate_and_verify' if regenerate else 'reuse_frozen_path', 'stage_wall_ms': None, 'timing_scope': 'Executable startup, map/config loading, Hybrid A* search, QP smoothing, scene initialization and path/obstacle export; excludes Python file staging and verification. Not a per-cycle solve time.', 'scene_source': str(scene_file), 'scene_sha256': _sha(scene_file), 'config_source': str(config_file), 'config_sha256': _sha(config_file), 'path_rows': len(frozen_path), 'obstacle_rows': len(frozen_obstacles), 'frozen_path_sha256': scene.get('path_sha256'), 'frozen_obstacles_sha256': scene.get('obstacles_sha256'), 'comparison_path': 'Frozen B201 path; regeneration must match before it is used.', 'validation': {'regeneration_performed': False, 'hybrid_astar_success_logged': None, 'qp_smoothing_logged': None, 'path_matches_frozen': None, 'obstacles_match_frozen': None}}
    if not regenerate:
        metadata['timing_scope'] = 'Cached first stage: no path planning time was measured.'
        _write_json(stage / 'metadata.json', metadata)
        return {'path': frozen_path.copy(), 'path_stage': metadata}
    try:
        root = _project_root(baseline_dir)
        executable = replay_executable(root)
        map_file = original_dir(root) / 'Maps/bitmap/B201_global_map.json'
        for required in (executable, map_file):
            if not required.is_file():
                raise FileNotFoundError(required)
        if scene.get('map_sha256') and _sha(map_file) != scene['map_sha256']:
            raise ValueError('Original B201 map hash differs from the frozen comparison map')
        (stage / 'config').mkdir()
        (stage / 'Maps/bitmap').mkdir(parents=True)
        for name in ('main', 'ilqr', 'hybrid_astar'):
            _write_json(stage / 'config' / (name + '.json'), configs[name])
        shutil.copy2(map_file, stage / 'Maps/bitmap/B201_global_map.json')
        metadata.update(executable_source=str(executable), executable_sha256=_sha(executable), map_source=str(map_file), map_sha256=_sha(map_file), generated_config_sha256={name: _sha(stage / 'config' / (name + '.json')) for name in ('main', 'ilqr', 'hybrid_astar')})
        env = os.environ.copy()
        dll_dirs = [executable.parent, baseline_dir / 'reproduction/build', Path('C:/msys64/mingw64/bin'), Path('C:/Program Files/mingw64/bin')]
        env['PATH'] = os.pathsep.join((str(p) for p in dll_dirs if p.is_dir())) + os.pathsep + env.get('PATH', '')
        with (stage / 'prepare.log').open('wb') as log:
            start = time.perf_counter()
            completed = subprocess.run([str(executable), '--prepare-scene'], cwd=stage, env=env, stdout=log, stderr=subprocess.STDOUT, check=False, timeout=300)
            metadata['stage_wall_ms'] = (time.perf_counter() - start) * 1000.0
        metadata['returncode'] = completed.returncode
        metadata['validation']['regeneration_performed'] = True
        text = (stage / 'prepare.log').read_text(encoding='utf-8', errors='replace')
        hybrid_success = 'Hybrid A* planning succeeded! Raw path points:' in text
        qp_logged = '[QP SMOOTHING]' in text
        metadata['validation'].update(hybrid_astar_success_logged=hybrid_success, qp_smoothing_logged=qp_logged)
        if completed.returncode:
            raise RuntimeError('Path scene executable failed; see path_stage/prepare.log')
        if not hybrid_success or not qp_logged:
            raise RuntimeError('Hybrid A* success and QP smoothing were not both logged; fallback is not accepted')
        generated_path = _read_rows(stage / 'frozen_global_path.txt', 3)
        generated_obs = _read_rows(stage / 'frozen_obstacles.txt', 4)
        path_match = generated_path.shape == frozen_path.shape and np.allclose(generated_path, frozen_path, rtol=0.0, atol=1e-10)
        obs_match = generated_obs.shape == frozen_obstacles.shape and np.allclose(generated_obs, frozen_obstacles, rtol=0.0, atol=1e-10)
        metadata['validation'].update(path_matches_frozen=bool(path_match), obstacles_match_frozen=bool(obs_match), absolute_tolerance=1e-10, path_max_abs_error=float(np.max(np.abs(generated_path - frozen_path))) if generated_path.shape == frozen_path.shape else None, obstacle_max_abs_error=float(np.max(np.abs(generated_obs - frozen_obstacles), initial=0.0)) if generated_obs.shape == frozen_obstacles.shape else None)
        metadata['generated_path_sha256'] = _sha(stage / 'frozen_global_path.txt')
        metadata['generated_obstacles_sha256'] = _sha(stage / 'frozen_obstacles.txt')
        metadata['log_sha256'] = _sha(stage / 'prepare.log')
        if not path_match or not obs_match:
            raise ValueError('Regenerated geometry/obstacles differ from frozen B201; comparison was not altered')
    except Exception as error:
        metadata['error'] = str(error)
        _write_json(stage / 'metadata.json', metadata)
        raise
    _write_json(stage / 'metadata.json', metadata)
    return {'path': frozen_path.copy(), 'path_stage': metadata}

def main():
    """第一阶段几何路径独立生成与核验入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-path', action='store_true', help='Read the frozen path without timing or rerunning the first stage')
    args = parser.parse_args()
    result = prepare_path(args.baseline_dir, args.output, regenerate=not args.reuse_path)
    print(json.dumps(result['path_stage'], ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()