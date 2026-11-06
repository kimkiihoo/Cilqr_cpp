"""单状态步求解与代价评估测试工具，用于单步算法调试与输出一致性检验。"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import numpy as np
import scipy
from original_backend import HERE, OriginalBackend, read_configs
from sqp_nmpc import solve_prepared

def serial(value):
    """递归清洗数据结构中的 NumPy 类型与非有限浮点数，便于 JSON 安全序列化。"""
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(item) for item in value]
    if isinstance(value, (float, np.floating)) and (not np.isfinite(value)):
        return 'NaN' if np.isnan(value) else '+Infinity' if value > 0 else '-Infinity'
    if isinstance(value, np.generic):
        return value.item()
    return value

def main():
    """加载指定场景与配置，执行单状态步求解并导出分析报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--effective-config', type=Path, help='Existing replay effective_config.json; reads main/ilqr/hybrid_astar exactly')
    parser.add_argument('--mode', choices=['evaluate', 'baseline', 'sqp'], default='evaluate')
    parser.add_argument('--maxiter', type=int, default=2, help='SQP smoke-test default; record any changed budget')
    parser.add_argument('--ftol', type=float, default=1e-06)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite ' + str(args.output))
    scene = json.loads(args.scene.read_text(encoding='utf-8'))
    if args.effective_config:
        recorded = json.loads(args.effective_config.read_text(encoding='utf-8'))
        configs = {key: recorded[key] for key in ['main', 'ilqr', 'hybrid_astar']}
    else:
        configs = read_configs()
    report = {'scope': 'One supplied state; no map simulation, arrival, collision-free or tracking claim.', 'mode': args.mode, 'scene_sha256': hashlib.sha256(args.scene.read_bytes()).hexdigest(), 'effective_config_source': str(args.effective_config) if args.effective_config else 'unchanged original handoff config', 'configs': configs, 'source_manifest': json.loads((HERE / 'source_manifest.json').read_text(encoding='utf-8')), 'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__}, 'results': {}}
    if args.mode == 'baseline':
        for method in ['r', 'al']:
            with OriginalBackend(scene['path'], configs) as backend:
                report['results'][method] = backend.solve_original(method, scene['start'], scene['initial_obstacles'])
    else:
        with OriginalBackend(scene['path'], configs) as backend:
            backend.prepare(scene['start'], scene['initial_obstacles'], target='r')
            if args.mode == 'evaluate':
                nominal = backend.nominal()
                result = backend.evaluate(nominal['controls'])
                result['original_nominal_states'] = nominal['states']
                result['nominal_vs_unclipped_rollout_max_abs'] = float(np.max(np.abs(nominal['states'] - result['states'])))
            else:
                result = solve_prepared(backend, maxiter=args.maxiter, ftol=args.ftol)
            report['results'][args.mode] = result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(serial(report), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(args.output.resolve())
if __name__ == '__main__':
    main()