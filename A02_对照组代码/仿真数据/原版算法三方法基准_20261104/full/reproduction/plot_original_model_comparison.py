"""原版三方法 (R-CILQR, AL-iLQR, SQP-NMPC) 仿真轨迹、控制量与横向偏差对比绘图脚本。"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from project_paths import existing, project_root
METHODS = ('cilqr', 'alilqr', 'sqp')
MAPS = ('B201', 'B207', 'B301')

def sha(path):
    """计算文件哈希。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def renderer():
    """配置图表样式与排版布局。"""
    root = project_root(Path(__file__).resolve().parent)
    script = existing(root, ['A02_对照组代码/图片/原版算法三方法基准_20261104_竖排图例/代码/plot_comparison.py', '06_重绘图片/原版算法三方法基准_20261104_竖排图例/代码/plot_comparison.py'])
    spec = importlib.util.spec_from_file_location('current_three_method_plot', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (module, script)

def load_run(path):
    """加载各方法运行数据。"""
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'frame_index', 'plan_time_s', 'time_s', 'velocity', 'gamma_dot', 'solve_time_ms'}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError('Missing required cycle fields: ' + str(path))
        rows = list(reader)
    if not rows:
        raise ValueError('Empty cycle input: ' + str(path))
    frames = [int(row['frame_index']) for row in rows]
    if any((frame < 0 for frame in frames)) or any((b <= a for a, b in zip(frames, frames[1:]))):
        raise ValueError('Frame indices must be nonnegative and strictly increasing: ' + str(path))
    return {frame: (index, row) for index, (frame, row) in enumerate(zip(frames, rows))}

def merge(runs, kind):
    """合并多方法时序数据。"""
    fields = ('velocity', 'gamma_dot') if kind == 'control' else ('solve_time_ms',)
    result = []
    for frame in sorted(set().union(*(set(rows) for rows in runs.values()))):
        available = [rows[frame][1] for rows in runs.values() if frame in rows]
        times = [float(row['plan_time_s']) for row in available]
        if any((not math.isfinite(t) or abs(t - times[0]) > 1e-10 for t in times)):
            raise ValueError('Methods have different planning clocks at frame ' + str(frame))
        record = {'frame_index': frame, 'plan_time_s': available[0]['plan_time_s']}
        for method in METHODS:
            index, source = runs[method].get(frame, ('', {}))
            for field in (*fields, 'time_s', 'plan_time_s'):
                record[method + '_' + field] = source.get(field, '')
            record[method + '_source_row_index'] = index
            if kind == 'solve':
                for field in ('executed', 'scipy_success', 'slsqp_status'):
                    record[method + '_' + field] = source.get(field, '')
        result.append(record)
    return result

def save_csv(path, rows):
    """导出绘图数值数据。"""
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    """绘制多方法对比曲线图。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.input / 'results' if (args.input / 'results').is_dir() else args.input
    data = {name: {method: load_run(source / name / method / 'cycles.csv') for method in METHODS} for name in MAPS}
    plotter, style_source = renderer()
    plotter.font_manager.findfont('Times New Roman', fallback_to_default=False)
    plotter.plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 9, 'axes.labelsize': 9.5, 'axes.titlesize': 10, 'legend.fontsize': 8.5, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'figure_data').mkdir()
    provenance = {'scope': 'All stored cycles; no optimizer rerun, smoothing or failed-frame removal.', 'input': str(args.input.resolve()), 'style_source': str(style_source), 'style_sha256': sha(style_source), 'script_sha256': sha(__file__), 'runs': {}}
    for name, runs in data.items():
        control, solve = (merge(runs, 'control'), merge(runs, 'solve'))
        save_csv(args.output / 'figure_data' / (name + '_control_outputs.csv'), control)
        save_csv(args.output / 'figure_data' / (name + '_solve_times.csv'), solve)
        plotter.plot(control, solve, args.output, name)
        provenance['runs'][name] = {method: {'cycles': len(runs[method]), 'source_sha256': sha(source / name / method / 'cycles.csv')} for method in METHODS}
    (args.output / 'figure_provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'maps': len(MAPS), 'figures': 6, 'merged_csv': 6, 'output': str(args.output.resolve())}, ensure_ascii=False))
if __name__ == '__main__':
    main()