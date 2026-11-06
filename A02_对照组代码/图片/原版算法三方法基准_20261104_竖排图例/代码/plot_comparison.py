"""论文插图排版专用: 原版三方法基准 (R-CILQR, AL-iLQR, SQP-NMPC) 竖排图例对比图绘制脚本。

遵循期刊论文标准格式:
- 包含平面路径航向轨迹、纵向车速时序、铰接角速度时序与道路走廊横向偏差子图；
- 统一采用 Frame Index = t / 0.1 s 作为横坐标；
- 导出高质量 600 DPI PNG 以及矢量 PDF 与 SVG 格式图表。
"""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter
import numpy as np
METHODS = ('cilqr', 'alilqr', 'sqp')
MAPS = ('B201', 'B207', 'B301')
NAMES = {'cilqr': 'R-CILQR', 'alilqr': 'AL-iLQR', 'sqp': 'SQP-NMPC'}
COLORS = {'cilqr': '#174A7E', 'alilqr': '#C4383C', 'sqp': '#8056A3'}
STYLES = {'cilqr': '-', 'alilqr': '--', 'sqp': '-.'}
MARKERS = {'cilqr': 'o', 'alilqr': '^', 'sqp': 'D'}

def number(value):
    """格式化浮点数值字符串。"""
    if value is None or str(value).strip() in ('', 'None', 'null'):
        return math.nan
    return float(value)

def values(rows, field):
    """提取数值数组。"""
    result = np.array([number(row[field]) for row in rows], dtype=float)
    result[~np.isfinite(result)] = np.nan
    return result

def load_merged_csv(path, kind):
    """加载并合并三方法的多列 CSV 绘图数据。"""
    fields = ('velocity', 'gamma_dot') if kind == 'control' else ('solve_time_ms',)
    required = {'frame_index', *(f'{method}_{field}' for method in METHODS for field in fields)}
    csv.field_size_limit(64 * 1024 * 1024)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        absent = required - set(reader.fieldnames or ())
        if absent:
            raise ValueError(f'Missing columns in {path}: {sorted(absent)}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'Empty input: {path}')
    for field in required:
        values(rows, field)
    x = values(rows, 'frame_index')
    if not np.isfinite(x).all() or np.any(x < 0) or np.any(x != np.rint(x)):
        raise ValueError(f'Invalid nonnegative integer frame_index in {path}')
    if np.any(np.diff(x) <= 0):
        raise ValueError(f'frame_index must be strictly increasing in {path}')
    return rows

def markers(y):
    """生成自适应等间距散点标记位置掩码。"""
    finite = np.flatnonzero(np.isfinite(y))
    if not finite.size:
        return []
    chosen = set(finite[::max(1, int(np.ceil(len(y) / 18)))].tolist())
    chosen.update((i for i in finite if i == 0 or i == len(y) - 1 or (not np.isfinite(y[i - 1])) or (not np.isfinite(y[i + 1]))))
    chosen.add(int(np.nanargmax(y)))
    return sorted(chosen)

def line(ax, x, y, method, control=False):
    """按各方法统一样式与颜色绘制折线。"""
    ax.plot(x, y, color=COLORS[method], linestyle=STYLES[method], linewidth=0.95, marker=MARKERS[method], markersize=2.8, markeredgewidth=0.55, markerfacecolor='white', markevery=markers(y), label=NAMES[method])

def axis(ax, x):
    """统一格式化时间帧横坐标轴。"""
    ax.grid(True, color='#E3E3E3', linewidth=0.45)
    ax.set_axisbelow(True)
    window_end = int(round(float(x[-1]))) + 1
    tick_step = 25 if window_end == 200 else 50
    ax.set_xticks(np.arange(0, window_end + 1, tick_step))
    ax.set_xlim(-0.5, window_end + 0.5)
    ax.margins(y=0.08)

def export(fig, out, stem):
    """将 Matplotlib 图表导出为 PNG/PDF/SVG 三种格式。"""
    for ax in fig.axes:
        if ax.get_ylabel() == 'Solve Time (ms)':
            finite_times = [float(v) for curve in ax.lines for v in curve.get_ydata() if math.isfinite(float(v))]
            if not finite_times or min(finite_times) <= 0:
                raise ValueError('Logarithmic timing plots require strictly positive finite times')
            ax.set_yscale('log', base=10)
            ax.set_ylim(0.8 * 10 ** math.floor(math.log10(min(finite_times))), 3.0 * 10 ** math.ceil(math.log10(max(finite_times))))
            ax.yaxis.set_major_locator(LogLocator(base=10))
            ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_ylabel('Computation Time (ms)')
            ax.set_title('Computation Time Comparison')
        if ax.get_legend() is not None:
            ax.legend(loc='upper right', ncol=1, handlelength=2.4, fontsize=8, borderpad=0.3, labelspacing=0.2, framealpha=1.0, fancybox=False, edgecolor='#CCCCCC')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        if ax.get_xlabel() != 'Frame Index':
            raise AssertionError('Each subplot must label the horizontal axis')
        if not all((t.label1.get_visible() for t in ax.xaxis.get_major_ticks())):
            raise AssertionError('Every major x tick must be visible')
        legend = ax.get_legend()
        if legend is not None:
            box = legend.get_window_extent(renderer).padded(3)
            overlaps = [curve.get_label() for curve in ax.lines if curve.get_transform().transform_path(curve.get_path()).intersects_bbox(box, filled=False)]
            if overlaps:
                raise AssertionError(f'{stem}: legend overlaps curves {overlaps}')
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(out / f'{stem}.{extension}', dpi=600, bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)

def plot(control, solve, out, map_name):
    """针对指定测试场景 (B201, B207, B301) 绘制完整的四子图排版。"""
    x = values(control, 'frame_index')
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.35), sharex=True, gridspec_kw={'hspace': 0.42})
    for ax, field, label in zip(axes, ('velocity', 'gamma_dot'), ('Velocity (m/s)', 'Articulation Rate (rad/s)')):
        for method in METHODS:
            line(ax, x, values(control, f'{method}_{field}'), method, control=True)
        axis(ax, x)
        ax.set_ylabel(label)
        ax.tick_params(axis='x', labelbottom=True)
        ax.set_xlabel('Frame Index', labelpad=2)
    axes[0].set_title('Control Output Comparison', pad=7)
    axes[0].legend(loc='best', ncol=len(METHODS), handlelength=2.7, columnspacing=1.5)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.115, top=0.92)
    export(fig, out, f'{map_name}_Control_Output_Comparison')
    x = values(solve, 'frame_index')
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for method in METHODS:
        line(ax, x, values(solve, f'{method}_solve_time_ms'), method)
    axis(ax, x)
    ax.set(xlabel='Frame Index', ylabel='Solve Time (ms)', title='Solve Time Comparison')
    ax.legend(loc='best', ncol=len(METHODS), handlelength=2.7, columnspacing=1.5)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.15, top=0.9)
    export(fig, out, f'{map_name}_Solve_Time_Comparison')

def main():
    """论文对比插图生成主函数。"""
    base = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=base / '绘图数据', help='Directory containing the six merged plotting CSVs')
    parser.add_argument('--output', type=Path, default=base / '图片', help='Directory for PNG/PDF/SVG figures')
    args = parser.parse_args()
    datasets = {name: (load_merged_csv(args.data / f'{name}_control_outputs.csv', 'control'), load_merged_csv(args.data / f'{name}_solve_times.csv', 'solve')) for name in MAPS}
    font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 9, 'axes.labelsize': 9.5, 'axes.titlesize': 10, 'legend.fontsize': 8.5, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    args.output.mkdir(parents=True, exist_ok=True)
    for name, (control, solve) in datasets.items():
        plot(control, solve, args.output, name)
    print(f'Exported 6 figures in PNG/PDF/SVG to {args.output.resolve()}')
if __name__ == '__main__':
    main()