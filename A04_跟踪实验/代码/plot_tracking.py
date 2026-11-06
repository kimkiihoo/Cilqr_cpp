"""铰接式车辆多场景多方法轨迹跟踪实验可视化对比公共绘图库。

提供一致的图表样式、学术色彩规范、帧轴 (Frame Index = t / 0.1 s) 格式化与多格式矢量导出支持。
"""
from pathlib import Path
import argparse
import csv
import json
import math
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
BASE = Path(__file__).resolve().parents[1]
MAPS = ('B207',)
METHODS = ('cilqr', 'alilqr', 'sqp')
NAMES = {'cilqr': 'R-CILQR', 'alilqr': 'AL-iLQR', 'sqp': 'SQP-NMPC', 'graph': 'Hybrid A*'}
COLORS = {'cilqr': '#174A7E', 'alilqr': '#C4383C', 'sqp': '#8056A3', 'graph': '#9A761C'}
STYLES = {'cilqr': '-', 'alilqr': '--', 'sqp': '-.', 'graph': '-'}
MARKERS = {'cilqr': 'o', 'alilqr': '^', 'sqp': 'D', 'graph': 's'}

def read_csv(path):
    """读取跟踪仿真输出 CSV 文件并返回数值字典。

参数:
    path (Path): CSV 数据文件路径。

返回:
    dict[str, np.ndarray]: 包含状态、参考与误差数组的字典。
"""
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return {name: np.array([float(r[name]) if r[name] not in ('', 'None', 'null') else np.nan for r in rows]) for name in reader.fieldnames if name in {'t', 'x', 'y', 'theta', 'gamma', 'v', 'gamma_dot', 'v_command', 'gamma_dot_command', 'ref_x', 'ref_y', 'ref_theta', 'ref_gamma', 'ref_v', 'ref_gamma_dot', 'front_lateral_error_m', 'rear_lateral_error_m', 'heading_error_rad', 'gamma_error_rad'}}

def trace(ax, x, y, method, label=None):
    """在指定 Matplotlib 子图上按算法样式绘制轨迹或误差时序折线。

参数:
    ax (plt.Axes): 目标坐标轴对象。
    x (np.ndarray): 横坐标数据 (时间帧或 X 坐标 [m])。
    y (np.ndarray): 纵坐标数据 (横向误差 [m] 或 Y 坐标 [m])。
    method (str): 算法标识符 ('cilqr', 'alilqr', 'sqp', 'graph')。
    label (str, optional): 图例标签文本。
"""
    ax.plot(x, y, color=COLORS[method], linestyle=STYLES[method], linewidth=1.05, marker=MARKERS[method], markersize=2.8, markeredgewidth=0.55, markerfacecolor='white', markevery=max(1, len(x) // 18), label=label or NAMES[method])

def common(ax):
    """设置子图公共网格线与坐标轴显示规范。

参数:
    ax (plt.Axes): 目标坐标轴对象。
"""
    ax.grid(True, color='#E3E3E3', linewidth=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', labelbottom=True)

def time_axis(ax, xmax):
    """格式化时间帧序列横坐标轴 (Frame Index = t / 0.1 s)。

参数:
    ax (plt.Axes): 目标坐标轴对象。
    xmax (float): 最大时间帧数值。
"""
    common(ax)
    step = 25 if xmax <= 210 else 50
    if xmax > 500:
        step = math.ceil(xmax / 8 / 50) * 50
    ticks = np.arange(0, xmax + 1e-08, step)
    if len(ticks) == 0 or abs(ticks[-1] - xmax) > step * 0.35:
        ticks = np.r_[ticks, xmax]
    ax.set_xticks(ticks)
    ax.set_xlim(-0.5, xmax + 0.5)
    ax.set_xlabel('Frame Index', labelpad=2)
    ax.margins(y=0.12)

def legend(ax, spatial=False):
    """配置并自适应避让折线曲线的单列紧凑图例。

参数:
    ax (plt.Axes): 目标坐标轴对象。
    spatial (bool): 是否为平面轨迹空间图。
"""
    if spatial:
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.26), ncol=1, handlelength=2.5, fontsize=9, borderpad=0.3, labelspacing=0.25, framealpha=1, fancybox=False, edgecolor='#CCCCCC')
        return
    for attempt in range(10):
        leg = ax.legend(loc='upper right', ncol=1, handlelength=2.5, fontsize=9, borderpad=0.3, labelspacing=0.25, framealpha=1, fancybox=False, edgecolor='#CCCCCC')
        ax.figure.canvas.draw()
        box = leg.get_window_extent(ax.figure.canvas.get_renderer()).padded(3)
        if not any((line.get_transform().transform_path(line.get_path()).intersects_bbox(box, filled=False) for line in ax.lines if line.get_label() != '_zero')):
            return
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + (high - low) * 0.25)
    raise RuntimeError('Could not reserve clear legend area')

def export(fig, folder, stem, records):
    """将绘制完成的图表导出为 600 DPI PNG 及矢量 PDF、SVG 格式文件。

参数:
    fig (plt.Figure): Matplotlib 画布对象。
    folder (Path): 输出目录路径。
    stem (str): 输出文件主名称。
    records (list[dict]): 输出元数据记录列表。
"""
    folder.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    for ax in fig.axes:
        if not ax.get_xlabel():
            raise AssertionError('Every subplot needs an x-axis label')
        if not all((t.label1.get_visible() for t in ax.xaxis.get_major_ticks())):
            raise AssertionError('Both subplots need visible x tick labels')
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(folder / f'{stem}.{extension}', dpi=600, bbox_inches='tight', pad_inches=0.07)
    records.append({'figure': stem, 'folder': folder.relative_to(BASE).as_posix(), 'formats': ['png', 'pdf', 'svg'], 'smoothing': False})
    plt.close(fig)

def errors(datasets, folder, stem, title, records):
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.85), sharex=False)
    for ax, field, label in zip(axes, ('front_lateral_error_m', 'rear_lateral_error_m'), ('Front Lateral Error (m)', 'Rear Lateral Error (m)')):
        xmax = 0
        for method, data in datasets.items():
            x = data['t'] / 0.1
            trace(ax, x, data[field], method)
            xmax = max(xmax, float(x[-1]))
        ax.axhline(0, color='#777777', linewidth=0.5, linestyle=':', label='_zero')
        time_axis(ax, xmax)
        ax.set_ylabel(label)
    axes[0].set_title(title, pad=7)
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.11, top=0.92, hspace=0.48)
    legend(axes[0])
    export(fig, folder, stem, records)

def rear(x, y, theta, gamma, lf=3.0, lr=3.0):
    return (x - lf * np.cos(theta) - lr * np.cos(theta + gamma), y - lf * np.sin(theta) - lr * np.sin(theta + gamma))

def trajectories(data, method, folder, stem, records, lf=3.0, lr=3.0):
    refrear = rear(data['ref_x'], data['ref_y'], data['ref_theta'], data['ref_gamma'], lf, lr)
    actualrear = rear(data['x'], data['y'], data['theta'], data['gamma'], lf, lr)
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.8))
    for ax, refxy, actualxy, title in zip(axes, ((data['ref_x'], data['ref_y']), refrear), ((data['x'], data['y']), actualrear), ('Front Axle', 'Rear Axle')):
        ax.plot(*refxy, linestyle='--', color='#666666', linewidth=1.1, label='Reference')
        trace(ax, *actualxy, method, label='Actual')
        ax.set(xlabel='X (m)', ylabel='Y (m)')
        ax.set_title(title, loc='left', pad=12)
        ax.set_aspect('equal', adjustable='datalim')
        common(ax)
        ax.margins(0.08)
        legend(ax, spatial=True)
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.075, top=0.92, hspace=0.6)
    export(fig, folder, stem, records)

def inputs(data, method, folder, stem, records):
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.85))
    for ax, field, reference, label in zip(axes, ('v', 'gamma_dot'), ('ref_v', 'ref_gamma_dot'), ('Velocity (m/s)', 'Articulation Rate (rad/s)')):
        x = data['t'] / 0.1
        ax.plot(x, data[reference], linestyle='--', color='#666666', linewidth=1.0, label='Reference')
        trace(ax, x, data[field], method, label='Actual')
        time_axis(ax, float(x[-1]))
        ax.set_ylabel(label)
    axes[0].set_title('Control Input Tracking', pad=7)
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.11, top=0.92, hspace=0.48)
    legend(axes[0])
    export(fig, folder, stem, records)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=BASE)
    args = parser.parse_args()
    root = args.root.resolve()
    font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10, 'axes.labelsize': 10.5, 'axes.titlesize': 11, 'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    records = []
    mapname = 'B207'
    base = root / '原始数据' / '跟踪' / mapname
    nominal = {method: read_csv(base / method / 'nominal.csv') for method in METHODS}
    for method in METHODS:
        dest = root / '绘图数据' / mapname / method
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base / method / 'nominal.csv', dest / 'nominal.csv')
    data = nominal['cilqr']
    fig = plt.figure(figsize=(7.5, 4.9))
    grid = fig.add_gridspec(2, 2, width_ratios=(0.95, 1.1), wspace=0.39, hspace=0.58)
    path_ax = fig.add_subplot(grid[:, 0])
    path_ax.plot(data['ref_x'], data['ref_y'], '--', color='#666666', linewidth=1.1, label='Reference')
    trace(path_ax, data['x'], data['y'], 'cilqr', label='Actual')
    path_ax.set(xlabel='X (m)', ylabel='Y (m)', title='Trajectory Tracking')
    path_ax.set_aspect('equal', adjustable='datalim')
    common(path_ax)
    path_ax.margins(0.12)
    legend(path_ax)
    for i, field, title in ((0, 'front_lateral_error_m', 'Front Axle'), (1, 'rear_lateral_error_m', 'Rear Axle')):
        ax = fig.add_subplot(grid[i, 1])
        trace(ax, data['t'] / 0.1, data[field], 'cilqr')
        ax.axhline(0, color='#777777', linewidth=0.5, linestyle=':')
        time_axis(ax, float(data['t'][-1] / 0.1))
        ax.set_ylabel('Lateral Error (m)')
        ax.set_title(title, pad=6)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.115, top=0.92)
    export(fig, root / '图片', 'B207_R_CILQR_Tracking', records)
    (root / '核验结果').mkdir(exist_ok=True)
    (root / '核验结果' / 'figure_manifest.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Exported {len(records)} figures in PNG/PDF/SVG.')
if __name__ == '__main__':
    main()