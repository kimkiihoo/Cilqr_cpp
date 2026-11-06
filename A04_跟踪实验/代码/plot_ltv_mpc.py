"""绘制原始时序轨迹的 LTV-MPC 跟踪结果及速度/纵向误差补充图。

保留全部仿真采样点，不作平滑，导出 600 DPI PNG、矢量 PDF 与 SVG。
"""
from pathlib import Path
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from plot_tracking import trace, common, legend

ROOT = Path(__file__).resolve().parents[1]


def seconds_axis(axis, start, end):
    """用实际仿真秒数表示参考与跟踪共享的时钟。"""
    common(axis)
    axis.set(xlim=(start, end), xticks=np.linspace(start, end, 5), xlabel='Time (s)')
    axis.margins(y=0.12)


def save_figure(figure, output_dir, stem):
    for extension in ('png', 'pdf', 'svg'):
        figure.savefig(output_dir / f'{stem}.{extension}', dpi=600,
                       bbox_inches='tight', pad_inches=0.07)
    plt.close(figure)


def save_plot_data(rows, columns, output_dir, stem):
    with (output_dir / f'{stem}.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row[column] for column in columns} for row in rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir', type=Path, default=ROOT / '原始数据/LTV_MPC/B201/cilqr')
    ap.add_argument('--output-dir', type=Path, default=ROOT / '图片/LTV_MPC')
    args = ap.parse_args()
    source = args.input_dir / 'nominal.csv'
    with source.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    columns = ['t', 'x', 'y', 'ref_x', 'ref_y', 'front_lateral_error_m',
               'rear_lateral_error_m', 'v', 'ref_v', 'front_longitudinal_error_m']
    data = {column: np.array([float(row[column]) for row in rows]) for column in columns}
    start, end = float(data['t'][0]), float(data['t'][-1])
    font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10,
                         'axes.labelsize': 10.5, 'axes.titlesize': 11, 'legend.fontsize': 9,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.linewidth': 0.6,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.5, 4.9))
    grid = fig.add_gridspec(2, 2, width_ratios=(0.95, 1.1), wspace=0.39, hspace=0.58)
    ax = fig.add_subplot(grid[:, 0])
    ax.plot(data['ref_x'], data['ref_y'], '--', color='#555555', linewidth=1.3, label='Reference')
    trace(ax, data['x'], data['y'], 'cilqr', label='LTV-MPC')
    ax.set(xlabel='X (m)', ylabel='Y (m)', title='Trajectory Tracking')
    ax.set_aspect('equal', adjustable='datalim')
    common(ax)
    ax.margins(0.12)
    legend(ax)
    # Front/rear lateral errors share one axis; all original samples are kept.
    lateral = fig.add_subplot(grid[0, 1])
    lateral.plot(data['t'], data['front_lateral_error_m'], color='#174A7E',
                 linewidth=0.8, label='Front Axle')
    lateral.plot(data['t'], data['rear_lateral_error_m'], color='#C56C2C',
                 linestyle='--', linewidth=0.9, label='Rear Axle')
    lateral.axhline(0, color='#777777', linewidth=0.5, linestyle=':')
    seconds_axis(lateral, start, end)
    lateral.set(ylabel='Lateral Error (m)', title='Lateral Tracking Errors')
    lateral.ticklabel_format(axis='y', style='plain', useOffset=False)
    lateral.legend(loc='upper right', ncol=1, handlelength=2.5, fontsize=9,
                   borderpad=0.3, labelspacing=0.25, framealpha=1,
                   fancybox=False, edgecolor='#CCCCCC')

    speed = fig.add_subplot(grid[1, 1])
    speed.plot(data['t'], data['ref_v'], '--', color='#555555',
               linewidth=1.3, label='Reference')
    trace(speed, data['t'], data['v'], 'cilqr', label='LTV-MPC')
    seconds_axis(speed, start, end)
    speed.set(ylabel='Speed (m/s)', title='Speed Tracking')
    speed.ticklabel_format(axis='y', style='plain', useOffset=False)
    legend(speed)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.115, top=0.92)
    main_stem = 'B201_R_CILQR_LTV_MPC_Tracking'
    save_figure(fig, args.output_dir, main_stem)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.9))
    axes[0].plot(data['t'], data['ref_v'], '--', color='#555555', linewidth=1.3, label='Reference')
    trace(axes[0], data['t'], data['v'], 'cilqr', label='LTV-MPC')
    axes[0].set(ylabel='Velocity (m/s)', title='Original Speed Profile Tracking')
    trace(axes[1], data['t'], data['front_longitudinal_error_m'], 'cilqr', label='LTV-MPC')
    axes[1].axhline(0, color='#777777', linewidth=0.5, linestyle=':')
    axes[1].set(ylabel='Longitudinal Error (m)', title='Front Axle')
    for axis in axes:
        seconds_axis(axis, start, end)
        axis.ticklabel_format(axis='y', style='plain', useOffset=False)
    legend(axes[0])
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.115, top=0.92, hspace=0.58)
    supplement_stem = 'B201_R_CILQR_LTV_MPC_Timing_Check'
    save_figure(fig, args.output_dir, supplement_stem)

    plot_data = ROOT / '绘图数据/LTV_MPC'
    plot_data.mkdir(parents=True, exist_ok=True)
    save_plot_data(rows, columns[:7] + ['v', 'ref_v'], plot_data, main_stem)
    save_plot_data(rows, ['t', 'ref_v', 'v', 'front_longitudinal_error_m'],
                   plot_data, supplement_stem)
    print('Exported original-time tracking figure and supplementary timing check (PNG/PDF/SVG).')


if __name__ == '__main__':
    main()
