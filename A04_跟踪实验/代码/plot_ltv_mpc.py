"""B201 场景下 R-CILQR 空间路径结合 LTV-MPC 跟踪控制结果高精绘图脚本。

绘制跟踪对比图：
- 左图：全局笛卡尔坐标系下的参考轨迹与实际跟踪轨迹对比；
- 右上图：前轴横向位置跟踪误差随帧序列变化的曲线；
- 右下图：后轴横向位置跟踪误差随帧序列变化的曲线。
所有数据点均以原始 0.01 s 步长连线绘制，不作任何平滑处理，导出 600 DPI PNG、矢量 PDF 与 SVG。
"""
from pathlib import Path
import argparse
import csv
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon
from plot_tracking import read_csv, trace, common, time_axis, legend
ROOT = Path(__file__).resolve().parents[1]

def main():
    """加载仿真轨迹与横向误差数据，生成符合学术出版标准的三联排版图。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir', type=Path, default=ROOT / '原始数据/LTV_MPC/B201/cilqr')
    ap.add_argument('--output-dir', type=Path, default=ROOT / '图片/LTV_MPC')
    args = ap.parse_args()
    source = args.input_dir / 'nominal.csv'
    data = read_csv(source)
    font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10, 'axes.labelsize': 10.5, 'axes.titlesize': 11, 'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
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
    for row, field, title in ((0, 'front_lateral_error_m', 'Front Axle'), (1, 'rear_lateral_error_m', 'Rear Axle')):
        axis = fig.add_subplot(grid[row, 1])
        trace(axis, data['t'] / 0.1, data[field], 'cilqr', label='LTV-MPC')
        axis.axhline(0, color='#777777', linewidth=0.5, linestyle=':')
        time_axis(axis, float(data['t'][-1] / 0.1))
        axis.set_xticks(np.arange(0, float(data['t'][-1] / 0.1) + 1e-08, 100))
        axis.set_ylabel('Lateral Error (m)')
        axis.set_title(title, pad=6)
        axis.ticklabel_format(axis='y', style='plain', useOffset=False)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.115, top=0.92)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(args.output_dir / f'B201_R_CILQR_LTV_MPC_Tracking.{extension}', dpi=600, bbox_inches='tight', pad_inches=0.07)
    plt.close(fig)
    plot_data = ROOT / '绘图数据/LTV_MPC'
    plot_data.mkdir(parents=True, exist_ok=True)
    with source.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    columns = ['t', 'x', 'y', 'ref_x', 'ref_y', 'front_lateral_error_m', 'rear_lateral_error_m']
    with (plot_data / 'B201_R_CILQR_LTV_MPC_Tracking.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['frame_index'] + columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({'frame_index': float(row['t']) / 0.1, **{c: row[c] for c in columns}})
    print('Exported the single LTV-MPC tracking figure (PNG/PDF/SVG).')
if __name__ == '__main__':
    main()