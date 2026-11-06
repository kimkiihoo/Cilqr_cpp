"""B201 场景下四方法 (R-CILQR, AL-iLQR, SQP-NMPC, Hybrid A* + ST*) 轨迹与时空图对比绘图脚本。

生成高精学术出版图表:
- 平面航向轨迹对比图；
- 沿程纵向车速剖面与几何限速包络图；
- 控制指令与横向偏差时序图；
- S-T 时空二维图 (包含动态障碍物时空占用区域投影)。
"""
from pathlib import Path
import argparse
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from st_speed_search import obstacle_residual, sample_path, path_arclength
COLORS = ['#174A7E', '#C4383C', '#8056A3', '#996515']
STYLES = ['-', '--', '-.', ':']
LABELS = ['R-CILQR', 'AL-iLQR', 'SQP-NMPC', 'Hybrid A* + ST*']

def export(fig, output, stem):
    """将画布以 600 DPI 导出为 PNG/PDF/SVG 文件。"""
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(output / (stem + '.' + extension), dpi=300, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)

def main():
    """加载多方法仿真数据并渲染出版级四方法性能对比图。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    runs = []
    for folder in ('cilqr', 'alilqr', 'sqp'):
        values = np.load(args.baseline / 'B201' / folder / 'trajectories.npz')
        runs.append((values['actual_states'], values['controls'][:, 0, :]))
    values = np.load(args.run / 'trajectories.npz')
    runs.append((values['actual_states'], values['controls']))
    path, obstacles = (values['path'], values['initial_obstacles'])
    times = values['times']
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), layout='constrained')
    xy, velocity, angle, rate = axes.ravel()
    xy.plot(path[:, 0], path[:, 1], color='.65', lw=1.0, label='Geometric path')
    for i, ((states, controls), label) in enumerate(zip(runs, LABELS)):
        kwargs = dict(color=COLORS[i], ls=STYLES[i], lw=1.2, label=label)
        xy.plot(states[:, 0], states[:, 1], **kwargs)
        velocity.plot(times[1:], controls[:, 0], **kwargs)
        angle.plot(times, states[:, 3], **kwargs)
        rate.plot(times[1:], controls[:, 1], **kwargs)
    for index, obstacle in enumerate(obstacles):
        end = obstacle[:2] + 40 * obstacle[3] * np.array([np.cos(obstacle[2]), np.sin(obstacle[2])])
        xy.plot([obstacle[0], end[0]], [obstacle[1], end[1]], color='.45', ls='--', lw=0.8)
        xy.scatter(obstacle[0], obstacle[1], marker='s', facecolor='none', edgecolor='.3', s=28)
        xy.annotate('Obs. ' + str(index + 1), obstacle[:2], xytext=(5, -10), textcoords='offset points', fontsize=8)
    xy.set(xlabel='x (m)', ylabel='y (m)', title='B201: executed trajectories')
    xy.set_aspect('equal', adjustable='datalim')
    xy.legend(loc='best', fontsize=8)
    velocity.set(xlabel='Time (s)', ylabel='Speed (m/s)', title='Applied speed')
    angle.set(xlabel='Time (s)', ylabel='Articulation angle (rad)', title='Executed articulation angle')
    rate.set(xlabel='Time (s)', ylabel='Articulation rate (rad/s)', title='Applied articulation rate')
    for ax in (velocity, angle, rate):
        ax.set_xlim(0, 40)
        ax.grid(alpha=0.15)
    export(fig, args.output, 'B201_four_method_execution')
    protocol = json.loads((args.run / 'protocol.json').read_text(encoding='utf-8'))
    s_grid = np.linspace(0, path_arclength(path)[-1], 500)
    t_grid = np.linspace(0, 40, 801)
    points = sample_path(path, s_grid)
    residual = obstacle_residual(points[None, :, :2], t_grid[:, None], obstacles, protocol['model_parameters'])
    fig, ax = plt.subplots(figsize=(7.2, 4.4), layout='constrained')
    ax.contourf(t_grid, s_grid, (residual > 0).T, levels=[0.5, 1.5], colors=['#c5c5c5'])
    ax.plot(times, values['planned_s'], color=COLORS[3], lw=1.5, label='ST* speed plan')
    ax.plot([], [], color='#c5c5c5', lw=7, label='Obstacle occupancy')
    ax.set(xlabel='Time (s)', ylabel='Distance along path (m)', xlim=(0, 40), ylim=(0, s_grid[-1]), title='B201: space-time speed planning')
    ax.legend(loc='upper left')
    export(fig, args.output, 'B201_ST_speed_plan')
    with (args.output / 'B201_four_method_plot_data.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'time_s', 'x', 'y', 'theta', 'gamma', 'applied_velocity', 'applied_gamma_dot'])
        for label, (states, controls) in zip(LABELS, runs):
            for k, state in enumerate(states):
                u = controls[k - 1] if k else [None, None]
                writer.writerow([label, times[k], *state, *u])
    print(args.output)
if __name__ == '__main__':
    main()