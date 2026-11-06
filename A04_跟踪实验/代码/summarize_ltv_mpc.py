"""B201 正常工况下 R-CILQR 空间路径 + LTV-MPC 跟踪实验精简汇总与报告导出脚本。

本脚本读取原始仿真数据与独立核验结果，严格核对文件 SHA-256 哈希值与验收指标，
生成标准精简汇总 CSV 表格及实验说明文本。
"""
from pathlib import Path
import csv
import hashlib
import json
ROOT = Path(__file__).resolve().parents[1]

def main():
    """读取 B201 LTV-MPC 跟踪仿真输出与核验报告，导出精简汇总表与实验说明。"""
    raw = ROOT / '原始数据/LTV_MPC/B201/cilqr'
    summary = json.loads((raw / 'summary.json').read_text(encoding='utf-8'))
    audit = json.loads((ROOT / '核验结果/ltv_mpc_independent_audit_B201.json').read_text(encoding='utf-8'))
    assert summary['map'] == audit['map'] == 'B201'
    assert summary['case'] == audit['case'] == 'nominal'
    assert audit['audit_passed'] and audit['acceptance']['all_engineering_targets_met']
    for filename, key in [('nominal.csv', 'actual_csv_sha256'), ('original_reference.csv', 'original_reference_sha256'), ('retimed_reference.csv', 'retimed_reference_sha256')]:
        current_hash = hashlib.sha256((raw / filename).read_bytes()).hexdigest()
        assert current_hash == summary[key], f'Changed source: {filename}'
        assert any((h['sha256'] == current_hash for h in audit['input_hashes']))
    geometry = audit['dense_actual_clock_geometry']
    corridor = audit['dense_global_path_corridor_diagnostic']
    timing = audit['controller_timing']
    row = {'map': 'B201', 'planner': 'R-CILQR', 'controller': 'LTV-MPC', 'case': 'nominal', 'front_lateral_rmse_m': summary['front_lateral_error_m_rmse'], 'rear_lateral_rmse_m': summary['rear_lateral_error_m_rmse'], 'front_lateral_max_abs_m': summary['front_lateral_error_m_max_abs'], 'rear_lateral_max_abs_m': summary['rear_lateral_error_m_max_abs'], 'dense_min_body_clearance_m': geometry['minimum_clearance_m'], 'dense_collision_samples': geometry['zero_clearance_or_collision_samples'], 'dense_corner_corridor_violation_m': corridor['joint_extended_corner_max_violation_m'], 'goal_distance_m': summary['final_goal_distance_m'], 'terminal_speed_mps': summary['final_speed_mps'], 'duration_s': summary['duration_s'], 'qp_failures': timing['qp_failures'], 'controller_mean_ms': timing['controller_mean_s'] * 1000, 'controller_max_ms': timing['controller_max_s'] * 1000, 'deadline_exceedance_cycles': timing['deadline_exceedance_cycles'], 'control_cycles': timing['cycles'], 'engineering_targets_met': audit['acceptance']['all_engineering_targets_met'], 'hard_real_time_demonstrated': False}
    output = ROOT / '汇总结果'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'B201_LTV_MPC_正常工况_精简汇总.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    text = f"B201正常工况：R-CILQR空间路径 + LTV-MPC跟踪\n\n实验范围\n只保留B201、正常初始状态、一次完整跟踪。原始初态为[x,y,theta,gamma,v,gamma_dot]=[10,20,1.2,0,5,0]。\n原R-CILQR的401个空间姿态、微小倒车段及终点全部保留。为满足跟踪约束，预先重排参考速度时序（巡航上限3 m/s，实际初速度仍为5 m/s并执行制动过渡），不按车辆实际位置重置参考。\n参考行程{summary['retimed_travel_time_s']:.3f} s，加末端稳定停车，共{summary['duration_s']:.1f} s。不能写成保留原40 s的速度时序，或将效果全部归因于控制器更换。\n沿用铰接车辆运动学并增加0.20/0.15 s一阶执行器滞后；控制周期0.05 s，积分步长0.01 s，预测60步/3 s。动态障碍始终按实际仿真时间运动。\n\n完整轨迹结果（包含起始过渡及停车，无删点、无平滑）\n前轴横向RMSE：{row['front_lateral_rmse_m']:.6f} m；最大绝对误差：{row['front_lateral_max_abs_m']:.6f} m。\n后轴横向RMSE：{row['rear_lateral_rmse_m']:.6f} m；最大绝对误差：{row['rear_lateral_max_abs_m']:.6f} m。\n误差采用实际状态与同一时刻固定重定时参考的差值，沿各自参考轴线的法向投影；独立原空间折线核验也达到预设精度目标。\n独立0.002 s稠密核验：完整前后车体（延伸至铰接点）最小障碍净空{row['dense_min_body_clearance_m']:.6f} m，碰撞样本{row['dense_collision_samples']}个；前后轴中心及车体角点均未超出所检查的±5 m路径走廊。\n终点距原目标(80,80)为{row['goal_distance_m']:.6f} m，末速度约0 m/s，完成停车。原参考本身终点也距目标约0.65 m。\nQP失败{row['qp_failures']}次、回退{timing['fallback_cycles']}次，实际状态和控制限制均在数值容差内。\n\n计算时间与适用范围\n完整控制计算平均{row['controller_mean_ms']:.3f} ms、最大{row['controller_max_ms']:.3f} ms；{row['deadline_exceedance_cycles']}/{row['control_cycles']}个周期超过50 ms（{timing['deadline_exceedance_fraction']:.2%}）。仿真没有注入计算墙钟延迟，因此不作为硬实时或实车安全保证。\n安全结论限于当前模型、动态障碍和稠密采样几何检查；道路走廊使用原路径顶点法向定义，不代表栅格地图边界的连续碰撞证明。\n\n文件与复现\n图片/LTV_MPC：唯一主图，PNG（600 dpi）、PDF、SVG各一份。左侧参考/实际轨迹，右侧前后轴横向误差；两张误差子图均显示Frame Index及横坐标数字。Frame Index=t/0.1 s，0.01 s中间样本全部保留。\n绘图数据/LTV_MPC：主图数值。原始数据/LTV_MPC/B201/cilqr：全状态、参考、控制周期、参数和QP快照。\n输入快照/full/B201：冻结原规划结果。核验结果：独立几何、精度、参考完整性和复现检查。\n运行“复现实验.ps1”将输入和代码复制到新目录，再重跑B201、核验、绘图及汇总；拒绝覆盖已存在目录。“仅重绘图片.ps1”只重绘正式数据。\n环境：Python 3.11，依赖版本见代码/requirements.txt。\n代码/tracking_experiment.py仅提供原参数与参考读取辅助函数；代码/plot_tracking.py仅提供既有绘图样式。本实验执行入口是ltv_mpc_tracking.py，未使用旧TVLQR控制器。\n"
    (ROOT / '实验结果说明.txt').write_text(text, encoding='utf-8-sig')
    print('Updated B201 nominal LTV-MPC summary and experiment notes.')
if __name__ == '__main__':
    main()