"""汇总 B201 原始时序轨迹的 LTV-MPC 跟踪数据及独立核验结果。

核对数据来源，保留通过与未通过的实际结果，导出 CSV 和中文说明。
"""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    raw = ROOT / '原始数据/LTV_MPC/B201/cilqr'
    summary = json.loads((raw / 'summary.json').read_text(encoding='utf-8'))
    audit = json.loads((ROOT / '核验结果/ltv_mpc_independent_audit_B201.json').read_text(encoding='utf-8'))
    assert summary['map'] == audit['map'] == 'B201'
    assert summary['case'] == audit['case'] == 'nominal'
    assert summary['reference_mode'] == audit['reference_mode'] == 'original_time'
    for filename, key in [('nominal.csv', 'actual_csv_sha256'),
                          ('original_reference.csv', 'original_reference_sha256'),
                          ('sampled_reference.csv', 'sampled_reference_sha256')]:
        current_hash = hashlib.sha256((raw / filename).read_bytes()).hexdigest()
        assert current_hash == summary[key], f'Changed source: {filename}'
        assert any(h['sha256'] == current_hash for h in audit['input_hashes']), filename
    geometry = audit['dense_actual_clock_geometry']
    corridor = audit['dense_global_path_corridor_diagnostic']
    timing = audit['controller_timing']
    metrics = audit['time_matched_metrics']
    row = {'map': 'B201', 'planner': 'R-CILQR', 'controller': 'LTV-MPC',
           'case': 'nominal', 'reference_mode': summary['reference_mode'],
           'original_duration_s': summary['original_duration_s'],
           'duration_s': summary['duration_s']}
    names = {'front_lateral_error_m': 'front_lateral', 'rear_lateral_error_m': 'rear_lateral',
             'front_longitudinal_error_m': 'front_longitudinal',
             'front_position_error_m': 'front_position', 'rear_position_error_m': 'rear_position'}
    for source, prefix in names.items():
        row[f'{prefix}_rmse_m'] = metrics[source]['rmse']
        row[f'{prefix}_max_abs_m'] = metrics[source]['max_abs']
    row.update({'speed_rmse_mps': metrics['speed_error_mps']['rmse'],
                'speed_max_abs_mps': metrics['speed_error_mps']['max_abs'],
                'dense_min_body_clearance_m': geometry['minimum_clearance_m'],
                'dense_collision_samples': geometry['zero_clearance_or_collision_samples'],
                'dense_corner_corridor_violation_m': corridor['joint_extended_corner_max_violation_m'],
                'goal_distance_m': summary['final_goal_distance_m'],
                'terminal_speed_mps': summary['final_speed_mps'],
                'qp_failures': timing['qp_failures'], 'fallback_cycles': timing['fallback_cycles'],
                'controller_mean_ms': timing['controller_mean_s'] * 1000,
                'controller_max_ms': timing['controller_max_s'] * 1000,
                'deadline_exceedance_cycles': timing['deadline_exceedance_cycles'],
                'control_cycles': timing['cycles'], 'audit_passed': audit['audit_passed'],
                'engineering_targets_met': audit['acceptance']['all_engineering_targets_met'],
                'hard_real_time_demonstrated': audit['acceptance']['hard_real_time_demonstrated']})
    output = ROOT / '汇总结果'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'B201_LTV_MPC_正常工况_精简汇总.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    check_text = '通过' if audit['audit_passed'] else '未通过'
    targets_text = '达到' if row['engineering_targets_met'] else '未全部达到'
    violations = audit['actual_state_and_input_max_violations']
    violation_text = '；'.join(f'{key}={value:.6g}' for key, value in violations.items())
    text = f"""B201正常工况：R-CILQR原始时序轨迹 + LTV-MPC跟踪

实验范围
保留B201原规划的401个位姿节点、400段控制量及原始{summary['original_duration_s']:.1f} s时序。
参考位姿在原时间节点间插值，参考速度和铰接角速度按原控制周期保持；不重新安排速度或时间，也不按实际位置重置参考。
实际仿真时长{summary['duration_s']:.1f} s。初态[x,y,theta,gamma,v,gamma_dot]=[10,20,1.2,0,5,0]。
采用铰接车辆运动学及0.20/0.15 s一阶执行器滞后；控制周期0.05 s，积分步长0.01 s，预测60步/3 s。动态障碍按实际仿真时间运动。
跟踪器采用原R-CILQR相同的输入数值范围：铰接角速度硬包络为±1.5 rad/s。原规划器±0.15685 rad/s为软惩罚阈值，不能与硬截断范围混用；这些数值是仿真模型设定。

原时钟下完整跟踪结果（保留全部采样点，无平滑）
前轴横向RMSE：{row['front_lateral_rmse_m']:.6f} m；最大绝对误差：{row['front_lateral_max_abs_m']:.6f} m。
后轴横向RMSE：{row['rear_lateral_rmse_m']:.6f} m；最大绝对误差：{row['rear_lateral_max_abs_m']:.6f} m。
前轴纵向RMSE：{row['front_longitudinal_rmse_m']:.6f} m；最大绝对误差：{row['front_longitudinal_max_abs_m']:.6f} m。
速度RMSE：{row['speed_rmse_mps']:.6f} m/s；最大绝对误差：{row['speed_max_abs_mps']:.6f} m/s。
前/后轴位置RMSE：{row['front_position_rmse_m']:.6f}/{row['rear_position_rmse_m']:.6f} m。
误差由实际状态减同一时刻原始参考计算；横向和纵向分量分别投影到参考轴线的法向与切向。
独立0.002 s稠密核验：完整前后车体最小障碍净距{row['dense_min_body_clearance_m']:.6f} m，碰撞样本{row['dense_collision_samples']}个，车体角点最大走廊超界{row['dense_corner_corridor_violation_m']:.6g} m。
终点距目标(80,80)为{row['goal_distance_m']:.6f} m，末速度{row['terminal_speed_mps']:.6f} m/s。
QP失败{row['qp_failures']}次，回退{row['fallback_cycles']}次。
状态和输入最大数值越界：{violation_text}（各项单位及阈值见独立核验JSON）。
独立核验：{check_text}；既定验收指标：{targets_text}。核验错误：{audit['audit_errors']}。

计算时间
完整控制计算平均{row['controller_mean_ms']:.3f} ms、最大{row['controller_max_ms']:.3f} ms；{row['deadline_exceedance_cycles']}/{row['control_cycles']}个周期超过50 ms（{timing['deadline_exceedance_fraction']:.2%}）。仿真未注入计算墙钟延迟。
结论适用于所采用的运动学及执行器滞后模型；净距来自稠密采样几何检查。

文件与复现
图片/LTV_MPC/B201_R_CILQR_LTV_MPC_Tracking：Fig.13主图，左侧参考/实际轨迹，右上以不同颜色显示前后轴横向误差，右下对比实际纵向速度与参考速度；横轴为Time (s)，无局部放大。
图片/LTV_MPC/B201_R_CILQR_LTV_MPC_Timing_Check：速度跟踪及前轴纵向误差补充图，用于时序核验或可选补充材料。
两图均输出600 dpi PNG、PDF与SVG，保留全部0.01 s样本。绘图数据/LTV_MPC保存对应数值。
原始数据/LTV_MPC/B201/cilqr保存全状态、固定原参考、控制周期、参数和QP快照。核验结果保存独立几何、精度与参考完整性检查。
运行“复现实验.ps1”复制冻结输入与代码到新目录，重跑B201、核验、绘图及汇总；拒绝覆盖已有目录。
"""
    (ROOT / '实验结果说明.txt').write_text(text, encoding='utf-8-sig')
    print('Updated original-time B201 LTV-MPC summary and experiment notes.')


if __name__ == '__main__':
    main()
