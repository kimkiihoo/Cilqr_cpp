"""多方法全场景仿真基准测试结果后处理与综合对比表格生成脚本。"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import zipfile
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
DATA = ROOT / '03_实验数据/原版算法三方法基准_20261104/full'
FIG = ROOT / '06_重绘图片/原版算法三方法基准_20261104'

def sha(path):
    """计算文件哈希值。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_csv(path):
    """读取 CSV 文件。"""
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def main():
    """整合三方法基准数据，计算关键统计指标并输出对比表格。"""
    audit = json.loads((DATA / 'independent_record_audit.json').read_text(encoding='utf-8'))
    assert audit['record_audit_pass'] and len(audit['runs']) == 9
    rows = []
    names = {'r': 'R-CILQR', 'al': 'AL-iLQR', 'sqp': 'SQP-NMPC (original R penalty)'}
    folders = {'r': 'cilqr', 'al': 'alilqr', 'sqp': 'sqp'}
    values_checked = 0
    for run in audit['runs']:
        m, method = (run['map'], run['method'])
        folder = folders[method]
        raw = read_csv(DATA / m / folder / 'cycles.csv')
        times = np.array([float(r['solve_time_ms']) for r in raw])
        rows.append({'map': m, 'method': names[method], 'N': 80, 'dt_s': 0.1, 'reference_speed_mps': {'B201': 5, 'B207': 6, 'B301': 15}[m], 'attempted_frames': run['attempts'], 'executed_frames': run['executed'], 'mean_computation_ms': float(times.mean()), 'p95_computation_ms': float(np.percentile(times, 95)), 'max_computation_ms': float(times.max()), 'over100ms_frames': int(sum(times > 100)), 'final_goal_distance_m': run['final_goal_distance_m'], 'original_residual_over1e_3_frames': run['original_constraint_residual_over_1e_3_frames'], 'sqp_success_frames': run['SQP_success_count'], 'sqp_failed_but_executed_frames': run['SQP_failed_but_executed_count'], 'sqp_hard_residual_over1e_6_frames': run['SQP_hard_residual_over_1e_6_frames']})
        for kind, fields in [('control_outputs', ['velocity', 'gamma_dot']), ('solve_times', ['solve_time_ms'])]:
            plotted = read_csv(FIG / 'figure_data' / f'{m}_{kind}.csv')
            raw_by_frame = {int(r['frame_index']): r for r in raw}
            for p in plotted:
                original = raw_by_frame.get(int(p['frame_index']))
                if original is None:
                    assert all((p[f'{folder}_{field}'] == '' for field in fields))
                    continue
                for field in fields:
                    assert float(p[f'{folder}_{field}']) == float(original[field])
                    values_checked += 1
    with (DATA / 'summary_table.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    files = []
    for mapname in ['B201', 'B207', 'B301']:
        for kind in ['Control_Output', 'Solve_Time']:
            for ext in ['png', 'pdf', 'svg']:
                path = FIG / f'{mapname}_{kind}_Comparison.{ext}'
                if ext == 'png':
                    with Image.open(path) as im:
                        assert all((abs(d - 600) < 0.1 for d in im.info['dpi']))
                elif ext == 'pdf':
                    assert path.read_bytes().startswith(b'%PDF')
                else:
                    tree = ET.parse(path)
                    texts = [''.join(e.itertext()) for e in tree.iter() if e.tag.endswith('}text')]
                    assert texts.count('Frame Index') == (2 if kind == 'Control_Output' else 1)
                    assert all((label in texts for label in ['R-CILQR', 'AL-iLQR', 'SQP-NMPC']))
                files.append({'path': str(path.relative_to(FIG)), 'sha256': sha(path)})
    (FIG / 'figure_export_audit.json').write_text(json.dumps({'passed': True, 'values_checked': values_checked, 'files': files, 'visual_review': 'Final control and timing contact sheets inspected before delivery'}, ensure_ascii=False, indent=2), encoding='utf-8')
    note = '原交接代码三算法基准：2026-11-04\n\n本批以用户指定的05_原始交接代码/Cilqr_cpp-CAV/Cilqr_cpp-CAV为唯一算法来源，重新建立可追溯基准。原R-CILQR和AL-iLQR优化器保持原样，SQP-NMPC直接调用原R的运动学模型、参考点匹配、罚函数目标和原约束残差定义。本批不是遗失的论文逐帧历史数据，也没有证据证明旧图已精确复原。\n\n设置依据\nN=80、dt=0.1 s、Q/R依据论文表1。B201/B207/B301采用5/6/15 m/s的重建设置，依据旧图量级及现存代码派生角速率阈值的机制；5/6是候选参数，没有历史配置确证。三场景固定观察窗口分别400/300/200帧。初态、参考路径和障碍沿用已核验的原代码冻结场景。参数证据见parameter_evidence.json。B201另保留N50诊断结果；它没有消除与旧图的差异，因此正式批次仍用论文N80，未按曲线外观调整权重。\n\nMPC对齐范围\nSQP使用SLSQP单次射击，maxiter=100、ftol=1e-6，直接评价原R精确罚函数目标。保留nearest+1参考点随候选轨迹重新匹配、预测内部速度平滑、原N点障碍预测和R近终点累计减速；没有加入前后车体包络、额外1/2或新的实际上一拍输入罚项。\n原R和AL本来就有不同的硬限幅、终点减速和预测gamma裁剪规则，原样保留。SQP使用原R外控制边界及预测内部速度差约束；几何和名义角速率仍由原R软罚函数处理，不能称为所有几何约束硬满足。R逐次线搜索的首速度变化区间是算法内部规则，SQP不伪造一个固定前一拍锚点来替代。详情见复现代码README。\n\n运行和计时\n各方法顺序运行、使用独立状态。每0.1 s直接执行返回的有限首控制，不加LQR或执行器滞后；SLSQP返回失败时也保留结果、执行策略和状态码。图中所有帧和峰值均保留。\n三方法统一记录周期内准备/初始化/优化的wall时间，排除障碍预测、额外成本残差检查、执行与文件导出。原R/AL内部日志及AL的std::async原样保留，因此结果属于当前实现与本机，不代表单线程理论复杂度。不能与此前clock计时合并。\n\n指标解释\n固定窗口完成不等于成功到达。summary_table.csv报告实际终点距离、全部耗时、原约束残差和SLSQP状态。原残差包含车道、障碍、铰接角和角速率等项，既不是碰撞次数，也不统一具有米的单位。SLSQP成功退出不等于原软约束残差为零。原AL的收敛返回字段是默认值，未用于计算收敛率；原代价另行重算。\n本批为规划输入的运动学执行，不是独立跟踪控制验证，也没有以原残差替代整车栅格碰撞检查。\n\n图表\n三地图各提供控制双子图和计算耗时图，包含R-CILQR、AL-iLQR、SQP-NMPC三条曲线，PNG为600 dpi，另有PDF/SVG。上下两个控制子图都显示Frame Index和数字刻度。折线直接连接原始点，不做平滑、缩放或删帧。figure_data保留直接绘图数据。\n\n复现\n当前完整项目根目录的“复现原版算法三方法.ps1”可编译原模型接口、重新运行三场景并核验和出图。运行器读取项目中已经核验的冻结基准场景，需保留当前项目目录结构。包内保存实际运行源码、二进制、配置、逐帧预测/控制、尝试日志和审计；不能把代码生成的候选参数说明改称历史记录。\n'
    for target in [DATA, FIG]:
        (target / '结果说明.txt').write_text(note, encoding='utf-8-sig')
        shutil.copy2(WORK / 'parameter_evidence.json', target / 'parameter_evidence.json')
    verification = DATA / 'verification'
    verification.mkdir(exist_ok=True)
    src = ROOT / '02_对照组代码/原模型MPC对照/verification'
    for path in src.glob('*.json'):
        shutil.copy2(path, verification / path.name)
    for name in ['plot_original_model_comparison.py', 'finalize_comparison.py', 'parameter_evidence.txt']:
        shutil.copy2(WORK / name, DATA / 'reproduction' / name)
    shutil.copy2(ROOT / '复现原版算法三方法.ps1', DATA / 'reproduction/复现原版算法三方法.ps1')
    shutil.copy2(ROOT / '.codex_work/reviewer_experiments_20261103/audit_original_model_mpc.py', DATA / 'reproduction/audit_original_model_mpc.py')
    diagnostic = ROOT / '03_实验数据/原版基准核验_20261104'
    shutil.copy2(diagnostic / 'independent_curve_features.json', verification / 'N50_N80_baseline_features.json')
    archives = []
    for archive, groups in [(ROOT / '06_重绘图片/原版算法三方法基准_20261104_图表.zip', [(FIG, 'figures')]), (ROOT / '03_实验数据/原版算法三方法基准_20261104_完整结果.zip', [(DATA, 'data'), (FIG, 'figures')])]:
        if archive.exists():
            raise FileExistsError(archive)
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for folder, label in groups:
                for path in sorted(folder.rglob('*')):
                    if path.is_file() and '__pycache__' not in path.parts and (not path.name.endswith('_preview.png')):
                        z.write(path, Path(label) / path.relative_to(folder))
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            count = len(z.namelist())
        archives.append({'path': str(archive), 'files': count, 'bytes': archive.stat().st_size, 'sha256': sha(archive)})
    print(json.dumps({'plot_values_checked': values_checked, 'runs': rows, 'archives': archives}, ensure_ascii=False))
if __name__ == '__main__':
    main()