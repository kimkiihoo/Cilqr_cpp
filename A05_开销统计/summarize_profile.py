from pathlib import Path
import csv, json, statistics as st
import numpy as np
from profile_paths import baseline_directory
OUT = Path(__file__).resolve().parent
RUN = OUT / 'B201_正式统计'
DEST = OUT / 'B201_结果汇总'
DEST.mkdir(exist_ok=True)
NAMES = {'cilqr': 'R-CILQR', 'alilqr': 'AL-iLQR', 'sqp': 'SQP-NMPC'}
PHASES = ['linearization_ms', 'derivatives_ms', 'backward_ms', 'forward_ms', 'cost_ms', 'al_update_ms']

def write_csv(name, rows):
    """保存字典行数据为 CSV 表格。"""
    keys = list(dict.fromkeys((k for r in rows for k in r)))
    with (DEST / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
cycle_rows = []
iteration_rows = []
summaries = {}
for method, name in NAMES.items():
    p = RUN / 'B201' / method
    attempts = [json.loads(x) for x in (p / 'attempts.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(attempts) == 400
    records = []
    native_iterations = []
    for a in attempts:
        r = a['result_metadata']
        t = r['timing_profile']
        row = {'method': name, 'frame_index': a['frame_index'], 'solver_wall_ms': r['solver_wall_ms']}
        if method != 'sqp':
            row.update({k: v for k, v in t.items() if k != 'iteration_timings'})
            row['other_cycle_ms'] = r['solver_wall_ms'] - sum((t[k] for k in PHASES))
            row['inner_loop_wall_ms'] = sum((x['wall_ms'] for x in t['iteration_timings']))
            row['inner_mean_ms'] = row['inner_loop_wall_ms'] / t['iterations']
            for x in t['iteration_timings']:
                iteration_rows.append({'method': name, 'frame_index': a['frame_index'], **x})
                native_iterations.append(x)
            if method == 'cilqr':
                assert r['native_fields']['iterations'] == t['iterations']
        else:
            row.update(t)
            row['iterations'] = r['iterations']
            row['objective_calls'] = r['objective_calls']
            row['status'] = r['status']
            row['gradient_evaluations'] = r['njev']
            row['prepare_ms'] = r['prepare_wall_ms']
            row['init_and_wrapper_ms'] = r['solver_wall_ms'] - t['minimize_ms'] - r['prepare_wall_ms']
        row['cycle_ms_per_iteration'] = row['solver_wall_ms'] / row['iterations'] if row['iterations'] else None
        records.append(row)
        cycle_rows.append(row)
    total_iters = sum((r['iterations'] for r in records))
    s = {'method': name, 'cycles': 400, 'iterations_total': total_iters, 'iterations_mean': total_iters / 400, 'iterations_median': st.median((r['iterations'] for r in records)), 'iterations_max': max((r['iterations'] for r in records)), 'solver_wall_mean_ms': st.mean((r['solver_wall_ms'] for r in records)), 'solver_wall_median_ms': st.median((r['solver_wall_ms'] for r in records)), 'solver_wall_p95_ms': float(np.percentile([r['solver_wall_ms'] for r in records], 95)), 'solver_wall_max_ms': max((r['solver_wall_ms'] for r in records)), 'amortized_cycle_ms_per_iteration': sum((r['solver_wall_ms'] for r in records)) / total_iters}
    if method != 'sqp':
        s.update({'outer_iterations_mean': st.mean((r['outer_iterations'] for r in records)), 'outer_iterations_max': max((r['outer_iterations'] for r in records)), 'al_updates_total': sum((r['al_updates'] for r in records)), 'measured_inner_iteration_mean_ms': st.mean((r['wall_ms'] for r in native_iterations))})
        for k in PHASES + ['other_cycle_ms', 'inner_loop_wall_ms']:
            s[k.replace('_ms', '_mean_ms')] = st.mean((r[k] for r in records))
    else:
        for k in ['objective_gradient_ms', 'constraint_jacobian_ms', 'slsqp_core_ms', 'constraint_values_ms', 'other_minimize_ms', 'minimize_ms', 'prepare_ms', 'init_and_wrapper_ms']:
            s[k.replace('_ms', '_mean_ms')] = st.mean((r[k] for r in records))
        s['amortized_minimize_ms_per_major_iteration'] = sum((r['minimize_ms'] for r in records)) / total_iters
        s['objective_calls_mean'] = st.mean((r['objective_calls'] for r in records))
        s['gradient_evaluations_mean'] = st.mean((r['gradient_evaluations'] for r in records))
        s['status_counts'] = {str(k): sum((r['status'] == k for r in records)) for k in sorted(set((r['status'] for r in records)))}
    summaries[method] = s
write_csv('01_逐周期计算开销.csv', cycle_rows)
write_csv('02_R与AL逐次内迭代计时.csv', iteration_rows)
write_csv('03_三方法开销汇总.csv', [{k: v for k, v in s.items() if not isinstance(v, dict)} for s in summaries.values()])
(DEST / 'summary.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
r, a, q = [summaries[k] for k in NAMES]
speedup = (1 - r['solver_wall_mean_ms'] / a['solver_wall_mean_ms']) * 100
draft = f"建议插入位置\nrevised_manuscript.docx 第4.3节 Performance analysis，Computational Efficiency 段落之后。\n建议用下列段落替换该小节中以“The computational efficiency of R-CILQR benefits from both the single-loop optimization structure and the warm-start strategy.”开头的机制解释段；原段中关于900周期最大耗时的结果可独立保留为原基准结果。\n\n英文正文（仅限B201；分项均为每个规划周期平均耗时）\nTo examine the computational cost in more detail, the B201 dynamic obstacle avoidance scenario was rerun with timing instrumentation using the original benchmark implementations and settings (N = 80 and Δt = 0.1 s) on an Intel Core i7-12700H processor. All 400 planning cycles per method were retained, and the returned control sequences and predicted trajectories matched the uninstrumented benchmark exactly. R-CILQR and AL-iLQR required an average of {r['iterations_mean']:.3f} and {a['iterations_mean']:.3f} inner iterations per planning cycle, with measured mean inner-iteration times of {r['measured_inner_iteration_mean_ms']:.3f} and {a['measured_inner_iteration_mean_ms']:.3f} ms, respectively. Their mean per-cycle times for model linearization, cost-derivative evaluation, backward recursion, and forward rollout/line search were {r['linearization_mean_ms']:.3f}/{a['linearization_mean_ms']:.3f}, {r['derivatives_mean_ms']:.3f}/{a['derivatives_mean_ms']:.3f}, {r['backward_mean_ms']:.3f}/{a['backward_mean_ms']:.3f}, and {r['forward_mean_ms']:.3f}/{a['forward_mean_ms']:.3f} ms, respectively (R-CILQR/AL-iLQR). Additional cost evaluation and native logging outside the forward pass took {r['cost_mean_ms']:.3f}/{a['cost_mean_ms']:.3f} ms, while the remaining per-cycle overhead was {r['other_cycle_mean_ms']:.3f}/{a['other_cycle_mean_ms']:.3f} ms, giving total mean planning times of {r['solver_wall_mean_ms']:.3f}/{a['solver_wall_mean_ms']:.3f} ms. AL-iLQR completed every cycle within its first outer iteration, with no multiplier updates. Thus, the observed runtime difference in this scenario is associated with fewer inner iterations and lower measured iteration cost, rather than repeated outer-loop updates. These measurements retain the original native logging and AL-iLQR parallel implementation and are specific to the tested implementation and hardware.\n\n可选SQP补充句（若希望三方法都给出剖析）\nFor SQP-NMPC, the average number of major iterations was {q['iterations_mean']:.3f}, and the mean planning time was {q['solver_wall_mean_ms']:.3f} ms. Objective-gradient evaluation and the SLSQP core accounted for {q['objective_gradient_mean_ms']:.3f} and {q['slsqp_core_mean_ms']:.3f} ms per planning cycle, respectively. The amortized minimization time was {q['amortized_minimize_ms_per_major_iteration']:.3f} ms per major iteration; this quantity is not a directly measured iLQR inner-iteration time.\n\n中文含义\n本次以原版B201设置重新运行，在同一批400周期中核实迭代次数和分项耗时。R平均{r['iterations_mean']:.3f}次内迭代，AL平均{a['iterations_mean']:.3f}次；平均实际单次内迭代耗时为{r['measured_inner_iteration_mean_ms']:.3f}和{a['measured_inner_iteration_mean_ms']:.3f} ms。AL每周期均只运行1次外层，没有乘子更新。因此B201不能用于证明“多次外层更新造成了时间差”，应将结论写成：该场景中R迭代更少，测得的单次迭代开销也更低。\n\n与原文衔接\n1. 本次是新增计时的独立复跑，不将新总耗时与原表2的2.41/18.50/513.46 ms拼成同一组分解。可保留表2并明确本段为独立profiling rerun；如替换表2，应连同跨三场景汇总和降幅一起重算，不能只替换单个数值。\n2. 本机实际CPU为i7-12700H；不要在新增段落继续写i5-12400F。原文原基准CPU描述另需核实。\n3. 此处没有进行热启动消融，不能把计时差独立归因于热启动。\n4. 这里的相同是计时版与各自原版输出一致；并不表示R/AL/SQP彼此使用完全相同的约束执行规则。\n5. 每次内迭代时间=所有真实内迭代墙钟时间总和/实际内迭代总数；不是各周期单次时间的简单平均，也不是总规划时间/迭代数。\n"
(DEST / '04_正文补充段落_中英文.txt').write_text(draft, encoding='utf-8-sig')
lines = ['B201计算开销统计', '', f'数据：{OUT.name}/B201_正式统计；每方法400周期，保留失败返回和所有峰值。', '原基准：' + str(baseline_directory()) + '。', '硬件：Intel Core i7-12700H。Python 3.11；实际版本及源文件哈希见protocol.json。', 'N=80；Δt=0.1 s；原B201参考速度5 m/s；原路径、初态、障碍、权重、终止规则、原R日志与AL std::async均保留。', '一次独立复跑；不是重复试验的统计显著性结论。计时含轻量记录开销，计时序列会受系统负载影响。', '', '方法 | 周期数 | 平均迭代次数 | 平均总耗时(ms) | 最大总耗时(ms) | 实测内迭代耗时(ms)']
for s in summaries.values():
    inner = f"{s['measured_inner_iteration_mean_ms']:.6f}" if 'measured_inner_iteration_mean_ms' in s else '不适用(SQP为主迭代)'
    lines.append(f"{s['method']} | 400 | {s['iterations_mean']:.6f} | {s['solver_wall_mean_ms']:.6f} | {s['solver_wall_max_ms']:.6f} | {inner}")
lines += ['', f'本次R相对AL平均总耗时降低{speedup:.4f}%。此值只属于本次B201。', 'AL外层迭代：每周期1次；乘子更新总数0。', '', '计时口径：', '总规划墙钟=原solve调用的实际耗时，包括周期准备、名义初始化、优化与原生日志；排除障碍预测、额外残差诊断、执行、导出。', '线性化：compute_df，单独测量模型状态/控制Jacobian。', '代价导数：compute_cost_derivatives / compute_al_derivatives，含AL并行任务创建及等待。', '反向：backward，包含原实现的矩阵分解、正则化、递推与原生输出。', '前向：forward，含所有线搜索尝试、轨迹滚动、候选代价与原生输出。', '额外代价：solve中初始化/外层初值和每次内迭代之后的代价计算，含相应原R文件写入，不重复计入forward内部代价。', 'AL更新：update_constraints；本场景没有触发。', '其他：总规划时间减去上述互斥阶段，包含准备、初始预测、收敛判断、约束检查、结果复制与计时记录开销。', '实测内迭代：内循环整个循环体，含所有退出路径；排除循环外准备、初始代价、AL外层检查及乘子更新。', '摊销总耗时/迭代数：总规划墙钟之和/迭代次数之和，含非迭代开销，单独命名，不冒充实测内迭代。', 'SQP梯度：原SciPy ScalarFunction._update_grad的实际耗时，含有限差分所需的原目标和轨迹求值。', 'SQP核心：原slsqp调用的实际耗时，包括QP求解、BFGS和线搜索内部工作；不能称为纯QP求解时间。', 'SQP约束Jacobian与约束值单列；其余minimize开销保留。SQP单次主迭代开销为minimize总时间/实际主迭代总数。', '', '输出一致性与分项相加核验：见B201_正式统计/output_equivalence_audit.json。', '重新统计：py -3.11 summarize_profile.py；重新核验：py -3.11 audit_outputs.py。', '重新运行：run_profile.py默认只运行B201，结果目录已存在时拒绝覆盖。请先在脚本中指定新的输出目录。']
(DEST / '00_结果与计时口径说明.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8-sig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={'width_ratios': [1.1, 1]})
for method, color, ls in [('R-CILQR', '#245d9a', '-'), ('AL-iLQR', '#bc6a1b', '--')]:
    rows = [x for x in cycle_rows if x['method'] == method]
    axes[0].plot([x['frame_index'] for x in rows], [x['iterations'] for x in rows], label=method, color=color, ls=ls, lw=1.1)
axes[0].set(xlabel='Planning cycle', ylabel='Actual inner iterations', xlim=(0, 399), ylim=(0, None), title='(a) Iteration counts')
axes[0].set_xticks([0, 100, 200, 300, 399])
axes[0].legend(frameon=False, loc='upper right')
labels = ['Linearization', 'Cost derivatives', 'Backward recursion', 'Forward / line search', 'Extra cost evaluation', 'Other overhead']
keys = ['linearization_mean_ms', 'derivatives_mean_ms', 'backward_mean_ms', 'forward_mean_ms', 'cost_mean_ms', 'other_cycle_mean_ms']
colors = ['#80a9cb', '#245d9a', '#b69b47', '#bc6a1b', '#dba875', '#9b9b9b']
bottom = np.zeros(2)
for label, key, color in zip(labels, keys, colors):
    values = np.array([r[key], a[key]])
    axes[1].bar([0, 1], values, bottom=bottom, label=label, color=color, width=0.5, edgecolor='white', linewidth=0.4)
    bottom += values
for i, v in enumerate(bottom):
    axes[1].text(i, v + max(bottom) * 0.02, f'{v:.3f} ms', ha='center', fontsize=10)
axes[1].set_xticks([0, 1], ['R-CILQR', 'AL-iLQR'])
axes[1].set(ylabel='Mean time per planning cycle (ms)', ylim=(0, max(bottom) * 1.18), title='(b) Timing breakdown')
axes[1].legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1))
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2)
    ax.set_axisbelow(True)
fig.suptitle('B201 dynamic obstacle avoidance: 400 cycles per method', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
for ext in ['png', 'pdf', 'svg']:
    fig.savefig(DEST / f'05_B201_iterations_and_timing.{ext}', dpi=300, bbox_inches='tight')
plt.close(fig)
print(json.dumps(summaries, ensure_ascii=False, indent=2))