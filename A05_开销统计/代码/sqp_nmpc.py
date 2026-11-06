"""基于 SciPy SLSQP 单打靶法 (Single-Shooting) 的非线性模型预测控制器 (SQP-NMPC)。

算法原理与优化构建:
1. 优化变量: 预测时域 N 步的控制序列 u \\in \\mathbb{R}^{N \\times 2}，包含速度指令 v [m/s] 与角速度指令 \\dot{\\gamma} [rad/s]；
2. 优化目标: 调用 C++ 原版 R-CILQR 代价函数作为精确目标函数；
3. 约束条件: 控制输入盒式界限与相邻预测步间加速度线性约束。
"""
import time
from sqp_profile import profile_slsqp
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from original_backend import OriginalBackend

def solve_prepared(backend: OriginalBackend, initial_controls=None, maxiter=100, ftol=1e-06):
    """对已配置初态与障碍物的后端执行 SLSQP 单打靶非线性模型预测控制求解。"""
    meta = backend.metadata
    if meta['usage'] != 1 or meta['profile'] != 1:
        raise ValueError('Prepare a dedicated evaluator with target=r before calling SQP')
    if maxiter < 1 or ftol <= 0:
        raise ValueError('maxiter and ftol must be positive')
    N = backend.N
    started = time.perf_counter()
    seed = backend.nominal()['controls'] if initial_controls is None else np.asarray(initial_controls, dtype=float)
    if seed.shape != (N, 2) or not np.isfinite(seed).all():
        raise ValueError('Initial controls must have finite shape (N,2)')
    lower = np.tile([-5.0, -1.5], N)
    upper = np.tile([30.0, 1.5], N)
    differences = np.zeros((N - 1, 2 * N))
    for i in range(N - 1):
        differences[i, 2 * i] = -1.0
        differences[i, 2 * (i + 1)] = 1.0
    constraints = [LinearConstraint(differences, meta['acc_min'] * meta['dt'], meta['acc_max'] * meta['dt'])]
    calls = 0

    def objective(flat):
        nonlocal calls
        calls += 1
        return backend.cost(flat.reshape(N, 2), kind='r_exact')
    with profile_slsqp() as timing_profile:
        result = minimize(objective, seed.ravel(), method='SLSQP', bounds=Bounds(lower, upper), constraints=constraints, options={'maxiter': int(maxiter), 'ftol': float(ftol), 'disp': False})
    elapsed = time.perf_counter() - started
    controls = np.asarray(result.x).reshape(N, 2)
    if np.isfinite(controls).all():
        evaluation = backend.evaluate(controls)
    else:
        evaluation = {'states': np.full((N + 1, 4), np.nan), 'controls': controls, 'matched_references': np.full((N + 1, 3), np.nan), 'residuals': np.array([np.nan]), 'r_exact_cost': float(result.fun), 'r_base_cost': np.nan, 'al_base_cost': np.nan, 'al_augmented_cost': np.nan, 'original_al_max_residual': np.nan, 'max_residual': np.nan, 'metadata': backend.metadata, 'evaluation_error': 'nonfinite returned control sequence'}
    dv = np.diff(controls[:, 0]) / meta['dt']
    hard_residual = max(0.0, float((lower - result.x).max()), float((result.x - upper).max()), float((dv - meta['acc_max']).max()), float((meta['acc_min'] - dv).max())) if np.isfinite(controls).all() else np.nan
    return {**evaluation, 'method': 'SQP-NMPC/original-R-penalty', 'scipy_success': bool(result.success), 'status': int(result.status), 'message': str(result.message), 'iterations': int(result.nit), 'timing_profile': timing_profile, 'objective_calls': calls, 'nfev': int(result.nfev), 'njev': int(getattr(result, 'njev', -1)), 'initialization_and_minimize_ms': elapsed * 1000, 'prepare_wall_ms': meta['prepare_wall_ms'], 'solver_wall_ms': elapsed * 1000 + meta['prepare_wall_ms'], 'solve_wall_ms': elapsed * 1000 + meta['prepare_wall_ms'], 'hard_control_max_residual': hard_residual, 'soft_geometry_residuals_are_not_hard_constraints': True, 'objective': 'Unchanged CILQRSolver::cal_cost_with_logging evaluated after unchanged SystemModel::dynamics', 'known_algorithm_difference': 'Original R line search bounds first speed relative to its current nominal first speed at every iteration; this is not a fixed feasible-set constraint and is not represented as an actual-previous-input bound or a frozen guessed anchor in SLSQP.', 'trajectory_difference': 'SQP uses exact un-clipped dynamics; the original cold-start generator and AL forward pass may clip predicted articulation. Original baselines preserve their own behavior.', 'options': {'maxiter': int(maxiter), 'ftol': float(ftol)}}