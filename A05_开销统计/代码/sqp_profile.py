"""SLSQP 求解器调用耗时与内迭代细分统计辅助模块。"""
import time
from contextlib import contextmanager
import scipy.optimize._slsqp_py as slsqp_module
from scipy.optimize._differentiable_functions import ScalarFunction

@contextmanager
def profile_slsqp():
    """包装 SLSQP 求解器并记录目标函数与梯度调用的耗时分解。"""
    p = {k: 0.0 for k in ['objective_gradient_ms', 'constraint_jacobian_ms', 'slsqp_core_ms', 'constraint_values_ms']}
    p['core_calls'] = 0
    originals = []

    def install(obj, name, key):
        original = getattr(obj, name)

        def measured(*args, **kwargs):
            t = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                p[key] += (time.perf_counter() - t) * 1000
                if key == 'slsqp_core_ms':
                    p['core_calls'] += 1
        originals.append((obj, name, original))
        setattr(obj, name, measured)
    install(ScalarFunction, '_update_grad', 'objective_gradient_ms')
    install(slsqp_module, '_eval_con_normals', 'constraint_jacobian_ms')
    install(slsqp_module, '_eval_constraint', 'constraint_values_ms')
    install(slsqp_module, 'slsqp', 'slsqp_core_ms')
    t = time.perf_counter()
    try:
        yield p
    finally:
        p['minimize_ms'] = (time.perf_counter() - t) * 1000
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)
        p['other_minimize_ms'] = p['minimize_ms'] - sum((p[k] for k in ['objective_gradient_ms', 'constraint_jacobian_ms', 'slsqp_core_ms', 'constraint_values_ms']))