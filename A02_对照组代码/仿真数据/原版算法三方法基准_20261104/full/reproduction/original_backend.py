"""Python 与原生 C++ 模型/求解器动态链接库 (original_model_bridge.dll) 的 ctypes 接口桥梁。

本模块提供与 C++ 底层内核的双向数据交互:
1. 加载并绑定 C++ 导出的原生 C API 函数指针签名；
2. 传递全局参考路径、场景障碍物与车辆配置参数；
3. 执行非线性模型单步前向积分、名义轨迹生成、目标函数与残差精确评估；
4. 调度执行原版 CILQRSolver (基于控制障碍函数/约束松弛惩罚) 与 ALILQRSolver (基于增广拉格朗日乘子法) 求解器。
"""
from pathlib import Path
import ctypes as C
import json
import os
import shutil
import tempfile
import time
import numpy as np
HERE = Path(__file__).resolve().parent
from project_paths import project_root, original_dir
ROOT = project_root(HERE)
ORIGINAL = original_dir(ROOT)
PTR = C.POINTER(C.c_double)
DLL = None
DLL_DIR = None
DLL_PATH = None

def library():
    """加载 C++ 动态链接库 original_model_bridge.dll 并绑定导出函数签名。"""
    global DLL, DLL_DIR, DLL_PATH
    if DLL is None:
        DLL_PATH = Path(os.environ.get('ORIGINAL_MODEL_BRIDGE_DLL', str(HERE / 'build/original_model_bridge.dll'))).resolve()
        directory = DLL_PATH.parent
        if hasattr(os, 'add_dll_directory'):
            DLL_DIR = os.add_dll_directory(str(directory))
        DLL = C.CDLL(str(DLL_PATH))
        signatures = {'ob_error': (C.c_char_p, []), 'ob_create': (C.c_void_p, [C.c_char_p, PTR, C.c_int, C.c_char_p]), 'ob_destroy': (None, [C.c_void_p]), 'ob_metadata': (C.c_char_p, [C.c_void_p]), 'ob_prepare': (C.c_int, [C.c_void_p, C.c_int, PTR, PTR, C.c_int]), 'ob_step': (C.c_int, [C.c_void_p, PTR, PTR, PTR]), 'ob_nominal': (C.c_int, [C.c_void_p, PTR, PTR]), 'ob_cost': (C.c_int, [C.c_void_p, PTR, C.c_int, PTR]), 'ob_evaluate': (C.c_int, [C.c_void_p, PTR, PTR, PTR, PTR, PTR]), 'ob_commit': (C.c_int, [C.c_void_p, PTR]), 'ob_original_solve': (C.c_int, [C.c_void_p, C.c_int, PTR, PTR, C.c_int, PTR, PTR, PTR])}
        for name, (restype, argtypes) in signatures.items():
            getattr(DLL, name).restype = restype
            getattr(DLL, name).argtypes = argtypes
    return DLL

def array(values, shape=None):
    """将输入数据转换为连续的高精度双精度浮点数组 (float64) 并校验数值有限性。"""
    out = np.ascontiguousarray(values, dtype=np.float64)
    if shape is not None:
        out = out.reshape(shape)
    if not np.isfinite(out).all():
        raise ValueError('Input contains nonfinite values')
    return out

def ptr(values):
    """获取 NumPy 数组的 ctypes 底层双精度浮点指针。"""
    return values.ctypes.data_as(PTR)

def read_configs(directory=ORIGINAL / 'config'):
    """从配置文件目录读取 main.json, ilqr.json, hybrid_astar.json 配置字典。"""
    return {name: json.loads((Path(directory) / (name + '.json')).read_text(encoding='utf-8')) for name in ['main', 'ilqr', 'hybrid_astar']}

class OriginalBackend:
    """C++ 桥接后端上下文管理器类，负责底层对象的创建、内存管理、配置加载与状态评估。"""

    def __init__(self, path, configs=None, log_directory=None):
        self.lib = library()
        self.path = array(path, (-1, 3))
        self.configs = configs if configs is not None else read_configs()
        self._temp = tempfile.TemporaryDirectory(prefix='original_mpc_')
        self._handle = None
        folder = Path(self._temp.name)
        for name in ['main', 'ilqr', 'hybrid_astar']:
            (folder / (name + '.json')).write_text(json.dumps(self.configs[name], ensure_ascii=True), encoding='utf-8')
        logs = Path(log_directory).resolve() if log_directory else folder / 'logs'
        logs.mkdir(parents=True, exist_ok=True)
        self.log_directory = logs
        try:
            config_bytes, log_bytes = (str(folder).encode('ascii'), str(logs).encode('ascii'))
        except UnicodeEncodeError as error:
            self._temp.cleanup()
            raise ValueError('Use an ASCII log directory; Python copies configs to an ASCII temporary directory') from error
        self._handle = self.lib.ob_create(config_bytes, ptr(self.path), len(self.path), log_bytes)
        if not self._handle:
            message = self.lib.ob_error().decode('utf-8', errors='replace')
            self._temp.cleanup()
            raise RuntimeError(message)
        self.N = int(self.metadata['N'])
        self.nobs = 0
        self.prepared = False

    def _check(self, code):
        if code < 0:
            raise RuntimeError(self.lib.ob_error().decode('utf-8', errors='replace'))
        return code

    @property
    def metadata(self):
        raw = self.lib.ob_metadata(self._handle)
        if raw is None:
            raise RuntimeError(self.lib.ob_error().decode('utf-8', errors='replace'))
        return json.loads(raw)

    def prepare(self, state, obstacles=(), target='r'):
        state, obs = (array(state, (4,)), array(obstacles, (-1, 4)))
        if target not in ['r', 'al']:
            raise ValueError('target must be r or al')
        self._check(self.lib.ob_prepare(self._handle, 1 if target == 'r' else 2, ptr(state), ptr(obs), len(obs)))
        self.nobs, self.prepared = (len(obs), True)
        return self.metadata

    def step(self, state, control):
        state, control = (array(state, (4,)), array(control, (2,)))
        out = np.empty(4)
        self._check(self.lib.ob_step(self._handle, ptr(state), ptr(control), ptr(out)))
        return out

    def nominal(self):
        states, controls = (np.empty((self.N + 1, 4)), np.empty((self.N, 2)))
        self._check(self.lib.ob_nominal(self._handle, ptr(states), ptr(controls)))
        return {'states': states, 'controls': controls}

    def cost(self, controls, kind='r_exact'):
        kinds = {'r_exact': 0, 'r_base': 1, 'al_base': 2}
        if kind not in kinds:
            raise ValueError('unknown cost kind')
        controls, cost = (array(controls, (self.N, 2)), np.empty(1))
        self._check(self.lib.ob_cost(self._handle, ptr(controls), kinds[kind], ptr(cost)))
        return float(cost[0])

    def evaluate(self, controls):
        controls = array(controls, (self.N, 2))
        states, costs = (np.empty((self.N + 1, 4)), np.empty(5))
        residual = np.empty((self.N + 1) * (4 + self.nobs) + 2 * self.N)
        refs = np.empty((self.N + 1, 3))
        count = self._check(self.lib.ob_evaluate(self._handle, ptr(controls), ptr(states), ptr(costs), ptr(residual), ptr(refs)))
        return {'states': states, 'controls': controls, 'matched_references': refs, 'r_exact_cost': float(costs[0]), 'r_base_cost': float(costs[1]), 'al_augmented_cost': float(costs[2]), 'al_base_cost': float(costs[3]), 'original_al_max_residual': float(costs[4]), 'residuals': residual[:count].copy(), 'max_residual': float(max(0.0, residual[:count].max(initial=0.0))), 'metadata': self.metadata}

    def commit(self, controls):
        """Retain an explicitly accepted SQP sequence for the original warm-start shifting rule."""
        controls = array(controls, (self.N, 2))
        self._check(self.lib.ob_commit(self._handle, ptr(controls)))

    def solve_original(self, method, state, obstacles=()):
        if method not in ['r', 'al']:
            raise ValueError('method must be r or al')
        state, obs = (array(state, (4,)), array(obstacles, (-1, 4)))
        states, controls, fields = (np.empty((self.N + 1, 4)), np.empty((self.N, 2)), np.empty(7))
        started = time.perf_counter()
        self._check(self.lib.ob_original_solve(self._handle, 1 if method == 'r' else 2, ptr(state), ptr(obs), len(obs), ptr(states), ptr(controls), ptr(fields)))
        return {'states': states, 'controls': controls, 'wall_s': time.perf_counter() - started, 'native_fields': dict(zip(['converged', 'iterations', 'final_cost', 'solve_time_ms'], fields[:4].tolist())), 'recomputed_native_cost': float(fields[4]), 'original_geometry_max_residual': float(fields[5]), 'solver_wall_ms': float(fields[6]), 'native_convergence_fields_available': method == 'r', 'metadata': self.metadata}

    def close(self):
        if self._handle:
            self.lib.ob_destroy(self._handle)
            self._handle = None
        self._temp.cleanup()

    def preserve_logs(self, destination):
        """Copy already-flushed original cost logs before closing their temporary context."""
        shutil.copytree(self.log_directory, Path(destination), dirs_exist_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()