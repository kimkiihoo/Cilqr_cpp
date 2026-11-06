"""开销评测模块的项目根目录与基准数据目录路径自适应定位工具。"""
from pathlib import Path

def project_root(start=Path(__file__).resolve().parent):
    """定位项目根目录路径。"""
    for candidate in (Path(start), *Path(start).parents):
        if (candidate / 'A02_对照组代码').is_dir() or (candidate / '03_实验数据').is_dir():
            return candidate
    raise FileNotFoundError('Cannot locate the project root')

def baseline_directory(root=None):
    """定位冻结基准测试数据根目录。"""
    root = Path(root) if root else project_root()
    candidates = [root / 'A02_对照组代码/仿真数据/原版算法三方法基准_20261104/full', root / '03_实验数据/原版算法三方法基准_20261104/full']
    for candidate in candidates:
        if (candidate / 'B201/scene/scene.json').is_file():
            return candidate
    raise FileNotFoundError('Cannot locate the frozen B201 benchmark; pass --baseline-dir')

def reference_run_directory(root=None):
    """定位参考运行结果数据目录。"""
    root = Path(root) if root else project_root()
    candidates = [root / 'A02_对照组代码/仿真数据/HybridA_ST_B201_20261105', root / '03_实验数据/HybridA_ST_B201_20261105']
    for candidate in candidates:
        if (candidate / 'st_search.json').is_file():
            return candidate
    raise FileNotFoundError('Cannot locate the B201 Hybrid A* + ST* run; pass --reference-run')