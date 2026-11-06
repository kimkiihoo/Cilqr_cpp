"""项目工程各目录、可执行程序与数据路径自适应解析定位工具模块。"""
from pathlib import Path

def project_root(start=None):
    """定位项目根目录路径。"""
    directory = Path(start or __file__).resolve()
    if directory.is_file():
        directory = directory.parent
    for folder in (directory, *directory.parents):
        if (folder / 'AGENTS.md').is_file():
            return folder
        if (folder / 'A01_原始论文与原始代码/原始代码/Cilqr_cpp-CAV/Cilqr_cpp-CAV').is_dir():
            return folder
        if (folder / '05_原始交接代码/Cilqr_cpp-CAV/Cilqr_cpp-CAV').is_dir():
            return folder
    raise FileNotFoundError('Cannot locate project root containing AGENTS.md or original code')

def existing(root, candidates):
    """校验并返回存在的路径。"""
    paths = [Path(root) / name for name in candidates]
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError('None of the resource locations exists: ' + ', '.join(map(str, paths)))

def original_dir(root):
    """定位原始 C++ 代码与地图工程目录。"""
    return existing(root, ['A01_原始论文与原始代码/原始代码/Cilqr_cpp-CAV/Cilqr_cpp-CAV', '05_原始交接代码/Cilqr_cpp-CAV/Cilqr_cpp-CAV'])

def baseline_dir(root):
    """定位冻结基准测试数据根目录。"""
    return existing(root, ['A02_对照组代码/仿真数据/原版算法三方法基准_20261104/full', '03_实验数据/原版算法三方法基准_20261104/full'])

def baseline_map(root, mapname):
    """定位指定地图的基准数据文件夹。"""
    current = baseline_dir(root) / mapname
    if (current / 'scene/scene.json').is_file():
        return current
    old = {'B201': '03_实验数据/原版基准核验_20261104/B201_v5_N80/B201', 'B207': '03_实验数据/原版基准核验_20261104/B207_v6_N80/B207', 'B301': '03_实验数据/原算法原地图复跑_20261103/B301'}
    return existing(root, [old[mapname]])

def replay_executable(root):
    """定位已编译的原生场景复现可执行程序。"""
    return existing(root, ['A02_对照组代码/原论文场景复现/build/paper_replay.exe', '02_对照组代码/原论文场景复现/build/paper_replay.exe'])

def results_dir(root):
    """定位结果导出目录。"""
    root = Path(root)
    return root / ('A02_对照组代码/仿真数据' if (root / 'A02_对照组代码').is_dir() else '03_实验数据')