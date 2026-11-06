"""复现与对照实验资源目录自适应定位工具。

通过层级向上探测特征标记文件 (AGENTS.md、原始代码目录等)，自适应定位:
1. 项目根目录 (project_root);
2. 原始交接工程目录 (original_dir);
3. 基准数据目录与三张测试场景地图目录 (baseline_dir, baseline_map);
4. 重放可执行程序二进制文件路径 (replay_executable);
5. 输出结果保存根目录 (results_dir)。
"""
from pathlib import Path


def project_root(start=None):
    """自适应向上查找包含项目特征标记文件的根目录。
    
    参数:
      start (Path|str|None): 起始查找路径，默认为当前文件所在目录。
      
    返回:
      Path: 项目工程根目录绝对路径。
      
    异常:
      FileNotFoundError: 若遍历至磁盘根目录仍未找到标记文件则抛出。
    """
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
    """在一组候选相对路径中返回首个在文件系统中实际存在的路径。
    
    参数:
      root (Path|str): 基础目录路径。
      candidates (list[str]): 候选相对子路径列表。
      
    返回:
      Path: 首个实际存在的绝对路径。
      
    异常:
      FileNotFoundError: 若所有候选路径均不存在则抛出。
    """
    paths = [Path(root) / name for name in candidates]
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError('None of the resource locations exists: ' + ', '.join(map(str, paths)))


def original_dir(root):
    """获取交接的原始 C++ 代码工程根目录路径。
    
    参数:
      root (Path|str): 项目根目录路径。
      
    返回:
      Path: 原始代码目录路径。
    """
    return existing(root, ['A01_原始论文与原始代码/原始代码/Cilqr_cpp-CAV/Cilqr_cpp-CAV',
                           '05_原始交接代码/Cilqr_cpp-CAV/Cilqr_cpp-CAV'])


def baseline_dir(root):
    """获取原版算法三方法基准仿真数据完整集合目录路径。
    
    参数:
      root (Path|str): 项目根目录路径。
      
    返回:
      Path: 原版算法基准数据 full 目录路径。
    """
    return existing(root, ['A02_对照组代码/仿真数据/原版算法三方法基准_20261104/full',
                           '03_实验数据/原版算法三方法基准_20261104/full'])


def baseline_map(root, mapname):
    """获取指定测试地图 (B201, B207, B301) 的基准场景配置目录。
    
    参数:
      root (Path|str): 项目根目录路径。
      mapname (str): 地图名称 ('B201', 'B207', 'B301')。
      
    返回:
      Path: 对应场景配置目录路径。
    """
    current = baseline_dir(root) / mapname
    if (current / 'scene/scene.json').is_file():
        return current
    old = {'B201': '03_实验数据/原版基准核验_20261104/B201_v5_N80/B201',
           'B207': '03_实验数据/原版基准核验_20261104/B207_v6_N80/B207',
           'B301': '03_实验数据/原算法原地图复跑_20261103/B301'}
    return existing(root, [old[mapname]])


def replay_executable(root):
    """获取编译生成的原始重放二进制程序 (paper_replay.exe) 路径。
    
    参数:
      root (Path|str): 项目根目录路径。
      
    返回:
      Path: 可执行程序二进制路径。
    """
    return existing(root, ['A02_对照组代码/原论文场景复现/build/paper_replay.exe',
                           '02_对照组代码/原论文场景复现/build/paper_replay.exe'])


def results_dir(root):
    """获取仿真数据保存主目录路径。
    
    参数:
      root (Path|str): 项目根目录路径。
      
    返回:
      Path: 仿真数据输出主目录。
    """
    root = Path(root)
    return root / ('A02_对照组代码/仿真数据' if (root / 'A02_对照组代码').is_dir() else '03_实验数据')
