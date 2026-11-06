"""Locate retained experiment resources before and after the A01--A06 move.

Only filesystem locations are resolved here; model and experiment settings are
read unchanged from the selected frozen input files.
"""
from pathlib import Path


def project_root(start=None):
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
    paths = [Path(root) / name for name in candidates]
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError('None of the resource locations exists: ' + ', '.join(map(str, paths)))


def original_dir(root):
    return existing(root, ['A01_原始论文与原始代码/原始代码/Cilqr_cpp-CAV/Cilqr_cpp-CAV',
                           '05_原始交接代码/Cilqr_cpp-CAV/Cilqr_cpp-CAV'])


def baseline_dir(root):
    return existing(root, ['A02_对照组代码/仿真数据/原版算法三方法基准_20261104/full',
                           '03_实验数据/原版算法三方法基准_20261104/full'])


def baseline_map(root, mapname):
    current = baseline_dir(root) / mapname
    if (current / 'scene/scene.json').is_file():
        return current
    old = {'B201': '03_实验数据/原版基准核验_20261104/B201_v5_N80/B201',
           'B207': '03_实验数据/原版基准核验_20261104/B207_v6_N80/B207',
           'B301': '03_实验数据/原算法原地图复跑_20261103/B301'}
    return existing(root, [old[mapname]])


def replay_executable(root):
    return existing(root, ['A02_对照组代码/原论文场景复现/build/paper_replay.exe',
                           '02_对照组代码/原论文场景复现/build/paper_replay.exe'])


def results_dir(root):
    root = Path(root)
    return root / ('A02_对照组代码/仿真数据' if (root / 'A02_对照组代码').is_dir() else '03_实验数据')
