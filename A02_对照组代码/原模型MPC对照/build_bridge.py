"""原生 C++ 代码桥接动态库 (original_model_bridge.dll) 的自动代码生成与 CMake 驱动编译脚本。"""
from pathlib import Path
import difflib
import hashlib
import json
import os
import shutil
import subprocess
HERE = Path(__file__).resolve().parent
from project_paths import project_root, original_dir
ROOT = project_root(HERE)
ORIGINAL = original_dir(ROOT)

def sha(path):
    """计算文件哈希值。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()

def function_text(source, signature):
    """提取特定函数定义的源码文本。"""
    start = source.index(signature)
    opening = source.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]

def generate():
    """从原生源文件中提取并生成桥接头文件与代码片段。"""
    dest = HERE / 'generated'
    (dest / 'include').mkdir(parents=True, exist_ok=True)
    records, differences = ([], [])
    for name in ['ilqr.h', 'utils.h', 'common_types.h', 'config_loader.h', 'hybrid_astar.h']:
        source = ORIGINAL / 'include' / name
        target = dest / 'include' / name
        before = source.read_text(encoding='utf-8')
        after = before
        if name == 'ilqr.h':
            after = after.replace('inline size_t find_closest_point(', 'size_t find_closest_point(')
            after = after.replace('inline double distance(', 'double distance(')
            for cls in ['CILQRSolver', 'ALILQRSolver']:
                needle = 'class ' + cls + '{'
                assert after.count(needle) == 1
                after = after.replace(needle, needle + '\n    friend struct OriginalBridgeAccess;')
        target.write_text(after, encoding='utf-8')
        records.append({'original': str(source.relative_to(ORIGINAL)), 'generated': str(target.relative_to(HERE)), 'original_sha256': sha(source), 'generated_sha256': sha(target), 'text_changed': before != after})
        if before != after:
            differences.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='original/include/' + name, tofile='generated/include/' + name))
    for name in ['ilqr.cpp', 'config_loader.cpp']:
        source, target = (ORIGINAL / 'src' / name, dest / name)
        shutil.copy2(source, target)
        assert sha(source) == sha(target)
        records.append({'original': 'src/' + name, 'generated': str(target.relative_to(HERE)), 'original_sha256': sha(source), 'generated_sha256': sha(target), 'text_changed': False})
    utils = (ORIGINAL / 'src/utils.cpp').read_text(encoding='utf-8')
    main = (ORIGINAL / 'src/main.cpp').read_text(encoding='utf-8')
    functions = [function_text(utils, 'ArticulatedLimits compute_articulated_limits('), function_text(utils, 'Trajectory predict_obstacle_trajectory('), function_text(main, 'void finalize_parameters(')]
    support = '// Original functions below are extracted verbatim; only log-path I/O is replaced.\n#include "ilqr.h"\n#include "utils.h"\n#include <filesystem>\n#include <cmath>\n#include <string>\nstd::string original_bridge_log_root;\nstd::string resolve_resource_path(const std::string& path) { return original_bridge_log_root + "/" + path; }\nvoid ensure_directory_exists(const std::string& path) { std::filesystem::create_directories(path); }\n'
    (dest / 'support.cpp').write_text(support + '\n\n'.join(functions) + '\n', encoding='utf-8')
    optimizer = (ORIGINAL / 'src/ilqr.cpp').read_text(encoding='utf-8')
    r_solve = function_text(optimizer, 'Solution CILQRSolver::solve(')
    preamble = r_solve[r_solve.index('{') + 1:r_solve.index('    Solution nominal_solution =')]
    preamble = preamble.replace('this->', 'r.').replace('ego.', 'r.ego.').replace('arg.', 'r.arg.')
    preamble = preamble.replace('r.r.', 'r.')
    (dest / 'r_prepare.inc').write_text(preamble, encoding='utf-8')
    (HERE / 'adaptation.diff').write_text(''.join(differences), encoding='utf-8')
    (HERE / 'source_manifest.json').write_text(json.dumps({'files': records, 'extracted_functions': ['utils.cpp:compute_articulated_limits', 'utils.cpp:predict_obstacle_trajectory', 'main.cpp:finalize_parameters', 'ilqr.cpp:CILQRSolver::solve preamble'], 'original_optimizer_sha256': sha(ORIGINAL / 'src/ilqr.cpp'), 'scope': 'Header friend access and helper linkage only; optimizer/model and config loader .cpp byte-identical. Two log path functions are standalone I/O shims; no planner geometry from any revision backend.'}, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    """桥接动态库编译生成主函数。"""
    generate()
    compiler = Path('C:/msys64/mingw64/bin/g++.exe')
    cmake = compiler.with_name('cmake.exe')
    ninja = compiler.with_name('ninja.exe')
    env = os.environ.copy()
    env['PATH'] = str(compiler.parent) + os.pathsep + env.get('PATH', '')
    build = HERE / 'build'
    build.mkdir(exist_ok=True)
    commands = [('configure', [str(cmake), '--fresh', '-S', str(HERE), '-B', str(build), '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_COMPILER=' + str(compiler), '-DCMAKE_MAKE_PROGRAM=' + str(ninja)]), ('build', [str(cmake), '--build', str(build), '--parallel', '2'])]
    for label, cmd in commands:
        with (build / (label + '.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run(cmd, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Build failed: ' + str(build / (label + '.log')))
    for name in ['libgcc_s_seh-1.dll', 'libstdc++-6.dll', 'libwinpthread-1.dll']:
        shutil.copy2(compiler.with_name(name), build / name)
    print(build / 'original_model_bridge.dll')
if __name__ == '__main__':
    main()