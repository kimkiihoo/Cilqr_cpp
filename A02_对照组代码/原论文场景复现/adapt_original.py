"""原始交接代码透传适配与复现环境构建脚本。

主要功能:
  将交接的原始 C++ 代码工程 (A01_原始论文与原始代码) 适配为支持无图形界面重放与确定性比对的版本:
  1. 移除运行时 matplotlib/OpenCV 调试窗口依赖，改为离线 JSON 数据序列化;
  2. 固定随机数发生器种子 (20261030u)，消除 RRT* 与局部几何捷径采样的随机波动;
  3. 注入 replay_support.h 运行时场景冻结与各周期优化结果 (控制序列、状态轨迹、耗时、收敛状态) 导出接口;
  4. 规范 cross-translation-unit 函数导出符号，生成修改差异清单 (adaptation.diff) 与 SHA-256 校验和。

注意:
  算法优化器内核 (CILQRSolver, ALILQRSolver)、运动学模型与约束评估逻辑保持完全不变。
"""
from pathlib import Path
import difflib, hashlib, json, shutil
from project_paths import project_root, original_dir


def sha(p):
    """计算指定文件的 SHA-256 十六进制摘要值。
    
    参数:
      p (Path): 待校验文件的 Path 路径对象。
      
    返回:
      str: 64 位十六进制 SHA-256 校验和字符串。
    """
    return hashlib.sha256(p.read_bytes()).hexdigest()


def replace_function(s, signature, body):
    """在 C++ 源码字符串中精确替换指定函数签名的实现函数体。
    
    参数:
      s (str): 原始 C++ 源文件字符串。
      signature (str): 目标函数的声明特征签名 (如 'void draw_bitmap_debug(')。
      body (str): 替换后的新函数体代码块。
      
    返回:
      str: 替换完成后的 C++ 源文件字符串。
    """
    start = s.index(signature)
    opening = s.index('{', start)
    depth = 1
    i = opening + 1
    while depth:
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
        i += 1
    return s[:opening] + '{\n' + body + '\n}' + s[i:]


def adapt(root, dest):
    """执行原始交接代码的构建适配，生成包含重放钩子的 C++ 工程。
    
    参数:
      root (Path): 项目根目录路径对象。
      dest (Path): 目标构建源码输出目录 (如 source/)。
      
    返回:
      Path: 原始交接代码源目录路径对象。
    """
    original = original_dir(root)
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    diffs = []
    for folder in ['src', 'include']:
        (dest / folder).mkdir(exist_ok=True)
        for p in sorted((original / folder).iterdir()):
            if not p.is_file():
                continue
            target = dest / folder / p.name
            shutil.copy2(p, target)
            before = p.read_text(encoding='utf-8')
            after = before
            if p.name == 'utils.cpp':
                # 剔除实时 matplotlib C++ 绘图调用，避免批处理或无显示环境下程序挂起
                after = after.replace('#include "matplotlibcpp.h"', '// matplotlib C++ debug renderer omitted; plotting is performed offline.')
                after = replace_function(after, 'void draw_bitmap_debug(', '    (void)map_data; (void)output_path; // Debug-only renderer disabled.')
                after = replace_function(after, 'std::string resolve_resource_path(', '    return relative_path; // Process CWD is the isolated run directory.')
                # 资源定位统一采用基于运行目录的相对路径
                after = after.replace('GetModuleFileNameA(NULL, buffer, MAX_PATH);', 'buffer[0] = 0; // Use relative ASCII paths inside the current run.')
                # 固定随机种子 (20261030u)，保证全局几何路径与回退采样具有严格可复现性
                after = after.replace('std::random_device rd;\n            std::mt19937 rng(rd());', 'std::mt19937 rng(20261030u); // Replay seed; only global-path shortcut sampling.')
                after = after.replace('unsigned int seed = std::random_device{}();', 'unsigned int seed = 20261030u; // Replay seed; only RRT* fallback.')
            if p.name == 'main.cpp':
                # 注入重放支持头文件与场景/帧数据序列化接口
                after = after.replace('#include <ctime>', '#include <ctime>\n#include "replay_support.h"')
                after = after.replace('if (!plan_global_path(bitmap_map, start, goal, global_plan, ha_params, enable_planning)) {', 'if (!replay_load_path(global_plan) && !plan_global_path(bitmap_map, start, goal, global_plan, ha_params, enable_planning)) {')
                after = after.replace('std::vector<State> obs_initial_states = generate_obstacles(global_plan, grid, run_config.obstacle_count, run_config.obstacle_distance, run_config.obstacle_speed);', 'std::vector<State> obs_initial_states;\n    if (!replay_load_obstacles(obs_initial_states)) obs_initial_states = generate_obstacles(global_plan, grid, run_config.obstacle_count, run_config.obstacle_distance, run_config.obstacle_speed);')
                after = after.replace('parse_arguments(argc, argv, run_config);', 'parse_arguments(argc, argv, run_config);\n    replay_runtime(arg, system_model, run_config);')
                after = after.replace('ObstacleData obs_data = initialize_obstacles(global_plan, bitmap_map, arg, run_config);', 'ObstacleData obs_data = initialize_obstacles(global_plan, bitmap_map, arg, run_config);\n    replay_save_scene(global_plan, obs_data.current_states);\n    if (argc > 1 && std::string(argv[1]) == "--prepare-scene") return 0;')
                after = after.replace('cur_ctrl = solution.control_sequence.controls[0];', 'if (solution.control_sequence.controls.empty()) { std::cerr << "REPLAY_FAILURE empty_control_sequence at cycle " << i << std::endl; return 2; }\n        cur_ctrl = solution.control_sequence.controls[0];')
                after = after.replace('dynamic_plot(global_plan_log, ego_log, obs_data.trajectories, solution, &bitmap_map, global_plan, ego.get_model(), arg, run_config.solver_type, run_config.selected_map);', 'replay_frame(ego_log, obs_data.trajectories, solution, global_plan, ego.get_model(), arg, run_config.solver_type, run_config.selected_map, i);')
            if p.name in ['ilqr.h', 'ilqr.cpp']:
                # 原工程在头文件声明 inline 但在 .cpp 中统一定义，修正为标准链接导出
                after = after.replace('inline size_t find_closest_point(', 'size_t find_closest_point(')
            if before != after:
                target.write_text(after, encoding='utf-8')
            records.append(dict(path=str(p.relative_to(original)).replace('\\', '/'), original_sha256=sha(p), adapted_sha256=sha(target), changed=before != after))
            if before != after:
                diffs.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='original/' + folder + '/' + p.name, tofile='adapted/' + folder + '/' + p.name))
    shutil.copy2(Path(__file__).with_name('replay_support.h'), dest / 'include/replay_support.h')
    (dest / 'adaptation.diff').write_text(''.join(diffs), encoding='utf-8')
    (dest / 'source_hashes.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    return original


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    adapt(project_root(here), here / 'source')
