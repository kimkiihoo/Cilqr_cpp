#pragma once
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <stdexcept>

/**
 * @file replay_support.h
 * @brief 仿真测试场景冻结与逐周期优化结果数据序列化支持模块。
 * 
 * 本头文件提供统一的输入输出 (I/O) 接口，专用于在不同算法（如 C-iLQR, AL-iLQR）
 * 之间共享并固化同一套测试环境，杜绝随机几何采样带来的场景差异：
 * 1. 冻结并读取全局参考路径 (x, y, heading)；
 * 2. 冻结并读取动态障碍物初始状态 (x, y, theta, speed)；
 * 3. 序列化导出各周期滚动时域规划解（控制序列、状态轨迹、收敛标志、求解耗时）。
 * 
 * 物理量纲定义:
 *   - 位置坐标: x, y [m]
 *   - 航向角与铰接角: theta, gamma, heading [rad]
 *   - 速度: velocity, speed [m/s]
 *   - 铰接角速率: gamma_dot [rad/s]
 *   - 采样周期: dt [s]
 */

/**
 * @brief 从本地文本文件加载冻结的全局参考路径点集。
 * 
 * 数据格式: 第一行为点数 N，随后各行为 "x y heading"。
 * 
 * @param[out] plan 全局路径对象，成功加载后更新其内部点集。
 * @return bool 若文件存在且加载成功返回 true；若文件不存在返回 false。
 * @throws std::runtime_error 若文件已打开但格式异常或数据为空则抛出。
 */
inline bool replay_load_path(GlobalPlan& plan) {
    std::ifstream f("frozen_global_path.txt");
    if (!f) return false;
    size_t count; f >> count; std::vector<Point> points;
    for (size_t i = 0; i < count; ++i) {
        double x, y, h;
        f >> x >> y >> h;
        points.emplace_back(x, y, h);
    }
    if (!f || points.empty()) throw std::runtime_error("Invalid frozen path");
    plan.set_plan(points);
    return true;
}

/**
 * @brief 从本地文本文件加载冻结的动态障碍物初始状态列表。
 * 
 * 数据格式: 第一行为障碍物个数 M，随后各行为 "x y theta speed"。
 * 
 * @param[out] obs 障碍物初始状态向量 [x (m), y (m), theta (rad), speed (m/s)]。
 * @return bool 若文件存在且加载成功返回 true；若文件不存在返回 false。
 * @throws std::runtime_error 若文件读取发生解析错误则抛出。
 */
inline bool replay_load_obstacles(std::vector<State>& obs) {
    std::ifstream f("frozen_obstacles.txt");
    if (!f) return false;
    size_t count; f >> count;
    for (size_t i = 0; i < count; ++i) {
        State s;
        f >> s[0] >> s[1] >> s[2] >> s[3];
        obs.push_back(s);
    }
    if (!f) throw std::runtime_error("Invalid frozen obstacles");
    return true;
}

/**
 * @brief 将当前采样的全局参考路径与动态障碍物初态固化至磁盘文本文件。
 * 
 * @param[in] plan 全局参考路径对象。
 * @param[in] obs 动态障碍物初始状态列表。
 */
inline void replay_save_scene(const GlobalPlan& plan, const std::vector<State>& obs) {
    std::ofstream f("frozen_global_path.txt");
    f << std::setprecision(17);
    const auto points = plan.get_points();
    f << points.size() << '\n';
    for (const auto& p : points) {
        f << p.x << ' ' << p.y << ' ' << p.heading << '\n';
    }
    std::ofstream g("frozen_obstacles.txt");
    g << std::setprecision(17) << obs.size() << '\n';
    for (const auto& s : obs) {
        g << s[0] << ' ' << s[1] << ' ' << s[2] << ' ' << s[3] << '\n';
    }
}

/**
 * @brief 导出当前仿真运行的关键实际生效参数为 JSON 文件。
 * 
 * @param[in] a 算法优化参数结构体 (包含预测时域 N、步长 dt、期望车速等)。
 * @param[in] m 铰接车辆几何参数模型 (前后轴距 lf, lr, 宽度 width, 碰撞体长度 box_length)。
 * @param[in] c 运行工况配置 (仿真总周期数 ITER 等)。
 */
inline void replay_runtime(const Arg& a, const SystemModel& m, const RunConfig& c) {
    std::ofstream f("effective_runtime.json");
    f << std::setprecision(17);
    f << "{\"N\":" << a.N << ",\"dt\":" << a.dt << ",\"steps\":" << c.ITER
      << ",\"desire_speed\":" << a.desire_speed << ",\"gamma_dot_max\":" << a.gamma_dot_max
      << ",\"gamma_dot_min\":" << a.gamma_dot_min << ",\"lf\":" << m.lf << ",\"lr\":" << m.lr
      << ",\"width\":" << m.width << ",\"box_length\":" << m.box_length << "}";
}

/**
 * @brief 输出双精度浮点数，若遇到非有限值则格式化为标准 JSON 字符串 ("NaN", "+Infinity", "-Infinity")。
 * 
 * @param[in,out] f 输出流引用。
 * @param[in] v 双精度浮点数值。
 */
inline void replay_number(std::ostream& f, double v) {
    if (std::isfinite(v)) {
        f << v;
    } else if (std::isnan(v)) {
        f << "\"NaN\"";
    } else {
        f << (v > 0 ? "\"+Infinity\"" : "\"-Infinity\"");
    }
}

/**
 * @brief 将浮点数向量格式化输出为 JSON 数组字符串。
 * 
 * @param[in,out] f 输出流引用。
 * @param[in] values 浮点数向量。
 */
inline void replay_array(std::ostream& f, const std::vector<double>& values) {
    f << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) f << ',';
        replay_number(f, values[i]);
    }
    f << ']';
}

/**
 * @brief 序列化保存单步优化周期 (Frame) 的完整状态、控制输入与环境数据。
 * 
 * @param[in] ego_log 自车历史状态记录 (包含当前时刻执行后的状态)。
 * @param[in] obs 各动态障碍物预测轨迹列表。
 * @param[in] sol 优化求解器输出的最优解 (包含控制序列、规划轨迹、耗时、收敛状态)。
 * @param[in] plan 全局参考路径。
 * @param[in] model 铰接车几何运动学模型。
 * @param[in] arg 算法优化配置参数。
 * @param[in] solver 求解器类型标识 ('cilqr' 或 'alilqr')。
 * @param[in] map 地图工况标识 ('B201', 'B207', 'B301')。
 * @param[in] cycle 当前推进的控制周期步数索引 (从 0 开始)。
 */
inline void replay_frame(const std::vector<std::vector<double>>& ego_log,
                         const std::vector<Trajectory>& obs, const Solution& sol,
                         const GlobalPlan& plan, const SystemModel& model, const Arg& arg,
                         const std::string& solver, const std::string& map, int cycle) {
    const std::string dir = "outputs/" + map + "/" + solver + "/data";
    ensure_directory_exists(dir);
    std::ofstream f(dir + "/" + solver + "_data_" + std::to_string(cycle) + ".json");
    if (!f) throw std::runtime_error("Cannot save frame");
    f << std::setprecision(17) << "{\"cycle\":" << cycle << ",\"time_s\":" << (cycle + 1) * arg.dt
      << ",\"plan_time_s\":" << cycle * arg.dt << ",\"convergence_info\":{\"converged\":"
      << (sol.converged ? "true" : "false") << ",\"iterations\":" << sol.iterations << ",\"final_cost\":";
    replay_number(f, sol.final_cost);
    f << ",\"solve_time_ms\":";
    replay_number(f, sol.solve_time_ms);
    f << "},\"control_sequence\":{";
    for (int j = 0; j < 2; ++j) {
        if (j) f << ',';
        f << (j == 0 ? "\"velocity\":" : "\"gamma\":");
        std::vector<double> a;
        for (const auto& u : sol.control_sequence.controls) a.push_back(u[j]);
        replay_array(f, a);
    }
    f << "},\"global_plan\":{";
    const auto points = plan.get_points();
    for (int j = 0; j < 3; ++j) {
        if (j) f << ',';
        f << (j == 0 ? "\"x\":" : j == 1 ? "\"y\":" : "\"heading\":");
        std::vector<double> a;
        for (const auto& p : points) a.push_back(j == 0 ? p.x : j == 1 ? p.y : p.heading);
        replay_array(f, a);
    }
    f << "},\"planned_trajectory\":{";
    for (int j = 0; j < 4; ++j) {
        if (j) f << ',';
        f << (j == 0 ? "\"x\":" : j == 1 ? "\"y\":" : j == 2 ? "\"theta\":" : "\"gamma\":");
        std::vector<double> a;
        for (const auto& x : sol.ego_trj.states) a.push_back(x[j]);
        replay_array(f, a);
    }
    f << "},\"ego_vehicle\":{";
    for (int j = 0; j < 4; ++j) {
        if (j) f << ',';
        f << (j == 0 ? "\"x\":" : j == 1 ? "\"y\":" : j == 2 ? "\"theta\":" : "\"gamma\":");
        replay_number(f, ego_log[j].back());
    }
    f << "},\"obstacles\":[";
    for (size_t i = 0; i < obs.size(); ++i) {
        if (i) f << ',';
        const auto s = obs[i].states.front();
        f << "{\"x\":";
        replay_number(f, s[0]);
        f << ",\"y\":";
        replay_number(f, s[1]);
        f << ",\"theta\":";
        replay_number(f, s[2]);
        f << ",\"speed\":";
        replay_number(f, s[3]);
        f << ",\"length\":" << arg.obs_length << ",\"width\":" << arg.obs_width << '}';
    }
    f << "],\"vehicle_model\":{\"lf\":" << model.lf << ",\"lr\":" << model.lr << ",\"len\":" << model.len
      << ",\"width\":" << model.width << ",\"box_length\":" << model.box_length << "}}";
}
