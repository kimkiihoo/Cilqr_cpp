#include "config_loader.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>

/**
 * @file config_loader.cpp
 * @brief 仿真实验与优化求解器 JSON 配置文件解析器实现。
 * 
 * 采用轻量化原生字符串解析方式，依次读取并解析三份主配置文件:
 * 1. config/main.json: 工况地图选择、起终点坐标、仿真步长与铰接车物理尺寸参数;
 * 2. config/ilqr.json: iLQR/C-iLQR 预测时域、收敛判据、Pure Pursuit 预瞄参数、Q/R 权重矩阵与物理约束边界;
 * 3. config/hybrid_astar.json: 混合 A* 全局路径规划网格分辨率、转向半径与启发代价惩罚系数。
 */
namespace {

/**
 * @brief 剔除 JSON 源码中字符串引号外部的所有空格、换行符与制表符。
 * @param[in] json 原始未处理的 JSON 字符串。
 * @return std::string 紧凑压缩后的 JSON 字符串。
 */
std::string remove_comments_and_whitespace(const std::string& json) {
    std::string result;
    bool in_string = false;
    for (size_t i = 0; i < json.length(); ++i) {
        char c = json[i];
        if (c == '"') in_string = !in_string;
        if (!in_string && (c == ' ' || c == '\n' || c == '\r' || c == '\t')) continue;
        result += c;
    }
    return result;
}

/**
 * @brief 在压缩后的 JSON 字符串中提取指定字典对象键对应的大括号内部子字符串。
 * @param[in] json 紧凑 JSON 字符串。
 * @param[in] key 目标字典节点的键名。
 * @return std::string 大括号内部的子串，若未找到则返回空字符串。
 */
std::string get_section(const std::string& json, const std::string& key) {
    std::string search_key = "\"" + key + "\":{";
    size_t start_pos = json.find(search_key);
    if (start_pos == std::string::npos) return "";
    
    start_pos += search_key.length();
    int brace_count = 1;
    size_t end_pos = start_pos;
    while (end_pos < json.length() && brace_count > 0) {
        if (json[end_pos] == '{') brace_count++;
        else if (json[end_pos] == '}') brace_count--;
        end_pos++;
    }
    
    if (brace_count == 0) {
        return json.substr(start_pos, end_pos - start_pos - 1);
    }
    return "";
}

/**
 * @brief 在紧凑 JSON 片段中提取键对应的原始字面值字符串。
 * @param[in] json 紧凑 JSON 字符串片段。
 * @param[in] key 目标字段键名。
 * @return std::string 字段值的字符串形式，若不存在返回空字符串。
 */
std::string get_value_str(const std::string& json, const std::string& key) {
    std::string search_key = "\"" + key + "\":";
    size_t start_pos = json.find(search_key);
    if (start_pos == std::string::npos) return "";
    
    start_pos += search_key.length();
    size_t end_pos = start_pos;
    while (end_pos < json.length() && json[end_pos] != ',' && json[end_pos] != '}') {
        end_pos++;
    }
    return json.substr(start_pos, end_pos - start_pos);
}

/**
 * @brief 解析双精度浮点型字段值，若解析失败或字段不存在则回退至默认值。
 * @param[in] json 紧凑 JSON 字符串片段。
 * @param[in] key 目标字段键名。
 * @param[in] default_val 默认缺省值。
 * @return double 解析得到的浮点数值。
 */
double get_double(const std::string& json, const std::string& key, double default_val) {
    std::string val_str = get_value_str(json, key);
    if (val_str.empty()) return default_val;
    try {
        return std::stod(val_str);
    } catch (...) {
        return default_val;
    }
}

/**
 * @brief 解析整型字段值，若解析失败或字段不存在则回退至默认值。
 * @param[in] json 紧凑 JSON 字符串片段。
 * @param[in] key 目标字段键名。
 * @param[in] default_val 默认缺省值。
 * @return int 解析得到的整数值。
 */
int get_int(const std::string& json, const std::string& key, int default_val) {
    std::string val_str = get_value_str(json, key);
    if (val_str.empty()) return default_val;
    try {
        return std::stoi(val_str);
    } catch (...) {
        return default_val;
    }
}

/**
 * @brief 解析布尔型字段值 ("true" / "false")，缺省返回默认值。
 * @param[in] json 紧凑 JSON 字符串片段。
 * @param[in] key 目标字段键名。
 * @param[in] default_val 默认缺省值。
 * @return bool 解析得到的布尔值。
 */
bool get_bool(const std::string& json, const std::string& key, bool default_val) {
    std::string val_str = get_value_str(json, key);
    if (val_str.empty()) return default_val;
    return (val_str == "true");
}

/**
 * @brief 解析浮点型一维数组字段 (形如 "[1.0, 2.0, 3.0]")。
 * @param[in] json 紧凑 JSON 字符串片段。
 * @param[in] key 目标字段键名。
 * @return std::vector<double> 解析出的浮点数列表。
 */
std::vector<double> get_vector(const std::string& json, const std::string& key) {
    std::vector<double> result;
    std::string search_key = "\"" + key + "\":[";
    size_t start_pos = json.find(search_key);
    if (start_pos == std::string::npos) return result;
    
    start_pos += search_key.length();
    size_t end_pos = json.find("]", start_pos);
    if (end_pos == std::string::npos) return result;
    
    std::string content = json.substr(start_pos, end_pos - start_pos);
    std::stringstream ss(content);
    std::string item;
    while (std::getline(ss, item, ',')) {
        try {
            result.push_back(std::stod(item));
        } catch (...) {}
    }
    return result;
}

} // namespace

/**
 * @brief 从磁盘读取指定文件的全部文本内容。
 * @param[in] filename 目标文件绝对或相对路径。
 * @return std::string 文件完整文本，若无法打开则返回空字符串。
 */
std::string load_file_content(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open config file " << filename << ". Using default parameters." << std::endl;
        return "";
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

/**
 * @brief 加载并解析三份主配置文件，将参数填充至各算法与仿真结构体。
 * 
 * 物理量纲与参数配置说明:
 *   - run_config:
 *       - start_x, start_y [m], start_theta [rad]: 车辆初始位姿
 *       - goal_x, goal_y [m], goal_theta [rad]: 全局规划目标位姿
 *       - ITER: 仿真最大推进周期数 (步长 dt = 0.1 s)
 *       - obstacle_speed [m/s], obstacle_count, obstacle_distance [m]: 动态障碍物分布
 *       - solver_type: 优化器算法类型 ('cilqr', 'alilqr', 'ilqr')
 *   - arg:
 *       - N: 滚动优化预测时域步数 (通常 50 步或 80 步)
 *       - dt: 离散采样周期 [s] (通常 0.1 s)
 *       - desire_speed [m/s]: 期望巡航速度
 *       - desire_heading [rad]: 期望航向偏差目标 (0 rad)
 *       - Q: 状态偏差权重对角矩阵 (x, y, theta, gamma)
 *       - R: 控制增量权重对角矩阵 (velocity, gamma_dot)
 *       - gamma_max, gamma_min [rad]: 铰接角机械硬约束
 *       - gamma_dot_max, gamma_dot_min [rad/s]: 铰接角速度动态约束
 *       - acc_max, acc_min [m/s^2]: 纵向加速度极限约束
 *       - obs_q1, obs_q2, lane_q1, lane_q2: 障碍物与道路走廊对数障碍/增广罚函数权重
 *   - system_model:
 *       - lf [m], lr [m]: 铰接车前后车体轴距
 *       - width [m]: 车体宽度
 *       - box_length [m]: 碰撞检测包围盒长度
 *   - ha_params:
 *       - grid_resolution [m]: 栅格地图离散分辨率
 *       - turning_radius [m]: 最小转弯半径
 *       - heuristic_weight: 启发式引导权重系数
 * 
 * @param[in] main_config_file 主运行与工况配置文件路径 (config/main.json)。
 * @param[in] ilqr_config_file iLQR 优化算法配置文件路径 (config/ilqr.json)。
 * @param[in] ha_config_file 混合 A* 算法配置文件路径 (config/hybrid_astar.json)。
 * @param[out] arg 算法参数与权重配置结构体。
 * @param[out] system_model 铰接车运动学与尺寸模型。
 * @param[out] ha_params 混合 A* 搜索参数结构体。
 * @param[out] run_config 仿真工况配置结构体。
 */
void load_config(const std::string& main_config_file, 
                 const std::string& ilqr_config_file,
                 const std::string& ha_config_file,
                 Arg& arg, 
                 SystemModel& system_model, 
                 HybridAStarParams& ha_params,
                 RunConfig& run_config) {
    
    // 1. 解析主工况与车辆参数配置文件 (main.json)
    std::string main_json = remove_comments_and_whitespace(load_file_content(main_config_file));
    if (!main_json.empty()) {
        // 读取选定的地图工况名称
        std::string sm = get_value_str(main_json, "selected_map");
        if (!sm.empty()) {
             if (sm.front() == '"') sm = sm.substr(1, sm.length() - 2);
             run_config.selected_map = sm;
        }

        // 读取对应地图的场景配置 (起终点位姿、最大步数、障碍物参数)
        std::string rcs = get_section(main_json, "run_configs");
        if (!rcs.empty()) {
            // 查找选定地图配置，若未指定则回退至 default 缺省节
            std::string rc = get_section(rcs, run_config.selected_map);
            if (rc.empty()) {
                std::cerr << "Warning: Config for map " << run_config.selected_map << " not found. Using default." << std::endl;
                rc = get_section(rcs, "default");
            }

            if (!rc.empty()) {
                run_config.start_x = get_double(rc, "start_x", run_config.start_x);
                run_config.start_y = get_double(rc, "start_y", run_config.start_y);
                run_config.start_theta = get_double(rc, "start_theta", run_config.start_theta);
                run_config.goal_x = get_double(rc, "goal_x", run_config.goal_x);
                run_config.goal_y = get_double(rc, "goal_y", run_config.goal_y);
                run_config.goal_theta = get_double(rc, "goal_theta", run_config.goal_theta);
                run_config.ITER = get_double(rc, "ITER", run_config.ITER);
                run_config.obstacle_speed = get_double(rc, "obstacle_speed", run_config.obstacle_speed);
                run_config.obstacle_count = get_int(rc, "obstacle_count", run_config.obstacle_count);
                run_config.obstacle_distance = get_double(rc, "obstacle_distance", run_config.obstacle_distance);
                
                std::string st = get_value_str(rc, "solver_type");
                if (!st.empty()) {
                     if (st.front() == '"') st = st.substr(1, st.length() - 2);
                     run_config.solver_type = st;
                }
            }
        }

        // 读取仿真离散步长与总时长 (dt [s], tf [s])
        std::string sim = get_section(main_json, "simulation");
        if (!sim.empty()) {
            arg.tf = get_double(sim, "tf", arg.tf);
            double dt = get_double(sim, "dt", arg.dt);
            arg.dt = dt;
            system_model.dt = dt;
        }

        // 读取铰接式车辆几何外形尺寸 (轴距 lf, lr [m], 宽度 width [m], 包围盒 box_length [m])
        std::string v = get_section(main_json, "vehicle");
        if (!v.empty()) {
            system_model.ego_rad = get_double(v, "ego_rad", system_model.ego_rad);
            system_model.lf = get_double(v, "lf", system_model.lf);
            system_model.lr = get_double(v, "lr", system_model.lr);
            system_model.width = get_double(v, "width", system_model.width);
            system_model.box_length = get_double(v, "box_length", system_model.box_length);
        }
    }

    // 2. 解析优化求解器参数配置文件 (ilqr.json)
    std::string ilqr_json = remove_comments_and_whitespace(load_file_content(ilqr_config_file));
    if (!ilqr_json.empty()) {
        // 读取 C-iLQR 算法迭代控制参数 (预测时域 N, 收敛容差 tol, 最大迭代轮次 max_iter)
        std::string cilqr = get_section(ilqr_json, "cilqr");
        if (!cilqr.empty()) {
            int N = get_int(cilqr, "N", arg.N);
            arg.N = N;
            system_model.N = N;
            arg.tol = get_double(cilqr, "tol", arg.tol);
            arg.rel_tol = get_double(cilqr, "rel_tol", arg.rel_tol);
            arg.max_iter = get_int(cilqr, "max_iter", arg.max_iter);
            arg.lamb_init = get_double(cilqr, "lamb_init", arg.lamb_init);
            arg.lamb_factor = get_double(cilqr, "lamb_factor", arg.lamb_factor);
            arg.lamb_max = get_double(cilqr, "lamb_max", arg.lamb_max);
        }

        // 读取 Pure Pursuit 初始前向名义解生成参数
        std::string pp = get_section(ilqr_json, "pure_pursuit");
        if (!pp.empty()) {
            arg.kv = get_double(pp, "kv", arg.kv);
            arg.kp = get_double(pp, "kp", arg.kp);
            arg.ld0 = get_double(pp, "ld0", arg.ld0);
            arg.ld_min = get_double(pp, "ld_min", arg.ld_min);
            arg.ld_max = get_double(pp, "ld_max", arg.ld_max);
        }

        // 读取目标代价函数权重系数与期望参考状态 (desire_speed [m/s], Q, R)
        std::string cw = get_section(ilqr_json, "cost_weights");
        if (!cw.empty()) {
            arg.desire_speed = get_double(cw, "desire_speed", arg.desire_speed);
            arg.desire_heading = get_double(cw, "desire_heading", arg.desire_heading);
            arg.v_rate_weight = get_double(cw, "v_rate_weight", arg.v_rate_weight);
            arg.ref_weight = get_double(cw, "ref_weight", arg.ref_weight);
            
            std::vector<double> Q_vec = get_vector(cw, "Q");
            if (Q_vec.size() == 4) {
                arg.Q.setZero();
                arg.Q(0,0) = Q_vec[0];
                arg.Q(1,1) = Q_vec[1];
                arg.Q(2,2) = Q_vec[2];
                arg.Q(3,3) = Q_vec[3];
            }
            std::vector<double> R_vec = get_vector(cw, "R");
            if (R_vec.size() == 2) {
                arg.R.setZero();
                arg.R(0,0) = R_vec[0];
                arg.R(1,1) = R_vec[1];
            }
        }

        // 读取物理与安全约束参数 (铰接角、角速率、纵向加速度、走廊限宽、避障椭圆安全裕度)
        std::string c = get_section(ilqr_json, "constraints");
        if (!c.empty()) {
            arg.if_cal_obs_cost = get_bool(c, "if_cal_obs_cost", arg.if_cal_obs_cost);
            arg.if_cal_lane_cost = get_bool(c, "if_cal_lane_cost", arg.if_cal_lane_cost);
            arg.if_cal_speed_rate_cost = get_bool(c, "if_cal_speed_rate_cost", arg.if_cal_speed_rate_cost);
            
            arg.if_cal_gamma_barrier = get_bool(c, "if_cal_gamma_barrier", arg.if_cal_gamma_barrier);
            arg.gamma_max = get_double(c, "gamma_max", arg.gamma_max);
            arg.gamma_max_q1 = get_double(c, "gamma_max_q1", arg.gamma_max_q1);
            arg.gamma_max_q2 = get_double(c, "gamma_max_q2", arg.gamma_max_q2);
            arg.gamma_min = get_double(c, "gamma_min", arg.gamma_min);
            arg.gamma_min_q1 = get_double(c, "gamma_min_q1", arg.gamma_min_q1);
            arg.gamma_min_q2 = get_double(c, "gamma_min_q2", arg.gamma_min_q2);
            
            arg.if_cal_gamma_dot_barrier = get_bool(c, "if_cal_gamma_dot_barrier", arg.if_cal_gamma_dot_barrier);
            arg.gamma_dot_max = get_double(c, "gamma_dot_max", arg.gamma_dot_max);
            arg.gamma_dot_max_q1 = get_double(c, "gamma_dot_max_q1", arg.gamma_dot_max_q1);
            arg.gamma_dot_max_q2 = get_double(c, "gamma_dot_max_q2", arg.gamma_dot_max_q2);
            arg.gamma_dot_min = get_double(c, "gamma_dot_min", arg.gamma_dot_min);
            arg.gamma_dot_min_q1 = get_double(c, "gamma_dot_min_q1", arg.gamma_dot_min_q1);
            arg.gamma_dot_min_q2 = get_double(c, "gamma_dot_min_q2", arg.gamma_dot_min_q2);
            
            arg.acc_max = get_double(c, "acc_max", arg.acc_max);
            arg.acc_min = get_double(c, "acc_min", arg.acc_min);
            arg.acc_q1 = get_double(c, "acc_q1", arg.acc_q1);
            arg.acc_q2 = get_double(c, "acc_q2", arg.acc_q2);
            
            arg.trace_safe_width_left = get_double(c, "trace_safe_width_left", arg.trace_safe_width_left);
            arg.trace_safe_width_right = get_double(c, "trace_safe_width_right", arg.trace_safe_width_right);
            arg.lane_q1 = get_double(c, "lane_q1", arg.lane_q1);
            arg.lane_q2 = get_double(c, "lane_q2", arg.lane_q2);
            
            arg.obs_q1 = get_double(c, "obs_q1", arg.obs_q1);
            arg.obs_q2 = get_double(c, "obs_q2", arg.obs_q2);
            arg.obs_length = get_double(c, "obs_length", arg.obs_length);
            arg.obs_width = get_double(c, "obs_width", arg.obs_width);
            arg.safe_a_buffer = get_double(c, "safe_a_buffer", arg.safe_a_buffer);
            arg.safe_b_buffer = get_double(c, "safe_b_buffer", arg.safe_b_buffer);
        }
    }

    // 3. 解析混合 A* 算法全局搜索参数配置文件 (hybrid_astar.json)
    std::string ha_json = remove_comments_and_whitespace(load_file_content(ha_config_file));
    if (!ha_json.empty()) {
        std::string ha = get_section(ha_json, "hybrid_astar");
        if (!ha.empty()) {
            ha_params.grid_resolution = get_double(ha, "grid_resolution", ha_params.grid_resolution);
            ha_params.heading_resolution = get_double(ha, "heading_resolution", ha_params.heading_resolution);
            ha_params.turning_radius = get_double(ha, "turning_radius", ha_params.turning_radius);
            ha_params.move_step = get_double(ha, "move_step", ha_params.move_step);
            ha_params.move_step_backwards = get_double(ha, "move_step_backwards", ha_params.move_step_backwards);
            ha_params.allow_reverse = get_bool(ha, "allow_reverse", ha_params.allow_reverse);
            ha_params.num_gamma_angles = get_int(ha, "num_gamma_angles", ha_params.num_gamma_angles);
            ha_params.steering_penalty = get_double(ha, "steering_penalty", ha_params.steering_penalty);
            ha_params.direction_change_penalty = get_double(ha, "direction_change_penalty", ha_params.direction_change_penalty);
            ha_params.backwards_penalty = get_double(ha, "backwards_penalty", ha_params.backwards_penalty);
            ha_params.heuristic_weight = get_double(ha, "heuristic_weight", ha_params.heuristic_weight);
            ha_params.max_iterations = get_int(ha, "max_iterations", ha_params.max_iterations);
            ha_params.num_nodes_to_keep = get_int(ha, "num_nodes_to_keep", ha_params.num_nodes_to_keep);
            ha_params.inflation_radius = get_double(ha, "inflation_radius", ha_params.inflation_radius);
            ha_params.goal_tolerance_xy = get_double(ha, "goal_tolerance_xy", ha_params.goal_tolerance_xy);
            ha_params.goal_tolerance_heading = get_double(ha, "goal_tolerance_heading", ha_params.goal_tolerance_heading);
            ha_params.clearance_weight = get_double(ha, "clearance_weight", ha_params.clearance_weight);
            ha_params.desired_clearance = get_double(ha, "desired_clearance", ha_params.desired_clearance);
            ha_params.max_clearance = get_double(ha, "max_clearance", ha_params.max_clearance);
            ha_params.min_rs_clearance = get_double(ha, "min_rs_clearance", ha_params.min_rs_clearance);
        }
    }
    
    std::cout << "Configuration loaded from config/main.json, config/ilqr.json, config/hybrid_astar.json" << std::endl;
}
