#include <iostream>
#include "ilqr.h"
#include "utils.h"
#include "config_loader.h"
#include <ctime>
#include "replay_support.h"

/**
 * @file main.cpp
 * @brief 铰接式矿用车辆局部避障规划重放仿真主程序。
 * 
 * 仿真主流程:
 * 1. 加载工况与算法配置 (main.json, ilqr.json, hybrid_astar.json);
 * 2. 构建地图栅格与障碍物空间，生成/装载全局几何参考路径;
 * 3. 初始自车状态 [x, y, theta, gamma] 与动态障碍物外推轨迹;
 * 4. 滚动时域规划推进 (Receding Horizon Planning):
 *    - 调用 C-iLQR 或 AL-iLQR 求解最优控制序列 u = [v, gamma_dot]^T;
 *    - 评估最大约束违例度 CMax;
 *    - 应用首步控制输入，通过非线性运动学方程推进一步;
 *    - 序列化导出每一帧 (Frame) 的完整状态与求解结果。
 * 
 * 物理量纲与状态定义:
 *   - 自车状态: x, y [m], theta [rad], gamma [rad]
 *   - 控制输入: velocity [m/s], gamma_dot [rad/s]
 *   - 仿真步长: dt = 0.1 s
 */

#ifdef _WIN32
#include <windows.h>
#endif

/**
 * @brief 动态障碍物集合容器结构体。
 */
struct ObstacleData {
    std::vector<State> current_states;
    std::vector<Trajectory> trajectories;
};

/**
 * @brief 解析命令行参数，覆写优化器求解类型配置 (--solver cilqr / alilqr)。
 * @param[in] argc 参数计数。
 * @param[in] argv 参数字符串数组。
 * @param[out] config 运行时配置结构体。
 */
void parse_arguments(int argc, char** argv, RunConfig& config) {
    for(int ai=1; ai<argc; ++ai){
        std::string a = argv[ai];
        if(a == "--solver" && ai+1 < argc){
            config.solver_type = argv[ai+1];
            ++ai;
        }
    }
}

/**
 * @brief 从 JSON 资源文件加载高程与栅格地图数据。
 * @param[in] selected_map 地图标识名称 ('B201', 'B207', 'B301')。
 * @param[in] solver_type 求解器类型标识。
 * @return MapData 包含网格尺寸、分辨率与占用栅格的地图数据结构。
 */
MapData load_and_process_map(const std::string& selected_map, const std::string& solver_type) {
    std::string map_file = resolve_resource_path("Maps/bitmap/" + selected_map + "_global_map.json");
    std::cout << "Loading map file: " << map_file << " (Selected: " << selected_map << ")" << std::endl;
    
    MapData bitmap_map = load_bitmap_map(map_file);
    std::cout << "Bitmap map loaded: " << bitmap_map.width << " x " << bitmap_map.height << std::endl;
    std::cout << "Map data size: " << bitmap_map.data.size() << std::endl;
    std::cout << "Map resolution: " << bitmap_map.resolution << std::endl;
    std::cout << "Map max_elevation: " << bitmap_map.max_elevation << std::endl;
    if (!bitmap_map.data.empty()) {
        std::cout << "First row size: " << bitmap_map.data[0].size() << std::endl;
    }
    
    // 地图栅格加载完成，离线批处理模式下省略冗余的中间地图重存步骤
    
    return bitmap_map;
}

// Function to perform global planning
/**
 * @brief 执行全局参考路径搜索或从冻结文件加载预存路径。
 * 
 * 流程:
 * 1. 优先从 frozen_global_path.txt 加载确定性参考路径;
 * 2. 若文件不存在，则调用混合 A* (Hybrid A*) 在膨胀栅格地图上搜索粗路径;
 * 3. 若混合 A* 失败则回退至 RRT*，最后通过 QP 凸优化进行几何平滑。
 * 
 * @param[in] bitmap_map 二值占用地图数据。
 * @param[in] config 包含起终点坐标的工况配置。
 * @param[out] m_map_info 导出的全局路径坐标记录 [x, y, heading]。
 * @param[in] ha_params 混合 A* 搜索启发式与网格参数。
 * @return GlobalPlan 平滑后的全局几何参考路径。
 */
GlobalPlan perform_global_planning(const MapData& bitmap_map, const RunConfig& config, std::vector<std::vector<double>>& m_map_info, const HybridAStarParams& ha_params) {
    GlobalPlan global_plan;
    bool enable_planning = true;
    Eigen::Vector3d start(config.start_x, config.start_y, config.start_theta);
    Eigen::Vector3d goal(config.goal_x, config.goal_y, config.goal_theta);
    
    if (!replay_load_path(global_plan) && !plan_global_path(bitmap_map, start, goal, global_plan, ha_params, enable_planning)) {
        std::cerr << "plan_global_path failed, fallback to load_map" << std::endl;
        auto fallback_info = load_map(config.start_x, config.start_y, config.start_theta);
        std::vector<Point> fallback_points;
        fallback_points.reserve(fallback_info[0].size());
        for (size_t i = 0; i < fallback_info[0].size(); ++i) {
            fallback_points.emplace_back(fallback_info[0][i], fallback_info[1][i], fallback_info[2][i]);
        }
        global_plan.set_plan(fallback_points);
    }

    // Update m_map_info interface
    const auto& planned_points = global_plan.get_points();
    m_map_info[0].clear(); m_map_info[1].clear(); m_map_info[2].clear();
    m_map_info[0].reserve(planned_points.size());
    m_map_info[1].reserve(planned_points.size());
    m_map_info[2].reserve(planned_points.size());
    for (const auto& p : planned_points) {
        m_map_info[0].push_back(p.x);
        m_map_info[1].push_back(p.y);
        m_map_info[2].push_back(p.heading);
    }
    
    std::cout << "Global path planned. m_map_info points: " << planned_points.size() << std::endl;
    std::cout << "Global path planned. m_map_info points: " << planned_points.size() << std::endl;
    save_m_map_info(m_map_info, config.solver_type, config.selected_map);
    
    return global_plan;
}

// Function to finalize solver parameters (derived values)
/**
 * @brief 动态派生铰接角速度动态约束上下界 [gamma_dot_min, gamma_dot_max]。
 * @param[in,out] arg 算法参数结构体。
 */
void finalize_parameters(Arg& arg) {
    SystemModel tmp_model;
    ArticulatedLimits lims = compute_articulated_limits(tmp_model, 0.35);
    
    // Derived parameters that depend on calculations
    double omega_bound = std::max(0.1, 0.5 * arg.desire_speed * lims.kappa_max);
    omega_bound = std::min(omega_bound, 0.4);
    
    // Restore derived values as they depend on system model limits
    arg.gamma_dot_max = omega_bound;
    arg.gamma_dot_min = -omega_bound;
    
    std::cout << "Computed articulated limits: kappa_max=" << lims.kappa_max << ", R_min=" << lims.R_min << std::endl;
    std::cout << "Derived gamma_dot limits: [" << arg.gamma_dot_min << ", " << arg.gamma_dot_max << "]" << std::endl;
}

// Function to initialize obstacles
/**
 * @brief 初始化动态障碍物集合，生成各障碍物未来预测轨迹。
 * @param[in] global_plan 全局参考路径。
 * @param[in] bitmap_map 地图数据。
 * @param[in] arg 算法参数 (包含预测时域 N, 步长 dt)。
 * @param[in] run_config 包含障碍物数量、车速与初态间距的配置。
 * @return ObstacleData 包含各障碍物当前状态与未来 N 步预测轨迹的容器。
 */
ObstacleData initialize_obstacles(const GlobalPlan& global_plan, const MapData& bitmap_map, const Arg& arg, const RunConfig& run_config) {
    ObstacleData obs_data;
    OccupancyGrid grid = make_occupancy_grid(bitmap_map, 0.1, 0.5);
    std::vector<State> obs_initial_states;
    if (!replay_load_obstacles(obs_initial_states)) obs_initial_states = generate_obstacles(global_plan, grid, run_config.obstacle_count, run_config.obstacle_distance, run_config.obstacle_speed);
    
    for(const auto& obs_state : obs_initial_states) {
        Trajectory obs_trj = predict_obstacle_trajectory(obs_state, arg.dt, arg.N);
        obs_data.trajectories.push_back(obs_trj);
        obs_data.current_states.push_back(obs_state);
    }
    return obs_data;
}

// Function to update obstacle states
/**
 * @brief 仿真推进单步，更新动态障碍物状态并重新计算其未来时域预测轨迹。
 * @param[in,out] obs_data 障碍物数据容器。
 * @param[in] arg 包含时域 N 与步长 dt 的参数结构体。
 */
void update_obstacle_states(ObstacleData& obs_data, const Arg& arg) {
    for(size_t obs_idx = 0; obs_idx < obs_data.current_states.size(); obs_idx++) {
        State& current_obs_state = obs_data.current_states[obs_idx];
        double obs_x = current_obs_state[0] + current_obs_state[3] * cos(current_obs_state[2]) * arg.dt;
        double obs_y = current_obs_state[1] + current_obs_state[3] * sin(current_obs_state[2]) * arg.dt;
        double obs_theta = current_obs_state[2];
        double obs_v = current_obs_state[3];
        current_obs_state = State(obs_x, obs_y, obs_theta, obs_v);
        
        obs_data.trajectories[obs_idx] = predict_obstacle_trajectory(current_obs_state, arg.dt, arg.N);
    }
}

/**
 * @brief 规划算法复现主函数。
 */
int main(int argc, char** argv){
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::cout << "=== Program Starting ===" << std::endl;
    std::cout << "Initializing variables..." << std::endl;
    
    // 0. 加载三份核心 JSON 配置文件 (工况、iLQR 算法、混合 A*)
    Arg arg;
    SystemModel system_model;
    HybridAStarParams ha_params;
    RunConfig run_config;
    // Load parameters from config files
    load_config("config/main.json", "config/ilqr.json", "config/hybrid_astar.json", arg, system_model, ha_params, run_config);
    finalize_parameters(arg); // Optional: calculate derived values

    // 1. 解析命令行覆盖参数并记录实际生效运行环境
    parse_arguments(argc, argv, run_config);
    replay_runtime(arg, system_model, run_config);
    
    std::cout << "======================== solver_type: " << run_config.solver_type << " ========================" <<std::endl;
    std::cout << "Start: (" << run_config.start_x << ", " << run_config.start_y << ", " << run_config.start_theta << ") | "
              << "Goal: (" << run_config.goal_x << ", " << run_config.goal_y << ", " << run_config.goal_theta << ")" << std::endl;

    // 2. 加载高程与占用栅格地图
    MapData bitmap_map = load_and_process_map(run_config.selected_map, run_config.solver_type);
    
    // 3. 执行全局几何参考路径规划或加载预存冻结路径
    std::vector<std::vector<double>> m_map_info(3);
    std::vector<std::vector<double>> global_plan_log(3);
    GlobalPlan global_plan = perform_global_planning(bitmap_map, run_config, m_map_info, ha_params);
    
    // Sync global plan log
    global_plan_log[0] = m_map_info[0];
    global_plan_log[1] = m_map_info[1];
    global_plan_log[2] = m_map_info[2];

    // 4. Initialize Parameters (Already done via load_config)
    // Arg arg = initialize_parameters(); // Removed

    // 5. 初始化自车初始运动学状态与几何模型 (x, y, theta, gamma)
    Vehicle ego;
    ego.set_state(run_config.start_x, run_config.start_y, run_config.start_theta, 0);
    ego.set_global_plan(global_plan);
    ego.set_model(system_model); // Use loaded system model
    
    std::vector<std::vector<double>> ego_log(4);
    for(int i=0;i<4;i++){
        ego_log[i].push_back(ego.get_state()[i]);
    }

    // 6. 初始化动态障碍物并在 prepare-scene 模式下保存场景文本后退出
    ObstacleData obs_data = initialize_obstacles(global_plan, bitmap_map, arg, run_config);
    replay_save_scene(global_plan, obs_data.current_states);
    if (argc > 1 && std::string(argv[1]) == "--prepare-scene") return 0;

    // 7. 实例化 C-iLQR 与 AL-iLQR 优化求解器
    CILQRSolver cilqr_solver(ego, obs_data.trajectories, arg, run_config.selected_map, "cilqr");
    ALILQRSolver alilqr_solver(ego, obs_data.trajectories, arg);
    Solution solution;
    Control cur_ctrl;
    State cur_state = ego.get_state();

    std::cout << "Variables initialized successfully." << std::endl;

    // 8. 滚动时域优化推进主循环 (Receding Horizon Control Loop)
    for(int i = 0; i < run_config.ITER; i++){
        std::cout<<"***** Iter ***** " << i <<std::endl;
        
        // 执行单步轨迹优化并统计 CPU 求解耗时
        clock_t start = clock();
        if(run_config.solver_type == "cilqr"){
            solution = cilqr_solver.solve(cur_state, obs_data.trajectories);
        } else if(run_config.solver_type == "alilqr"){
            solution = alilqr_solver.solve(cur_state, obs_data.trajectories);
        } else {
            std::cerr << "Wrong Solver Type :" << run_config.solver_type << std::endl; 
        }
        clock_t end = clock();
        double cpu_time_used = static_cast<double>(end - start) / CLOCKS_PER_SEC;
        solution.solve_time_ms = cpu_time_used * 1000;
        std::cout << "CPU time used: " << solution.solve_time_ms << " ms\n";
        
        // 评估当前规划轨迹的最大非线性约束违例度 CMax
        ego.set_state(cur_state);
        ego.set_local_plan();
        double cmax = compute_max_violation(solution, ego, obs_data.trajectories, arg);
        std::cout << "CMax: " << cmax << "\n";

        // 应用首步控制输入，通过非线性离散运动学模型更新自车状态
        if (solution.control_sequence.controls.empty()) { std::cerr << "REPLAY_FAILURE empty_control_sequence at cycle " << i << std::endl; return 2; }
        cur_ctrl = solution.control_sequence.controls[0];
        cur_state = ego.get_model().dynamics(cur_state, cur_ctrl);
        
        // 记录自车执行后的状态历史
        for(int j=0;j<4;j++){
            ego_log[j].push_back(cur_state[j]);
        }
        
        // 步进更新动态障碍物物理位置并外推新时域预测轨迹
        update_obstacle_states(obs_data, arg);

        // 序列化导出当前周期的全量数据帧 (JSON 格式)
        replay_frame(ego_log, obs_data.trajectories, solution, global_plan, ego.get_model(), arg, run_config.solver_type, run_config.selected_map, i);
    }

    return 0;
}
