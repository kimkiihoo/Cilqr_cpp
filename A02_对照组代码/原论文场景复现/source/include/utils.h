#pragma once

/**
 * ============================================================================
 * 文件名称: utils.h
 * 文件功能: 路径平滑 (QP 优化)、地图与障碍物处理、几何碰撞检测与辅助工具函数
 * ============================================================================
 */

#include <vector>
#include <cstdint>
#include "Eigen/Eigen"
#include "common_types.h"
#include <memory>
#include <string>
#include <sstream>
#include <fstream>

// 前置类与结构体声明
class GlobalPlan;
class Trajectory;
struct Solution;
class SystemModel;
struct Arg;
class Vehicle;
struct HybridAStarParams;

/**
 * @brief 基础优化参数结构体
 */
struct Params {
    Eigen::MatrixXd Q;    // 状态加权矩阵
    Eigen::MatrixXd R;    // 控制加权矩阵
    Eigen::MatrixXd Qf;   // 终端状态加权矩阵
    double dt;            // 步长 [单位: s]
    int N;                // 预测步数
    int max_iter;         // 最大迭代数
    double tol;           // 收敛容差
};

/**
 * @brief 车辆几何外形尺寸结构体
 */
struct VehicleModel {
    double L1;     // 前车厢有效特征长度 [单位: m]
    double L2;     // 后车厢有效特征长度 [单位: m]
    double width;  // 车体全宽 [单位: m]
};

/**
 * @brief 基于铰接几何推导的最大曲率与最小转弯半径
 */
struct ArticulatedLimits {
    double kappa_max; // 路径允许最大曲率 [单位: 1/m]
    double R_min;     // 路径允许最小转弯半径 [单位: m]
};

// 绘图与可视化诊断辅助函数
void my_plot(const std::vector<std::vector<double>>& global_plan_log,
    const std::vector<std::vector<double>>& ego_log,
    const Trajectory& obs_traj,
    const Solution& solution,
    const MapData* map_data = nullptr);

void dynamic_plot(const std::vector<std::vector<double>>& global_plan_log,
    const std::vector<std::vector<double>>& ego_log,
    const std::vector<Trajectory>& obs_trajectories,
    const Solution& solution,
    const MapData* map_data,
    const GlobalPlan& global_plan,
    const SystemModel& vehicle_model,
    const Arg& arg,
    const std::string& solver_type,
    const std::string& map_name);

void draw_bitmap_debug(const MapData& map_data, const std::string& output_path = "");
void save_map_data(const MapData* map_data, const std::string& solver_type, const std::string& map_name);
std::string resolve_resource_path(const std::string& relative_path);
void save_m_map_info(const std::vector<std::vector<double>>& m_map_info, const std::string& solver_type, const std::string& map_name);
void ensure_directory_exists(const std::string& path);

// 地图加载与转换接口
std::vector<std::vector<double>> load_map(double startx=0, double starty=0, double theta=0);
MapData load_bitmap_map(const std::string& file_path);
SemanticMapData load_semantic_map(const std::string& file_path);
void fill_global_path_points(std::vector<std::vector<double>>& global_plan_log);
void set_global_path(GlobalPlan& global_plan, const std::vector<std::vector<double>>& global_plan_log);

/**
 * @brief 全局路径规划封装函数 (优先调用混合 A*，失败时回退)
 */
bool plan_global_path(const MapData& bitmap_map,
                      const Eigen::Vector3d& start,
                      const Eigen::Vector3d& goal,
                      GlobalPlan& out_plan,
                      const HybridAStarParams& ha_params,
                      bool enable_rrt = true);

/**
 * @brief 基于铰接车辆几何尺寸与机械角限位推导物理最大曲率与最小转弯半径
 */
ArticulatedLimits compute_articulated_limits(const SystemModel& model, double gamma_max_mech);

void init_params(Params& params);
void init_vehicle_model(VehicleModel& vehicle_model);
void init_obstacle_trajectory(Trajectory& obs_traj);

/**
 * @brief 根据动态障碍物初始状态与恒速假设外推预测未来 N 步轨迹
 * 
 * @param initial_state 障碍物当前位姿 [x (m), y (m), heading (rad), speed (m/s)]
 * @param dt 离散时间步长 [单位: s]
 * @param N 预测外推步数
 * @return Trajectory 外推轨迹序列
 */
Trajectory predict_obstacle_trajectory(const State& initial_state, double dt, int N);

/**
 * @brief 从高程地图生成二值占用网格图并执行各向同性膨胀
 */
OccupancyGrid make_occupancy_grid(const MapData& map, double elevation_threshold_ratio = 0.1, double inflate_radius_m = 0.5);

// 几何碰撞检测接口
bool is_collision_free_segment(const OccupancyGrid& grid, const Eigen::Vector2d& a, const Eigen::Vector2d& b);
bool is_collision_free_polyline(const OccupancyGrid& grid, const std::vector<Eigen::Vector2d>& pts);
double nearest_obstacle_distance_world(const OccupancyGrid& grid, const Eigen::Vector2d& p);

std::vector<State> generate_obstacles(const GlobalPlan& global_plan, const OccupancyGrid& grid, int num_obstacles, double distance_from_path, double obstacle_speed);

/**
 * @brief RRT* 采样规划算法参数结构体
 */
struct RRTStarParams {
    int max_iters = 20000;             // 最大采样迭代上限
    double step_size = 0.5;            // 单步增量步长 [单位: m]
    double goal_tolerance = 1.0;       // 到达目标容差 [单位: m]
    double rewire_radius_factor = 1.5; // 近邻重布线半径倍率
    double inflation_radius = 0.5;     // 障碍物膨胀半径 [单位: m]
    double goal_sample_rate = 0.1;     // 目标点启发采样偏向概率
    double corridor_sample_rate = 0.2; // 参考走廊区域采样偏向概率
    double corridor_width = 2.0;       // 采样走廊半宽 [单位: m]
    int connect_attempt_interval = 100;// 双向树快速连接触发周期
    int max_connect_steps = 200;       // 连接尝试单次最大扩展步数
    double cost_obstacle_weight = 0.1; // 障碍物斥力代价权重
    double max_clearance = 10.0;       // 归一化最大净距 [单位: m]
};

bool rrt_star_plan(const MapData& map,
                   const Eigen::Vector3d& start,
                   const Eigen::Vector3d& goal,
                   std::vector<Point>& out_points,
                   const RRTStarParams& params);

/**
 * @brief 基于凸二次规划 (QP) 的折线路径曲率平滑与走廊约束优化算法
 * 
 * 优化模型:
 * min \sum ||p_{i-1} - 2 p_i + p_{i+1}||^2 (曲率最小化平滑代价) + \lambda \sum ||p_i - p_{orig, i}||^2
 * s.t. p_i 严格限制在自适应安全凸走廊范围内 (通过局部占用网格射线检测求得)
 * 
 * @param path 输入离散路点坐标序列
 * @param grid 静态环境占用网格
 * @param iterations QP 迭代求解循环步数
 * @param curvature_weight 曲率平滑项二次型惩罚权重 (默认 1200.0)
 * @return std::vector<Eigen::Vector2d> 优化平滑后的高阶连续路径点序列
 */
std::vector<Eigen::Vector2d> optimize_path_qp(const std::vector<Eigen::Vector2d>& path,
                                              const OccupancyGrid& grid,
                                              int iterations = 35,
                                              double curvature_weight = 1200.0);

// 计算当前解的最大物理与几何约束违背量
double compute_max_violation(const Solution& solution, Vehicle& ego, const std::vector<Trajectory>& obs_list, const Arg& arg);
