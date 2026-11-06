#pragma once

/**
 * ============================================================================
 * 文件名称: hybrid_astar.h
 * 文件功能: 考虑车辆非完整性约束的混合 A* (Hybrid A*) 初始几何路径规划算法定义
 * ============================================================================
 */

#include "utils.h"
#include "common_types.h"
#include <vector>
#include <Eigen/Eigen>

/**
 * @brief 混合 A* 算法超参数配置结构体
 */
struct HybridAStarParams {
    // 空间离散化与航向角分辨率
    double grid_resolution = 0.1;          // 空间位置二维网格分辨率 [单位: m]
    double heading_resolution = 10.0;      // 连续航向角离散化角度区间 [单位: deg]

    // 车辆运动基元与代价权重
    double turning_radius = 4.0;           // 车辆最小转弯几何半径 R_min [单位: m]
    double move_step = 1.0;                // 前进运动基元离散推进步长 [单位: m]
    double move_step_backwards = 0.3;      // 倒车运动基元推进步长 [单位: m]
    bool   allow_reverse = false;          // 是否允许倒车搜索动作
    int    num_gamma_angles = 5;           // 铰接转向角候选离散分支数 [无量纲]
    double steering_penalty = 0.2;         // 打舵偏角代价增益系数
    double direction_change_penalty = 1.5; // 前进/倒退挡位换向惩罚代价
    double backwards_penalty = 1.2;        // 倒车行为代价惩罚乘子
    double heuristic_weight = 1.0;         // 启发式代价评估权重系数 (A* Heuristic Factor)

    // 搜索剪枝与终止容差
    int    max_iterations = 60000;         // A* 优先队列节点扩展最大迭代上限
    int    num_nodes_to_keep = 30000;      // Open 表节点保留上限 (防止内存溢出剪枝)

    // 静态障碍物膨胀与目标到达容差
    double inflation_radius = 2.0;         // 静态地图障碍物安全膨胀半径 [单位: m]
    double goal_tolerance_xy = 0.8;        // 终点目标位置欧氏距离容差 [单位: m]
    double goal_tolerance_heading = 15.0;  // 终点目标航向角偏角容差 [单位: deg]

    // 避障清距代价 (Clearance Cost, 越靠近障碍物惩罚越高)
    double clearance_weight = 12.0;        // 障碍物清距代价权重系数 (偏好走宽阔区域)
    double desired_clearance = 3.0;        // 期望与障碍物保持的最小安全距离 [单位: m]
    double max_clearance = 12.0;           // 清距代价归一化截断上限 [单位: m]
    double min_rs_clearance = 1.5;         // Reeds-Shepp (RS) 解析连接曲线允许的最小障碍净距 [单位: m]
};

/**
 * @brief 二维网格最短距离场 (Dijkstra 预计算启发式地图)
 */
struct DijkstraDistanceMap {
    std::vector<std::vector<double>> distances; // 二维栅格最短欧氏距离图 distances[y][x] [单位: m]
    int width;                                   // 栅格横向列数
    int height;                                  // 栅格纵向行数
    double resolution;                           // 栅格物理分辨率 [单位: m/格]
    double origin_x;                             // 大地坐标系原点 X [单位: m]
    double origin_y;                             // 大地坐标系原点 Y [单位: m]
    bool is_valid;                               // 距离图有效性标志位
    
    DijkstraDistanceMap() : width(0), height(0), resolution(0.1), origin_x(0), origin_y(0), is_valid(false) {}
    
    // 获取世界坐标系 (world_x, world_y) 到目标点的最短无碰撞网格距离
    double getDistance(double world_x, double world_y) const;
};

/**
 * @brief 预计算从目标点到整张地图所有连通栅格的 Dijkstra 无碰撞距离场
 * 
 * @param grid 静态占用二值栅格网格
 * @param goal_x 终点目标坐标 X [单位: m]
 * @param goal_y 终点目标坐标 Y [单位: m]
 * @return DijkstraDistanceMap 距离图对象
 */
DijkstraDistanceMap computeDijkstraDistances(const OccupancyGrid& grid, 
                                           double goal_x, double goal_y);

/**
 * @brief 混合 A* 运动规划主接口函数
 * 
 * @param map 地图高程与原始栅格数据
 * @param start 起始三维位姿 [x (m), y (m), theta (rad)]
 * @param goal 终点目标位姿 [x (m), y (m), theta (rad)]
 * @param out_points 输出搜索生成的无碰撞路点序列 (含位姿与航向)
 * @param params 混合 A* 搜索超参数
 * @return true 规划成功找到可行路径; false 规划失败
 */
bool hybrid_astar_plan(const MapData& map,
                       const Eigen::Vector3d& start,
                       const Eigen::Vector3d& goal,
                       std::vector<Point>& out_points,
                       const HybridAStarParams& params);
