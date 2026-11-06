#pragma once

/**
 * ============================================================================
 * 文件名称: common_types.h
 * 文件功能: 铰接式矿用车辆规划与控制系统的基础数据类型、状态/控制空间定义与地图结构
 *
 * 物理量与量纲定义:
 * 1. 车辆状态空间 State (Eigen::Vector4d):
 *    - State[0]: 前轴中心大地坐标 X [单位: m]
 *    - State[1]: 前轴中心大地坐标 Y [单位: m]
 *    - State[2]: 前车体全局航向角 theta (Yaw Angle) [单位: rad]
 *    - State[3]: 前后车体相对铰接角 gamma (Articulation Angle) [单位: rad]
 *
 * 2. 控制输入空间 Control (Eigen::Vector2d):
 *    - Control[0]: 期望纵向行驶线速度 v [单位: m/s]
 *    - Control[1]: 期望铰接角变化率 gamma_dot = d(gamma)/dt [单位: rad/s]
 *
 * 3. 几何参考点 Point:
 *    - x, y: 平面坐标 [单位: m]
 *    - heading: 曲线切线航向角 [单位: rad]
 * ============================================================================
 */

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <cstdint>
#include <string>

/**
 * @brief 运行阶段全局工况与场景配置参数结构体
 */
struct RunConfig {
    double start_x = 70;                 // 起始位置 X 坐标 [单位: m]
    double start_y = 60;                 // 起始位置 Y 坐标 [单位: m]
    double start_theta = 1.2;            // 起始前车体航向角 [单位: rad]
    double goal_x = 180;                 // 终点目标位置 X 坐标 [单位: m]
    double goal_y = 127;                 // 终点目标位置 Y 坐标 [单位: m]
    double goal_theta = 1.8;             // 终点期望前车体航向角 [单位: rad]
    double ITER = 280;                   // 仿真推进最大执行步数 [无量纲]
    std::string solver_type = "cilqr";   // 选用求解器类型 ("cilqr" 或 "alilqr")
    std::string selected_map = "B301";   // 选用地图场景标识 ("B201", "B207", "B301")
    double obstacle_speed = 1.0;         // 动态障碍物恒定巡航速度 [单位: m/s]
    int obstacle_count = 4;              // 环境中动态障碍物总数 [无量纲]
    double obstacle_distance = 3.5;      // 障碍物间距或初始偏移距离 [单位: m]
};

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 状态与控制输入类型别名定义
typedef Eigen::Vector4d State;    // 4维状态向量: [x (m), y (m), theta (rad), gamma (rad)]^T
typedef Eigen::Vector2d Control;  // 2维控制向量: [v (m/s), gamma_dot (rad/s)]^T

/**
 * @brief 二维平面带方向的几何路径路点结构体
 */
struct Point{
    Point(double X,double Y,double Heading){
        x = X;
        y = Y;
        heading = Heading;
    }
    bool operator==(const Point& other) const {
        return (x == other.x && y == other.y && heading == other.heading);
    }
    double x;        // 大地坐标 X [单位: m]
    double y;        // 大地坐标 Y [单位: m]
    double heading;  // 路径切向方位角 [单位: rad]
};

/**
 * @brief 矿区高程/地形栅格地图数据结构体
 */
struct MapData {
    int width;                              // 地图栅格横向列数 [格数]
    int height;                             // 地图栅格纵向行数 [格数]
    double resolution;                      // 单个栅格的实际物理物理分辨率 [单位: m/格]
    std::vector<double> origin;             // 大地坐标系下的地图左下角原点 [x (m), y (m)]
    double max_elevation;                   // 地图最大高程上限 [单位: m]
    std::vector<std::vector<double>> data;  // 逐栅格高程矩阵数据 [单位: m]
};

/**
 * @brief 语义地图离散特征点结构体
 */
struct SemanticMapPoint {
    double x;  // 特征点 X 坐标 [单位: m]
    double y;  // 特征点 Y 坐标 [单位: m]
    int type;  // 语义类型标识 (如道路边缘、作业区边界、障碍物等)
};

/**
 * @brief 语义地图点云数据集结构体
 */
struct SemanticMapData {
    std::vector<SemanticMapPoint> points;  // 语义特征点集合
};

/**
 * @brief 车辆运动规划二值占用网格图 (Occupancy Grid Map)
 */
struct OccupancyGrid {
    int width;                   // 网格横向栅格数 [格数]
    int height;                  // 网格纵向栅格数 [格数]
    double resolution;           // 单栅格实际物理分辨率 [单位: m/格]
    double origin_x;             // 地图原点大地坐标 X [单位: m]
    double origin_y;             // 地图原点大地坐标 Y [单位: m]
    std::vector<uint8_t> cells;  // 栅格占用状态 (0: 自由可通行, 1: 静态障碍物或边界占用)
};
