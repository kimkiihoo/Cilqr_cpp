#pragma once

/**
 * ============================================================================
 * 文件名称: config_loader.h
 * 文件功能: 算法运行参数与车辆动力学模型 JSON 配置文件的加载与解析声明
 * ============================================================================
 */

#include <string>
#include "ilqr.h"
#include "hybrid_astar.h"
#include "utils.h"

/**
 * @brief 从指定的多个 JSON 配置文件中解析并填充算法参数与物理模型
 * 
 * @param main_config_file 主运行与场景地图配置文件路径 (main.json)
 * @param ilqr_config_file iLQR 优化器权重、物理边界与屏障参数配置文件路径 (ilqr.json)
 * @param ha_config_file   Hybrid A* 混合搜索与启发式参数配置文件路径 (hybrid_astar.json)
 * @param arg              输出参数结构体，存储 iLQR 优化权重与控制约束
 * @param system_model     输出车辆非线性运动学物理尺寸与状态方程模型
 * @param ha_params        输出 Hybrid A* 几何路径搜索超参数
 * @param run_config       输出全局起点、目标点与工况配置
 */
void load_config(const std::string& main_config_file, 
                 const std::string& ilqr_config_file,
                 const std::string& ha_config_file,
                 Arg& arg, 
                 SystemModel& system_model, 
                 HybridAStarParams& ha_params,
                 RunConfig& run_config);
