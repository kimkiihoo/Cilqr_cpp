// Original functions below are extracted verbatim; only log-path I/O is replaced.
/**
 * @file support.cpp
 * @brief 动态库运行时辅助支持函数实现 (物理约束计算、障碍物外推预测与参数派生)。
 * 
 * 本实现文件为 ctypes 导出的 C-API 提供底层几何计算支持，保持算法逻辑完全不变，
 * 仅对内部资源定位与调试输出重定向至独立沙盒运行目录。
 */
#include "ilqr.h"
#include "utils.h"
#include <filesystem>
#include <cmath>
#include <string>

/// 运行时日志与场景输出的绝对根目录路径
std::string original_bridge_log_root;

/**
 * @brief 解析相对资源路径为当前沙盒运行环境下的绝对路径。
 * 
 * @param[in] path 相对文件路径。
 * @return std::string 结合根目录后的完整路径。
 */
std::string resolve_resource_path(const std::string& path) { return original_bridge_log_root + "/" + path; }

/**
 * @brief 确保指定目录及其父级目录在文件系统中存在。
 * 
 * @param[in] path 待创建的目标目录路径。
 */
void ensure_directory_exists(const std::string& path) { std::filesystem::create_directories(path); }

/**
 * @brief 计算铰接式车辆在给定的最大转向角几何限制下的运动学极限参数。
 * 
 * 转向曲率公式:
 *   \kappa = \frac{\tan(\gamma)}{l_r + l_f \cos(\gamma)}
 *   R_{min} = \frac{1}{\kappa_{max}}
 * 
 * @param[in] model 铰接车辆几何参数模型 (前后轴距 lf, lr [m])。
 * @param[in] gamma_max_mech 铰接角机械硬极限值 [rad]。
 * @return ArticulatedLimits 包含最大曲率 kappa_max [1/m] 与最小转弯半径 R_min [m]。
 */
ArticulatedLimits compute_articulated_limits(const SystemModel& model, double gamma_max_mech) {
    // 保守推导：使用运动学近似 kappa = tan(gamma) / L_eff，其中有效轴距 L_eff = lr + lf * cos(gamma)
    // 取机械上限 gamma_max_mech 的保守截断值，计算最大可能曲率与最小半径
    double lf = model.lf;
    double lr = model.lr;
    double gamma = std::min(std::max(gamma_max_mech, 0.0), 0.6); // 保守截断：不超过约 0.6 rad (~34.4 deg)
    double L_eff = lr + lf * std::cos(gamma);
    double kappa = std::abs(std::tan(gamma) / std::max(L_eff, 1e-3));
    ArticulatedLimits limits;
    limits.kappa_max = kappa;
    limits.R_min = (kappa > 1e-6) ? 1.0 / kappa : 1e6;
    return limits;
}

/**
 * @brief 采用匀速直线运动模型外推预测动态障碍物在未来 N 步的离散状态轨迹。
 * 
 * 预测递推公式:
 *   x_k = x_0 + v \cdot \cos(\theta) \cdot (k \cdot \Delta t)
 *   y_k = y_0 + v \cdot \sin(\theta) \cdot (k \cdot \Delta t)
 *   \theta_k = \theta_0, \quad v_k = v_0
 * 
 * @param[in] initial_state 障碍物当前时刻初始状态 [x (m), y (m), theta (rad), v (m/s)]。
 * @param[in] dt 离散预测时间步长 [s]。
 * @param[in] N 预测时域步数。
 * @return Trajectory 包含 N 个离散预测状态的状态序列。
 */
Trajectory predict_obstacle_trajectory(const State& initial_state, double dt, int N) {
    Trajectory predicted_trajectory;
    
    // 从初始状态提取二维位置、航向角和标量线速度
    double x = initial_state[0];      // x 坐标 [m]
    double y = initial_state[1];      // y 坐标 [m]
    double theta = initial_state[2];  // 航向角 [rad]
    double v = initial_state[3];      // 巡航线速度 [m/s]
    
    // 基于匀速直线运动 (Constant Velocity) 假设生成离散预测点
    for (int i = 0; i < N; i++) {
        // 计算当前预测时刻的时间跨度 [s]
        double current_time = i * dt;
        
        // 匀速直线运动外推
        double pred_x = x + v * cos(theta) * current_time;
        double pred_y = y + v * sin(theta) * current_time;
        double pred_theta = theta;  // 假设障碍物航向保持恒定
        double pred_v = v;          // 假设障碍物速度保持恒定
        
        // 创建预测状态并添加到轨迹序列中
        State predicted_state(pred_x, pred_y, pred_theta, pred_v);
        predicted_trajectory.push_back(predicted_state);
    }
    
    return predicted_trajectory;
}

/**
 * @brief 结合车辆物理限制与期望速度，自动派生并修正铰接角速度动态约束区间。
 * 
 * 派生规则:
 *   \omega_{bound} = \mathrm{clamp}(0.5 \cdot v_{des} \cdot \kappa_{max}, 0.1, 0.4) \quad [\mathrm{rad/s}]
 *   \dot{\gamma}_{min} = -\omega_{bound}, \quad \dot{\gamma}_{max} = \omega_{bound}
 * 
 * @param[in,out] arg 算法优化配置参数结构体，动态覆写 gamma_dot_min 与 gamma_dot_max。
 */
void finalize_parameters(Arg& arg) {
    SystemModel tmp_model;
    ArticulatedLimits lims = compute_articulated_limits(tmp_model, 0.35);
    
    // 基于期望车速与最大曲率计算角速度理论需求并施加安全上下界限 [0.1, 0.4] rad/s
    double omega_bound = std::max(0.1, 0.5 * arg.desire_speed * lims.kappa_max);
    omega_bound = std::min(omega_bound, 0.4);
    
    // 更新算法内部生效的铰接角速度约束上下限
    arg.gamma_dot_max = omega_bound;
    arg.gamma_dot_min = -omega_bound;
    
    std::cout << "Computed articulated limits: kappa_max=" << lims.kappa_max << ", R_min=" << lims.R_min << std::endl;
    std::cout << "Derived gamma_dot limits: [" << arg.gamma_dot_min << ", " << arg.gamma_dot_max << "]" << std::endl;
}
