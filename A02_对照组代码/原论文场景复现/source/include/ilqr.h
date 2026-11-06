#ifndef ILQR_H
#define ILQR_H

/**
 * ============================================================================
 * 文件名称: ilqr.h
 * 文件功能: 约束迭代线性二次型调节器 (C-iLQR) 与增广拉格朗日 iLQR (AL-iLQR) 核心定义
 *
 * 算法数学原理:
 * 1. 车辆状态空间与运动学方程 (4维状态, 2维控制):
 *    State   X = [x, y, \theta, \gamma]^T  [单位: m, m, rad, rad]
 *    Control U = [v, \dot{\gamma}]^T        [单位: m/s, rad/s]
 *    运动学离散状态转移方程:
 *    x_{k+1} = x_k + v_k * cos(\theta_k) * dt
 *    y_{k+1} = y_k + v_k * sin(\theta_k) * dt
 *    \theta_{k+1} = \theta_k - \frac{v_k * sin(\gamma_k) + l_r * \dot{\gamma}_k}{l_r + l_f * cos(\gamma_k)} * dt
 *    \gamma_{k+1} = \gamma_k + \dot{\gamma}_k * dt
 *
 * 2. C-iLQR (基于控制障碍函数/松弛屏障惩罚):
 *    - 目标函数融合状态跟踪误差二次型、控制偏差二次型、以及道路边界与椭圆障碍物的软屏障代价 (Barrier Function)。
 *    - 采用 Levenberg-Marquardt 正则化阻尼技术 (\lambda) 保证动作价值 Hessian 正定性。
 *
 * 3. AL-iLQR (基于增广拉格朗日法 / ALTRO 架构):
 *    - 双层优化框架: 外层更新一阶拉格朗日乘子 \lambda 与二阶惩罚权重 \mu；内层调用未约束 iLQR 优化增广目标函数。
 *    - 显式满足控制盒式界限、铰接角物理机械极限与避障不等式约束。
 * ============================================================================
 */

#include "common_types.h"
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <algorithm>

using namespace Eigen;

/**
 * @brief 优化算法超参数、车辆物理极限与代价函数权重配置结构体
 */
struct Arg{
    // 仿真时钟与推进步长
    double tf = 1000;                     // 仿真总时长上限 [单位: s]
    double dt = 0.1;                      // 离散优化与模型推进步长 [单位: s]

    // iLQR 求解器收敛与正则化超参数
    int N = 50;                           // 滚动优化预测时域步数 (Prediction Horizon) [无量纲]
    double tol = 1;                       // 目标函数收敛绝对变化量容差
    double rel_tol = 1e-5;                // 目标函数收敛相对变化率容差
    int max_iter = 50;                    // 单控制周期最大允许内迭代次数
    double lamb_init = 7.0;               // LM 正则化阻尼因子初始值 \lambda_0
    double lamb_factor = 2;               // 正则化阻尼因子自适应缩放倍率
    double lamb_max = 6000;               // 正则化阻尼因子允许上限

    // 纯跟踪 (Pure Pursuit) 冷启动参考线搜索参数
    double kv = 0.3;                      // 前视距离关于车速的比例系数 [单位: s]
    double kp = 0.8;                      // 车速比例控制器增益
    double ld0 = 3;                       // 基础最小前视距离 [单位: m]
    double ld_min = 3;                    // 前视距离绝对下限 [单位: m]
    double ld_max = 20;                   // 前视距离绝对上限 [单位: m]

    // 目标代价与参考状态
    double desire_speed = 15;             // 期望巡航速度 [单位: m/s]
    double desire_heading = 0;            // 期望航向角 [单位: rad]
    bool if_cal_obs_cost = true;          // 是否激活障碍物避障代价项
    bool if_cal_lane_cost = true;         // 是否激活车道道路边界走廊惩罚代价项
    bool if_cal_speed_rate_cost = true;   // 是否激活相邻步车速加速度变化率惩罚项
    double v_rate_weight = 5.0;           // 纵向加速度平滑性二次型惩罚权重 [单位: s^2/m^2]
    
    // 铰接角 gamma 物理机械硬边界与控制屏障函数 (CBF) 参数:
    // 物理约束: gamma_min <= gamma <= gamma_max
    bool if_cal_gamma_barrier = true;     // 是否对铰接角状态施加屏障函数
    double gamma_max = 1.0;               // 铰接角最大允许上限 [单位: rad]
    double gamma_max_q1 = 1.0;            // 铰接角上限屏障函数权重系数 q1
    double gamma_max_q2 = 3.0;            // 铰接角上限屏障函数曲率指数系数 q2
    double gamma_min = -1.0;              // 铰接角最小允许下限 [单位: rad]
    double gamma_min_q1 = 1.0;            // 铰接角下限屏障函数权重系数 q1
    double gamma_min_q2 = 3.0;            // 铰接角下限屏障函数曲率指数系数 q2
    
    // 铰接角速率 gamma_dot 转向执行器速度界限与屏障参数:
    // 物理约束: gamma_dot_min <= gamma_dot <= gamma_dot_max
    bool if_cal_gamma_dot_barrier = true; // 是否对控制量铰接角速度施加屏障函数
    double gamma_dot_max = 0.3;           // 铰接角速率正向允许上限 [单位: rad/s]
    double gamma_dot_max_q1 = 10.0;       // 铰接角速率上限屏障权重系数 q1
    double gamma_dot_max_q2 = 2.0;        // 铰接角速率上限屏障曲率系数 q2
    double gamma_dot_min = -0.3;          // 铰接角速率负向允许下限 [单位: rad/s]
    double gamma_dot_min_q1 = 10.0;       // 铰接角速率下限屏障权重系数 q1
    double gamma_dot_min_q2 = 2.0;        // 铰接角速率下限屏障曲率系数 q2
    
    // 纵向加速度限制 (v_k - v_{k-1}) / dt
    double acc_max = 3.0;                 // 最大允许纵向加速度 [单位: m/s^2]
    double acc_min = -4.0;                // 最大允许制动减速度负向界限 [单位: m/s^2]
    double acc_q1 = 1.0;                  // 加速度屏障代价权重系数
    double acc_q2 = 2.0;                  // 加速度屏障代价曲率系数
    
    // 道路走廊限宽与边界屏障参数
    double trace_safe_width_left = 5;     // 参考线左侧道路边界走廊安全限宽 [单位: m]
    double trace_safe_width_right = 5;    // 参考线右侧道路边界走廊安全限宽 [单位: m]
    double lane_q1 = 5;                   // 车道边界屏障代价权重系数
    double lane_q2 = 3;                   // 车道边界屏障曲率指数系数

    // 动态障碍物包络尺寸与避障安全缓冲区
    double obs_q1 = 4.25;                 // 避障屏障代价权重系数
    double obs_q2 = 3.4;                  // 避障屏障代价曲率指数系数
    double obs_length = 5;                // 动态障碍物几何包络盒全长 [单位: m]
    double obs_width = 3;                 // 动态障碍物几何包络盒全宽 [单位: m]
    double safe_a_buffer = 0.5;           // 避障膨胀椭圆沿长半轴的附加安全缓冲距离 [单位: m]
    double safe_b_buffer = 0.5;           // 避障膨胀椭圆沿短半轴的附加安全缓冲距离 [单位: m]

    // 状态与控制输入二次型加权矩阵
    Matrix4d Q;                           // 状态跟踪权重矩阵 (4x4): 对角元素分别对应 [X, Y, theta, gamma]
    Matrix2d R;                           // 控制输入权重矩阵 (2x2): 对角元素分别对应 [v, gamma_dot]
    double ref_weight = 8.0;              // 参考路径全局横向位置跟踪偏差权重因子 [单位: 1/m^2]

    Arg() {
        // 初始化状态跟踪对角加权矩阵 Q:
        // Q(0,0)=0.2, Q(1,1)=0.2 为位置跟踪权重; Q(2,2)=0.1 为航向角权重; Q(3,3)=1.5 为铰接角权重
        Q << 0.2, 0, 0, 0,
             0, 0.2, 0, 0,
             0, 0, 0.1, 0,
             0, 0, 0, 1.5;

        // 初始化控制量对角加权矩阵 R:
        // R(0,0)=1.0 为速度输入权重; R(1,1)=50.0 为高增益铰接转向角速率惩罚权重
        R << 1.0, 0,
             0, 50;
    }
};

/**
 * @brief 控制障碍函数 (CBF / Barrier Function) 及其一阶、二阶导数输出信息体
 */
struct BarrieInfo{
    double b;        // 屏障惩罚标量代价值
    VectorXd d_b;    // 屏障惩罚关于状态的一阶偏导梯度向量
    MatrixXd dd_b;   // 屏障惩罚关于状态的二阶 Hessian 对称矩阵
};

// 平面两点间几何欧氏距离计算
inline double distance(const Point& p1, const Point& p2);

// 空间路径最近路点查找
size_t find_closest_point(const std::vector<Point>& path,const State& state);

// 航向角归一化至主值区间 [-pi, pi]
inline double angle_wrap(double theta) {
    theta = fmod(theta + M_PI, 2.0*M_PI); 
    if (theta < 0.0){
        theta += 2.0*M_PI;
    }
    return theta - M_PI;
}

// 计算单项非线性屏障惩罚代价值及其一阶、二阶梯度
BarrieInfo barrierFunction(double q1, double q2, double c, VectorXd dc);

/**
 * @brief 全局参考路径点集持有类
 */
class GlobalPlan{
    private:
        std::vector<Point> points;
    public:
        std::vector<Point> get_points() const{
            return this->points;
        };
        void set_plan(const std::vector<Point>& points){
            this->points = points;
        };
};

/**
 * @brief 车辆当前滚动时域内的局部路径截取类
 */
class LocalPlan{
    private:
        std::vector<Point> points;
    public:
        std::vector<Point> get_points() const{
            return this->points;
        };
        void set_plan(const GlobalPlan& global_plan,const State& vehicle_state,size_t num_points_to_extract);
};

/**
 * @brief 铰接式矿用车辆运动学非线性状态方程与雅可比计算类
 */
class SystemModel{
    public:
        double ego_rad = 7;     // 自车等效外接包络圆半径 [单位: m]
        double lf      = 3;     // 前轴中心至铰接连接点的纵向轴距 [单位: m]
        double lr      = 3;     // 后轴中心至铰接连接点的纵向轴距 [单位: m]
        double len     = 3;     // 车体几何基准特征长度 [单位: m]
        double width   = 3;     // 车体横向全宽 [单位: m]
        double box_length = 5;  // 单节车体包络盒长度 [单位: m]
        double dt = 0.1;        // 模型离散步长 [单位: s]
        size_t N = 50;          // 预测步数
        SystemModel() = default;
        SystemModel(double dt,size_t N):dt(dt),N(N){};

        // 连续非线性运动学方程离散步前向推演
        State dynamics(const State& X, const Control& U);

        // 动力学状态转移矩阵雅可比 df/dx (4x4)
        Matrix4d get_jacobian_state(const Vector4d& X, const Vector2d& U);

        // 动力学控制输入矩阵雅可比 df/du (4x2)
        Matrix<double,4,2> get_jacobian_control(const Vector4d& X, const Vector2d& U);
};

/**
 * @brief 车辆实体对象，维护当前真实位姿、全局路径、局部计划与物理模型
 */
class Vehicle{
    private:
        State state;
        GlobalPlan global_plan;
        LocalPlan local_plan;
        SystemModel model;
    public:
        Vehicle();

        void set_state(double x,double y,double heading,double v){
            this->state << x,y,heading,v;
        };
        void set_state(const State& X){
            this->state << X;
        };
        Vector4d get_state() const{
            return this->state;
        };

        void set_global_plan(const GlobalPlan& global_plan){
            this->global_plan = global_plan;
        };
        GlobalPlan get_global_plan(){
            return this->global_plan;
        };
        void set_local_plan(){
            size_t num_points_to_extract = static_cast<size_t>(std::max<double>((state[3] * model.dt * model.N),100) + 300);
            this->local_plan.set_plan(this->global_plan,this->state,num_points_to_extract);
        };
        LocalPlan get_local_plan(){
            return this->local_plan;
        };
        void set_model(const SystemModel& model){
            this->model = model;
        };
        SystemModel get_model(){
            return this->model;
        };
};

/**
 * @brief 离散状态轨迹容器
 */
class Trajectory{
    public:
        std::vector<State> states;

        Trajectory() = default;
        Trajectory(const std::vector<State>& states) : states(states) {}
        std::vector<State> get_states() const{
            return this->states;
        };
        void set_states(const std::vector<State>& states){
            this->states = states;
        };
        void push_back(const State& state){
            this->states.push_back(state);
        };
        State back(){
            return this->states.back();
        };
};

/**
 * @brief 离散控制输入序列容器
 */
class ControlSequence {
    public:
        std::vector<Control> controls; 
        ControlSequence() = default;
        explicit ControlSequence(const std::vector<Control>& ctrls) : controls(ctrls) {}
        
        std::vector<Control> get_control_sequence() const { 
            return controls; 
        }
        void push_back(const Control& control) { 
            controls.push_back(control); 
        }
        Control& operator[](size_t i) {
            if (i >= controls.size()) 
                throw std::out_of_range("ControlSquence index out of range");
            return controls[i]; 
        }
        size_t size() const { return controls.size(); }
};

/**
 * @brief 求解器单周期完整输出解结构体
 */
struct Solution {
    Solution(){}
    Solution(Trajectory ego_trj,ControlSequence control_sequence){
        this->ego_trj = ego_trj;
        this->control_sequence = control_sequence;
    }
    Solution(const Solution& solution){
        this->ego_trj = solution.ego_trj;
        this->control_sequence = solution.control_sequence;
        this->converged = solution.converged;
        this->iterations = solution.iterations;
        this->final_cost = solution.final_cost;
        this->solve_time_ms = solution.solve_time_ms;
    }
    Trajectory ego_trj;               // 自车全时域预测状态轨迹序列 (N+1 步)
    ControlSequence control_sequence; // 自车优化控制量序列 (N 步)
    
    // 收敛与诊断信息
    bool converged = false;           // 求解器是否收敛标志
    int iterations = 0;               // 实际经历的内迭代总次数
    double final_cost = 0.0;          // 最终目标函数收敛代价值
    double solve_time_ms = 0.0;       // 纯求解器算法内部耗时 [单位: ms]
};

/**
 * @brief 基于控制障碍函数与松弛屏障惩罚的约束 iLQR 求解器 (CILQRSolver)
 */
class CILQRSolver{
    private:
        double lamb ;
        double average_gradient = 0.0;
        bool converged = false;
        Solution pre_solution;
        Vehicle ego;
        std::vector<Trajectory> obs_list;
        Arg arg;
        std::string solver_name;
        std::string map_name;
        
        // 日志记录相关
        std::string log_filename;
        std::ofstream cost_log_file;
        bool enable_logging = true;
        
        // iLQR 核心反向递推矩阵序列
        std::vector<MatrixXd> k;          // 开环前馈控制修正向量序列 k_k (2x1)
        std::vector<MatrixXd> K;          // 状态反馈局部增益矩阵序列 K_k (2x4)
        std::vector<MatrixXd> df_dx;      // 动力学状态雅可比 df/dx 序列 (4x4)
        std::vector<MatrixXd> df_du;      // 动力学控制雅可比 df/du 序列 (4x2)
        std::vector<MatrixXd> lx;         // 步阶段代价关于状态的一阶导数 (4x1)
        std::vector<MatrixXd> lu;         // 步阶段代价关于控制的一阶导数 (2x1)
        std::vector<MatrixXd> lxx;        // 步阶段代价关于状态的二阶 Hessian (4x4)
        std::vector<MatrixXd> luu;        // 步阶段代价关于控制的二阶 Hessian (2x2)
        std::vector<MatrixXd> lux;        // 步阶段代价关于控制与状态的交叉二阶导 (2x4)
        std::vector<MatrixXd> Qu;         // Q 动作价值函数关于控制量的一阶梯度 (2x1)
        std::vector<MatrixXd> Quu;        // Q 动作价值函数关于控制量的二阶 Hessian (2x2)

        Control pure_pursuit(const State& X_cur);
        Solution get_nominal_solution(const State& init_state);
        double cal_cost(const Solution& solution);
        double cal_cost_with_logging(const Solution& solution, int iteration);
        void compute_df(const Solution& solution);
        void compute_cost_derivatives(const Solution& solution);
        void backward();
        Solution forward(const Solution& solution);
        
        void init_cost_logging();
        void log_cost_breakdown(int iteration, double total_cost, 
                                  double J_position, double J_heading, double J_gamma_state, double J_lateral_ref,
                                  double J_velocity, double J_gamma_ctrl, 
                                  double J_obs, double J_lane, double J_gamma_barrier, double J_gamma_dot_barrier,
                                  double J_speed_rate, double lambda_value);
        void close_cost_logging();

    public:
        CILQRSolver(const Vehicle& ego, const std::vector<Trajectory>& obs_list, const Arg& arg, const std::string& map_name, const std::string& solver_name = "cilqr") 
        : ego(ego), obs_list(obs_list), arg(arg), map_name(map_name), solver_name(solver_name), lamb(arg.lamb_init),
        k(arg.N),
        K(arg.N),
        df_dx(arg.N),
        df_du(arg.N),
        lx(arg.N+1),
        lu(arg.N),
        lxx(arg.N+1),
        luu(arg.N),
        lux(arg.N),
        Qu(arg.N),
        Quu(arg.N){
            for(int i = 0; i < arg.N; ++i) {
                k[i] = MatrixXd::Zero(2,1);
                K[i] = MatrixXd::Zero(2,4);
                df_dx[i] = MatrixXd::Zero(4,4);
                df_du[i] = MatrixXd::Zero(4,2);
                lu[i] = MatrixXd::Zero(2,1);
                luu[i] = MatrixXd::Zero(2,2);
                lux[i] = MatrixXd::Zero(2,4);
                Qu[i] = MatrixXd::Zero(2,1);
                Quu[i] = MatrixXd::Zero(2,2);
            }
            for(int i = 0; i < arg.N+1; ++i) {
                lx[i] = MatrixXd::Zero(4,1);
                lxx[i] = MatrixXd::Zero(4,4);
            }
            init_cost_logging();
        }
        
        ~CILQRSolver() {
            close_cost_logging();
        }
        
        // 核心优化求解接口
        Solution solve(const State& init_state,const std::vector<Trajectory>& obs_list);
};

/**
 * @brief 基于增广拉格朗日乘子法 (Augmented Lagrangian) 的约束 iLQR 求解器 (ALILQRSolver)
 */
class ALILQRSolver{
    private:
        double lamb;
        double average_gradient = 0.0;
        bool converged = false;
        Solution pre_solution;
        Vehicle ego;
        std::vector<Trajectory> obs_list;
        Arg arg;
        
        std::vector<MatrixXd> k;
        std::vector<MatrixXd> K;
        std::vector<MatrixXd> df_dx;
        std::vector<MatrixXd> df_du;
        std::vector<MatrixXd> lx;
        std::vector<MatrixXd> lu;
        std::vector<MatrixXd> lxx;
        std::vector<MatrixXd> luu;
        std::vector<MatrixXd> lux;
        std::vector<MatrixXd> Qu;
        std::vector<MatrixXd> Quu;
        
        // ALTRO 增广拉格朗日外层参数
        int max_outer_iters = 20;         // 最大外层乘子迭代次数
        int max_inner_iters = 50;         // 每轮外层对应的内层最大 iLQR 迭代次数
        double constraint_tol = 1e-3;     // 约束违背量收敛容差
        double penalty_scaling = 10.0;    // 惩罚因子每轮放大倍率
        double penalty_max = 1e8;         // 惩罚因子数值上限
        
        // 拉格朗日乘子 (lambda) 与二次惩罚系数 (mu)
        // 状态约束 (N+1 步)
        std::vector<double> lambda_gamma_max;
        std::vector<double> mu_gamma_max;
        std::vector<double> lambda_gamma_min;
        std::vector<double> mu_gamma_min;
        
        std::vector<double> lambda_obs;
        std::vector<double> mu_obs;
        
        std::vector<double> lambda_lane;
        std::vector<double> mu_lane;
        
        // 控制约束 (N 步)
        std::vector<double> lambda_gdot_max;
        std::vector<double> mu_gdot_max;
        std::vector<double> lambda_gdot_min;
        std::vector<double> mu_gdot_min;

        Control pure_pursuit(const State& X_cur);
        Solution get_nominal_solution(const State& init_state);
        double cal_cost(const Solution& solution);
        double cal_cost_with_logging(const Solution& solution, int iteration);
        void compute_df(const Solution& solution);
        void compute_al_derivatives(const Solution& solution);
        void backward();
        Solution forward(const Solution& solution);
        
        void update_constraints(const Solution& solution);
        double max_constraint_violation(const Solution& solution);
        void initialize_penalties();
        void shift_penalties();

    public:
        ALILQRSolver(const Vehicle& ego, const std::vector<Trajectory>& obs_list, const Arg& arg)
        : ego(ego), obs_list(obs_list), arg(arg), lamb(arg.lamb_init),
        k(arg.N), K(arg.N), df_dx(arg.N), df_du(arg.N), lx(arg.N+1), lu(arg.N), lxx(arg.N+1), luu(arg.N), lux(arg.N), Qu(arg.N), Quu(arg.N),
        lambda_gamma_max(arg.N+1, 0.0), mu_gamma_max(arg.N+1, 1.0),
        lambda_gamma_min(arg.N+1, 0.0), mu_gamma_min(arg.N+1, 1.0),
        lambda_obs(arg.N+1, 0.0), mu_obs(arg.N+1, 1.0),
        lambda_lane(arg.N+1, 0.0), mu_lane(arg.N+1, 1.0),
        lambda_gdot_max(arg.N, 0.0), mu_gdot_max(arg.N, 1.0),
        lambda_gdot_min(arg.N, 0.0), mu_gdot_min(arg.N, 1.0)
        {
            for(int i = 0; i < arg.N; ++i) {
                k[i] = MatrixXd::Zero(2,1);
                K[i] = MatrixXd::Zero(2,4);
                df_dx[i] = MatrixXd::Zero(4,4);
                df_du[i] = MatrixXd::Zero(4,2);
                lu[i] = MatrixXd::Zero(2,1);
                luu[i] = MatrixXd::Zero(2,2);
                lux[i] = MatrixXd::Zero(2,4);
                Qu[i] = MatrixXd::Zero(2,1);
                Quu[i] = MatrixXd::Zero(2,2);
            }
            for(int i = 0; i < arg.N+1; ++i) {
                lx[i] = MatrixXd::Zero(4,1);
                lxx[i] = MatrixXd::Zero(4,4);
            }
        }
        Solution solve(const State& init_state,const std::vector<Trajectory>& obs_list);
};

#endif // ILQR_H
