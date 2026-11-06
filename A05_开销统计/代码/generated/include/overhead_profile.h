#pragma once
#include <chrono>
#include <vector>
#include <sstream>
#include <iomanip>

/**
 * @file overhead_profile.h
 * @brief 算法执行各计算环节的高精度毫秒级时间开销插桩与分析模块。
 * 
 * 专门用于细粒度剖析 iLQR / C-iLQR / AL-iLQR 各阶段的计算开销，包括:
 * - 模型线性化耗时 (linearization_ms): 离散状态转移方程雅可比矩阵 A_k, B_k 的计算;
 * - 代价导数计算耗时 (derivatives_ms): 目标函数与约束罚项的一阶和二阶导数 l_x, l_u, l_xx, l_uu, l_ux;
 * - 反向通道耗时 (backward_ms): 动态规划黎卡提方程反向递推求解局部控制增益 k, K;
 * - 前向通道耗时 (forward_ms): 非线性系统前向模拟滚推与线性搜索 (line search);
 * - 代价与残差评估耗时 (cost_ms): 轨迹总代价、障碍物残差与边界违例度计算;
 * - 增广拉格朗日更新耗时 (al_update_ms): 乘子向量与惩罚因子外迭代更新。
 */

namespace overhead {

using Clock = std::chrono::steady_clock;

/**
 * @brief 计算从指定时间戳到当前时刻所流逝的毫秒数 (ms)。
 * 
 * @param[in] t 起始时间戳。
 * @return double 经过的时间间隔，单位为毫秒 (ms)。
 */
inline double ms(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

/**
 * @brief 单次优化内迭代周期的细分耗时记录结构体。
 */
struct Iter {
    int outer = 0;          ///< 当前外迭代轮次索引 (AL-iLQR 乘子更新轮次，从 1 开始)
    int inner = 0;          ///< 当前内迭代轮次索引 (iLQR 反向/前向通道轮次，从 1 开始)
    double wall = 0;         ///< 本轮内迭代总耗时 [ms]
    double linearization = 0;///< 运动学模型雅可比矩阵计算耗时 [ms]
    double derivatives = 0;  ///< 代价函数一阶与二阶偏导数计算耗时 [ms]
    double backward = 0;     ///< 黎卡提反向通道递推耗时 [ms]
    double forward = 0;      ///< 前向模拟滚推与线性搜索耗时 [ms]
    double cost = 0;         ///< 轨迹代价与约束违例评估耗时 [ms]
};

/**
 * @brief 单次规划求解过程的累积耗时与迭代序列记录结构体。
 */
struct Profile {
    std::vector<Iter> iterations; ///< 各内迭代周期的耗时明细列表
    int outer = 0;                 ///< 外迭代总轮次计数
    int updates = 0;               ///< 乘子与惩罚因子更新次数
    double linearization = 0;      ///< 雅可比线性化总耗时 [ms]
    double derivatives = 0;        ///< 代价导数计算总耗时 [ms]
    double backward = 0;           ///< 反向通道总耗时 [ms]
    double forward = 0;            ///< 前向通道总耗时 [ms]
    double cost = 0;               ///< 代价评估总耗时 [ms]
    double al_update = 0;          ///< 增广拉格朗日外迭代更新总耗时 [ms]
};

/// 线程局部单例，保证多线程或独立求解时不发生状态竞争
extern thread_local Profile data;

/**
 * @brief 重置当前线程的开销统计数据，清空迭代列表并预分配存储空间。
 */
inline void reset() {
    data = Profile{};
    data.iterations.reserve(1000);
}

/**
 * @brief 基于 RAII 机制的作用域计时器，析构时自动将耗时累加至目标变量。
 */
struct Timer {
    double& total;                   ///< 目标累加变量引用 [ms]
    Clock::time_point start = Clock::now(); ///< 计时器创建时刻

    /**
     * @brief 构造函数，立即记录当前时钟戳。
     * @param[in,out] x 待累加的双精度毫秒耗时变量。
     */
    explicit Timer(double& x) : total(x) {}

    /**
     * @brief 析构函数，计算作用域生命周期耗时并累加至 total。
     */
    ~Timer() {
        total += ms(start);
    }
};

/**
 * @brief 基于 RAII 机制的单次内迭代周期范围计时器。
 * 
 * 记录本轮迭代开始与结束时的各项阶段累计值差额，生成 Iter 记录追加至 Profile。
 */
struct Iteration {
    Iter row;
    Clock::time_point start = Clock::now();
    double l, d, b, f, c;

    /**
     * @brief 构造函数，记录本轮迭代进入时的各环节初始累计耗时。
     * @param[in] o 外迭代轮次索引。
     * @param[in] i 内迭代轮次索引。
     */
    Iteration(int o, int i)
        : l(data.linearization), d(data.derivatives), b(data.backward), f(data.forward), c(data.cost) {
        row.outer = o;
        row.inner = i;
    }

    /**
     * @brief 析构函数，计算本轮迭代净耗时并推入全局 iterations 记录。
     */
    ~Iteration() {
        row.wall = ms(start);
        row.linearization = data.linearization - l;
        row.derivatives = data.derivatives - d;
        row.backward = data.backward - b;
        row.forward = data.forward - f;
        row.cost = data.cost - c;
        data.iterations.push_back(row);
    }
};

/**
 * @brief 将当前线程收集到的 Profile 性能统计数据序列化为标准 JSON 字符串。
 * 
 * @return std::string 包含总耗时与各轮迭代耗时明细的 JSON 文本。
 */
inline std::string json() {
    std::ostringstream s;
    s << std::setprecision(17);
    s << "{\"iterations\":" << data.iterations.size()
      << ",\"outer_iterations\":" << data.outer
      << ",\"al_updates\":" << data.updates
      << ",\"linearization_ms\":" << data.linearization
      << ",\"derivatives_ms\":" << data.derivatives
      << ",\"backward_ms\":" << data.backward
      << ",\"forward_ms\":" << data.forward
      << ",\"cost_ms\":" << data.cost
      << ",\"al_update_ms\":" << data.al_update
      << ",\"iteration_timings\":[";
    for (size_t i = 0; i < data.iterations.size(); ++i) {
        const auto& r = data.iterations[i];
        if (i) s << ',';
        s << "{\"outer\":" << r.outer
          << ",\"inner\":" << r.inner
          << ",\"wall_ms\":" << r.wall
          << ",\"linearization_ms\":" << r.linearization
          << ",\"derivatives_ms\":" << r.derivatives
          << ",\"backward_ms\":" << r.backward
          << ",\"forward_ms\":" << r.forward
          << ",\"cost_ms\":" << r.cost << '}';
    }
    s << "]}";
    return s.str();
}

} // namespace overhead
