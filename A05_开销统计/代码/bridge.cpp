#include "overhead_profile.h"
/**
 * ============================================================================
 * 文件名称: bridge.cpp
 * 文件功能: C++ 原生轨迹优化求解器与运动学模型的 Python C-API 外部交互桥接层
 *
 * 架构设计与技术要点:
 * 1. 严格无侵入封装:
 *    - 底层优化算法 (CILQRSolver, ALILQRSolver) 与系统动力学 (SystemModel) 直接编译自原生源文件；
 *    - 采用辅助结构体 OriginalBridgeAccess 访问保护成员，确保原生代码字节级无修改。
 * 2. 状态与控制空间定义:
 *    - 车辆状态向量 State \in \mathbb{R}^4: [x, y, \theta, \gamma]^T
 *      * x     : 前轴中心笛卡尔横坐标 [单位: m]
 *      * y     : 前轴中心笛卡尔纵坐标 [单位: m]
 *      * \theta : 前车体全局航向角 [单位: rad]
 *      * \gamma : 前后车体相对铰接角 [单位: rad]
 *    - 控制输入向量 Control \in \mathbb{R}^2: [v, \dot{\gamma}]^T
 *      * v            : 前向纵向车速 [单位: m/s]
 *      * \dot{\gamma} : 铰接角变化率 [单位: rad/s]
 * 3. 诊断残差与约束评估口径:
 *    - 铰接角极限: c_\gamma = [\gamma - \gamma_{max}, \gamma_{min} - \gamma]^T \le 0 [单位: rad]
 *    - 车道走廊限宽: c_{lane} = [e_{lat} - w_{left}, -e_{lat} - w_{right}]^T \le 0 [单位: m]
 *    - 椭圆动态避障: c_{obs} = 1 - (ox / A)^2 - (oy / B)^2 \le 0 [无量纲，<=0 表示位于安全包络外]
 *    - 铰接角速度界限: c_{\dot{\gamma}} = [\dot{\gamma} - \dot{\gamma}_{max}, \dot{\gamma}_{min} - \dot{\gamma}]^T \le 0 [单位: rad/s]
 * ============================================================================
 */
#include "ilqr.h"
#include "config_loader.h"
#include "utils.h"
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif

extern std::string original_bridge_log_root;
void finalize_parameters(Arg& arg);
static thread_local std::string last_error, last_json;

/**
 * @brief 求解器内部成员受控访问辅助结构体
 */
struct OriginalBridgeAccess {
    static void prepare(CILQRSolver& r, const State& init_state, const std::vector<Trajectory>& obs_list) {
#include "generated/r_prepare.inc"
    }
    static void prepare(ALILQRSolver& s, const State& x, const std::vector<Trajectory>& obs) {
        s.ego.set_state(x); s.obs_list=obs; s.ego.set_local_plan();
    }
    static void mute_evaluator_logging(CILQRSolver& s) { s.enable_logging=false; s.close_cost_logging(); }
    static const Arg& arg(CILQRSolver& s) { return s.arg; }
    static const Arg& arg(ALILQRSolver& s) { return s.arg; }
    static Vehicle& vehicle(CILQRSolver& s) { return s.ego; }
    static Vehicle& vehicle(ALILQRSolver& s) { return s.ego; }
    static double cost(CILQRSolver& s,const Solution& v) { return s.cal_cost_with_logging(v,-300); }
    static double cost(ALILQRSolver& s,const Solution& v) { return s.cal_cost(v); }
    static double violation(ALILQRSolver& s,const Solution& v) { return s.max_constraint_violation(v); }
    static Solution nominal(CILQRSolver& s,const State& x) { return s.get_nominal_solution(x); }
    static Solution nominal(ALILQRSolver& s,const State& x) { return s.get_nominal_solution(x); }
    static void commit(CILQRSolver& s,const Solution& v) { s.pre_solution=v; }
    static void commit(ALILQRSolver& s,const Solution& v) { s.pre_solution=v; }
    static void initialize(ALILQRSolver& s) { s.initialize_penalties(); }
    static double base(CILQRSolver& s,const Solution& v) {
        const Arg saved=s.arg;
        s.arg.if_cal_obs_cost=false; s.arg.if_cal_lane_cost=false;
        s.arg.if_cal_gamma_barrier=false; s.arg.if_cal_gamma_dot_barrier=false;
        double value;
        try { value=cost(s,v); } catch(...) { s.arg=saved; throw; }
        s.arg=saved; return value;
    }
    static double base(ALILQRSolver& s,const Solution& v) {
        std::vector<std::vector<double>*> arrays={&s.lambda_gamma_max,&s.mu_gamma_max,&s.lambda_gamma_min,&s.mu_gamma_min,
          &s.lambda_obs,&s.mu_obs,&s.lambda_lane,&s.mu_lane,&s.lambda_gdot_max,&s.mu_gdot_max,&s.lambda_gdot_min,&s.mu_gdot_min};
        std::vector<std::vector<double>> saved;
        for(auto* a:arrays) { saved.push_back(*a); std::fill(a->begin(),a->end(),0.); }
        double value;
        try { value=cost(s,v); } catch(...) { for(size_t i=0;i<arrays.size();++i)*arrays[i]=saved[i]; throw; }
        for(size_t i=0;i<arrays.size();++i)*arrays[i]=saved[i];
        return value;
    }
};

/**
 * @brief 运行上下文结构体，持有配置、运动学模型、求解器与障碍物状态
 */
struct Context {
    Arg arg; SystemModel model; Vehicle ego;
    std::unique_ptr<CILQRSolver> r;
    std::unique_ptr<ALILQRSolver> al;
    std::vector<Trajectory> obstacles;
    State state=State::Zero();
    int usage=0; // 0=未分配, 1=MPC评估器模式, 2=原生优化器求解模式 (独立上下文杜绝终端减速状态污染)
    int profile=0;
    bool prepared=false;
    double prepare_wall_ms=0.;
};

static void finite(const double* values,size_t n,const char* label) {
    if(!values && n) throw std::invalid_argument(std::string("null ")+label);
    for(size_t i=0;i<n;++i) if(!std::isfinite(values[i])) throw std::invalid_argument(std::string("nonfinite ")+label);
}
static Context& context(void* ptr) { if(!ptr)throw std::invalid_argument("null context");return *static_cast<Context*>(ptr); }
static State state_from(const double* data) { finite(data,4,"state");return State(data[0],data[1],data[2],data[3]); }
static std::vector<Trajectory> predictions(Context& c,const double* obs,int count) {
    if(count<0)throw std::invalid_argument("negative obstacle count");finite(obs,4*count,"obstacles");
    std::vector<Trajectory> out;
    for(int i=0;i<count;++i)out.push_back(predict_obstacle_trajectory(state_from(obs+4*i),c.arg.dt,c.arg.N));
    return out;
}
static Solution rollout(Context& c,const double* u) {
    finite(u,2*c.arg.N,"controls");
    Solution sol;sol.ego_trj.states.push_back(c.state);
    for(int i=0;i<c.arg.N;++i) {
        Control control(u[2*i],u[2*i+1]);sol.control_sequence.controls.push_back(control);
        sol.ego_trj.states.push_back(c.model.dynamics(sol.ego_trj.states.back(),control));
    }
    return sol;
}
static void write_solution(const Solution& sol,double* states,double* controls) {
    if(states)for(size_t i=0;i<sol.ego_trj.states.size();++i)for(int j=0;j<4;++j)states[4*i+j]=sol.ego_trj.states[i][j];
    if(controls)for(size_t i=0;i<sol.control_sequence.controls.size();++i)for(int j=0;j<2;++j)controls[2*i+j]=sol.control_sequence.controls[i][j];
}

static std::vector<double> residuals(Context& c,const Solution& sol,double* matched_refs=nullptr) {
    auto& vehicle=c.profile==1?OriginalBridgeAccess::vehicle(*c.r):OriginalBridgeAccess::vehicle(*c.al);
    const auto& a=c.profile==1?OriginalBridgeAccess::arg(*c.r):OriginalBridgeAccess::arg(*c.al);
    const auto path=vehicle.get_local_plan().get_points();
    std::vector<double> result;
    for(int i=0;i<=a.N;++i) {
        const auto& x=sol.ego_trj.states[i];
        size_t j=find_closest_point(path,x);if(j+1<path.size())++j;
        const auto& ref=path[j];
        if(matched_refs) {matched_refs[3*i]=ref.x;matched_refs[3*i+1]=ref.y;matched_refs[3*i+2]=ref.heading;}
        result.push_back(x[3]-a.gamma_max);result.push_back(a.gamma_min-x[3]);
        if(a.if_cal_lane_cost) {
            const double lateral=(x[0]-ref.x)*(-std::sin(ref.heading))+(x[1]-ref.y)*std::cos(ref.heading);
            result.push_back(lateral-a.trace_safe_width_left);result.push_back(-lateral-a.trace_safe_width_right);
        }
        if(a.if_cal_obs_cost) for(const auto& trajectory:c.obstacles) {
            if(i>=static_cast<int>(trajectory.states.size()))continue;
            const auto& o=trajectory.states[i];const double dx=x[0]-o[0],dy=x[1]-o[1];
            const double A=a.obs_length/2+c.model.ego_rad/2+a.safe_a_buffer;
            const double B=a.obs_width/2+c.model.ego_rad/2+a.safe_b_buffer;
            const double ox=std::cos(o[2])*dx+std::sin(o[2])*dy;
            const double oy=-std::sin(o[2])*dx+std::cos(o[2])*dy;
            result.push_back(1-std::pow(ox,2)/std::pow(A,2)-std::pow(oy,2)/std::pow(B,2));
        }
    }
    for(const auto& u:sol.control_sequence.controls) {result.push_back(u[1]-a.gamma_dot_max);result.push_back(a.gamma_dot_min-u[1]);}
    return result;
}

template<class Fn>static int call(Fn fn) {try {last_error.clear();return fn();}catch(const std::exception& e){last_error=e.what();return -1;}catch(...){last_error="unknown C++ exception";return -1;}}
API const char* ob_error() {return last_error.c_str();}
API void* ob_create(const char* config_dir,const double* path,int npath,const char* log_dir) {
    try {
        last_error.clear();if(npath<2)throw std::invalid_argument("path needs at least 2 points");finite(path,3*npath,"path");
        auto c=std::make_unique<Context>();HybridAStarParams h;RunConfig run;
        const std::string folder(config_dir);original_bridge_log_root=log_dir;
        for(const auto& name:{"main.json","ilqr.json","hybrid_astar.json"})if(!std::filesystem::is_regular_file(folder+"/"+name))throw std::invalid_argument("missing configuration");
        load_config(folder+"/main.json",folder+"/ilqr.json",folder+"/hybrid_astar.json",c->arg,c->model,h,run);
        finalize_parameters(c->arg);
        if(c->arg.N<2||c->arg.N>1000||c->arg.dt<=0)throw std::invalid_argument("unsupported horizon or dt");
        GlobalPlan plan;std::vector<Point> points;
        for(int i=0;i<npath;++i)points.push_back({path[3*i],path[3*i+1],path[3*i+2]});
        plan.set_plan(points);c->ego.set_global_plan(plan);c->ego.set_model(c->model);
        c->r=std::make_unique<CILQRSolver>(c->ego,c->obstacles,c->arg,"original_bridge","cilqr");
        c->al=std::make_unique<ALILQRSolver>(c->ego,c->obstacles,c->arg);
        OriginalBridgeAccess::initialize(*c->al);
        return c.release();
    }catch(const std::exception& e){last_error=e.what();return nullptr;}
}
API void ob_destroy(void* ptr) {delete static_cast<Context*>(ptr);}
API int ob_prepare(void* ptr,int profile,const double* state,const double* obs,int nobs) {return call([&]{
    auto& c=context(ptr);if(c.usage==2)throw std::logic_error("use a separate context for MPC evaluation and original solve");
    if(profile!=1&&profile!=2)throw std::invalid_argument("profile must be 1=R or 2=AL");
    c.usage=1;c.profile=profile;c.state=state_from(state);c.obstacles=predictions(c,obs,nobs);
    if(profile==1)OriginalBridgeAccess::prepare(*c.al,c.state,c.obstacles);
    else OriginalBridgeAccess::prepare(*c.r,c.state,c.obstacles);
    const auto started=std::chrono::steady_clock::now();
    if(profile==1)OriginalBridgeAccess::prepare(*c.r,c.state,c.obstacles);
    else OriginalBridgeAccess::prepare(*c.al,c.state,c.obstacles);
    c.prepare_wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    OriginalBridgeAccess::mute_evaluator_logging(*c.r);c.prepared=true;return 0;
});}
API int ob_step(void* ptr,const double* state,const double* control,double* out) {return call([&]{
    auto& c=context(ptr);finite(control,2,"control");auto x=c.model.dynamics(state_from(state),Control(control[0],control[1]));for(int i=0;i<4;++i)out[i]=x[i];return 0;
});}
API int ob_nominal(void* ptr,double* states,double* controls) {return call([&]{
    auto& c=context(ptr);if(!c.prepared)throw std::logic_error("prepare first");
    auto sol=c.profile==1?OriginalBridgeAccess::nominal(*c.r,c.state):OriginalBridgeAccess::nominal(*c.al,c.state);
    write_solution(sol,states,controls);return 0;
});}
API int ob_cost(void* ptr,const double* controls,int kind,double* cost) {return call([&]{
    auto& c=context(ptr);if(!c.prepared)throw std::logic_error("prepare first");auto sol=rollout(c,controls);
    if(kind==0) {*cost=OriginalBridgeAccess::cost(*c.r,sol);}
    else if(kind==1) {*cost=OriginalBridgeAccess::base(*c.r,sol);}
    else if(kind==2) {*cost=OriginalBridgeAccess::base(*c.al,sol);}
    else throw std::invalid_argument("cost kind must be 0 exact-R, 1 R-base, 2 AL-base");
    return 0;
});}
API int ob_evaluate(void* ptr,const double* controls,double* states,double* costs,double* residual,double* refs) {return call([&]{
    auto& c=context(ptr);if(!c.prepared)throw std::logic_error("prepare first");auto sol=rollout(c,controls);write_solution(sol,states,nullptr);
    costs[0]=OriginalBridgeAccess::cost(*c.r,sol);costs[1]=OriginalBridgeAccess::base(*c.r,sol);
    costs[2]=OriginalBridgeAccess::cost(*c.al,sol);costs[3]=OriginalBridgeAccess::base(*c.al,sol);
    costs[4]=OriginalBridgeAccess::violation(*c.al,sol);
    auto values=residuals(c,sol,refs);for(size_t i=0;i<values.size();++i)residual[i]=values[i];return static_cast<int>(values.size());
});}
API int ob_commit(void* ptr,const double* controls) {return call([&]{
    auto& c=context(ptr);if(!c.prepared)throw std::logic_error("prepare first");const auto sol=rollout(c,controls);
    if(c.profile==1)OriginalBridgeAccess::commit(*c.r,sol);else OriginalBridgeAccess::commit(*c.al,sol);return 0;
});}
API int ob_original_solve(void* ptr,int method,const double* state,const double* obs,int nobs,double* states,double* controls,double* fields) {return call([&]{
    auto& c=context(ptr);if(c.usage==1)throw std::logic_error("use a separate context for MPC evaluation and original solve");
    if(method!=1&&method!=2)throw std::invalid_argument("method must be 1=R or 2=AL");
    c.usage=2;c.profile=method;c.state=state_from(state);c.obstacles=predictions(c,obs,nobs);
    overhead::reset();
    const auto started=std::chrono::steady_clock::now();
    const auto sol=method==1?c.r->solve(c.state,c.obstacles):c.al->solve(c.state,c.obstacles);
    const double solver_wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    if(sol.control_sequence.size()!=static_cast<size_t>(c.arg.N)||sol.ego_trj.states.size()!=static_cast<size_t>(c.arg.N+1))throw std::runtime_error("incomplete original solution");
    write_solution(sol,states,controls);fields[0]=sol.converged;fields[1]=sol.iterations;fields[2]=sol.final_cost;fields[3]=sol.solve_time_ms;
    fields[4]=method==1?OriginalBridgeAccess::cost(*c.r,sol):OriginalBridgeAccess::cost(*c.al,sol);
    const auto diagnostic=residuals(c,sol);fields[5]=0.;for(double value:diagnostic)fields[5]=std::max(fields[5],value);
    fields[6]=solver_wall_ms;return 0;
});}
API const char* ob_metadata(void* ptr) {
    const int result=call([&]{auto& c=context(ptr);const auto& a=c.arg;
      std::ostringstream s;s<<std::setprecision(17)<<"{";
      auto number=[&](const char* key,double value){s<<'"'<<key<<"":"<<value<<',';};
      number("N",a.N);number("dt",a.dt);number("lf",c.model.lf);number("lr",c.model.lr);number("ego_rad",c.model.ego_rad);
      number("configured_speed",a.desire_speed);number("r_effective_speed",OriginalBridgeAccess::arg(*c.r).desire_speed);number("al_effective_speed",OriginalBridgeAccess::arg(*c.al).desire_speed);
      number("prepare_wall_ms",c.prepare_wall_ms);
      number("gamma_min",a.gamma_min);number("gamma_max",a.gamma_max);number("gamma_dot_min",a.gamma_dot_min);number("gamma_dot_max",a.gamma_dot_max);
      number("acc_min",a.acc_min);number("acc_max",a.acc_max);number("trace_safe_width_left",a.trace_safe_width_left);number("trace_safe_width_right",a.trace_safe_width_right);
      number("obs_length",a.obs_length);number("obs_width",a.obs_width);number("safe_a_buffer",a.safe_a_buffer);number("safe_b_buffer",a.safe_b_buffer);
      number("ref_weight",a.ref_weight);number("v_rate_weight",a.v_rate_weight);number("if_cal_speed_rate_cost",a.if_cal_speed_rate_cost);
      number("if_cal_lane_cost",a.if_cal_lane_cost);number("if_cal_obs_cost",a.if_cal_obs_cost);number("if_cal_gamma_barrier",a.if_cal_gamma_barrier);number("if_cal_gamma_dot_barrier",a.if_cal_gamma_dot_barrier);
      number("lane_q1",a.lane_q1);number("lane_q2",a.lane_q2);number("obs_q1",a.obs_q1);number("obs_q2",a.obs_q2);
      number("gamma_max_q1",a.gamma_max_q1);number("gamma_max_q2",a.gamma_max_q2);number("gamma_min_q1",a.gamma_min_q1);number("gamma_min_q2",a.gamma_min_q2);
      number("gamma_dot_max_q1",a.gamma_dot_max_q1);number("gamma_dot_max_q2",a.gamma_dot_max_q2);number("gamma_dot_min_q1",a.gamma_dot_min_q1);number("gamma_dot_min_q2",a.gamma_dot_min_q2);
      s<<""Q":[";for(int i=0;i<4;++i){if(i)s<<',';s<<'[';for(int j=0;j<4;++j){if(j)s<<',';s<<a.Q(i,j);}s<<']';}
      s<<"],"R":[";for(int i=0;i<2;++i){if(i)s<<',';s<<'[';for(int j=0;j<2;++j){if(j)s<<',';s<<a.R(i,j);}s<<']';}
      s<<"],"obstacle_prediction_count":"<<a.N<<","usage":"<<c.usage<<","profile":"<<c.profile<<"}";
      last_json=s.str();return 0;});return result<0?nullptr:last_json.c_str();
}

API const char* ob_timing_profile() { last_json=overhead::json(); return last_json.c_str(); }

namespace overhead { thread_local Profile data; }
