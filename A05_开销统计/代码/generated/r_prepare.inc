
    // 注入当前仿真环境状态: 更新自车状态、动态障碍物列表及局部路径初始解
    r.ego.set_state(init_state);
    r.obs_list = obs_list;
    r.ego.set_local_plan();

    // 终点接近区自适应平滑降速逻辑:
    // 计算车辆当前位置 (cur_state[0], cur_state[1]) 与全局路径目标点 (goal_pt.x, goal_pt.y) 之间的直线欧氏距离 [m]
    {
        const auto& global_points = r.ego.get_global_plan().get_points();
        if (!global_points.empty()) {
            const Point& goal_pt = global_points.back();
            const Vector4d cur_state = r.ego.get_state();
            double dx = cur_state[0] - goal_pt.x;
            double dy = cur_state[1] - goal_pt.y;
            double dist_to_goal = std::sqrt(dx*dx + dy*dy);
            // 当距离目标点小于 10.0 m 时，期望巡航速度按距离比例线性衰减，平稳停车
            if (dist_to_goal < 10.0) {
                r.arg.desire_speed = r.arg.desire_speed * dist_to_goal / 10.0;
                if (r.arg.desire_speed < 0.1) r.arg.desire_speed = 0;
            }
            // 终点附近保持期望航向角偏差为 0 rad
            r.arg.desire_heading = 0;
        }
    }
