"""时空图 (s, t, v) 速度搜索算法的独立单元测试与边界工况验证套件。

包含测试用例:
- 加减速可行性与加速度限幅边界测试；
- 遇到移动障碍物阻挡时的零速原位停车等待能力测试；
- 静态死锁与不可行环境的优雅退出测试；
- 终点碰撞风险预判与提前减速停车测试。
"""
import math
import unittest
import numpy as np
from geometric_speed_limits import build_geometric_speed_profile
from st_speed_search import STSettings, obstacle_residual, plan_speed, sample_path, sample_speed_plan

class STSpeedSearchTests(unittest.TestCase):
    """包含 ST 速度搜索核心逻辑、避障策略与动力学约束的单元测试类。"""

    def setUp(self):
        self.path = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        self.meta = {'obs_length': 2.0, 'obs_width': 2.0, 'ego_rad': 0.0, 'safe_a_buffer': 0.0, 'safe_b_buffer': 0.0}
        self.settings = STSettings(v_max=2.0, a_min=-2.0, a_max=2.0, max_time=20.0)

    def assert_valid(self, plan, obstacles, settings=None):
        settings = settings or self.settings
        self.assertTrue(plan['success'], plan['reason'])
        t = np.arange(0.0, settings.max_time + 0.0001, 0.01)
        q = sample_speed_plan(plan, t)
        residual = obstacle_residual(sample_path(self.path, q['s']), t, obstacles, self.meta)
        self.assertLessEqual(float(np.max(residual)), 1e-10)
        np.testing.assert_allclose(np.diff(plan['s']), 0.5 * (plan['speeds'][:-1] + plan['speeds'][1:]) * np.diff(plan['times']))
        acceleration = np.diff(plan['speeds']) / np.diff(plan['times'])
        self.assertLessEqual(float(np.max(acceleration)), settings.a_max + 1e-10)
        self.assertGreaterEqual(float(np.min(acceleration)), settings.a_min - 1e-10)
        return (t, q)

    def test_unobstructed_stopped_arrival(self):
        plan = plan_speed(self.path, [], self.meta, self.settings)
        self.assert_valid(plan, [])
        self.assertTrue(plan['reached_goal'])
        self.assertEqual(plan['speeds'][-1], 0.0)
        self.assertEqual(plan['checked_until_s'], 20.0)

    def test_wait_then_pass_moving_obstacle(self):
        obstacles = [[1.0, 0.0, math.pi / 2, 0.2]]
        plan = plan_speed(self.path, obstacles, self.meta, self.settings)
        self.assert_valid(plan, obstacles)
        self.assertTrue(np.any((plan['speeds'][:-1] == 0.0) & (plan['speeds'][1:] == 0.0)))
        self.assertTrue(plan['reached_goal'])

    def test_static_block_distinguishes_goal_failure_and_safe_partial(self):
        obstacles = [[5.0, 0.0, 0.0, 0.0]]
        complete = plan_speed(self.path, obstacles, self.meta, self.settings)
        self.assertFalse(complete['success'])
        partial = plan_speed(self.path, obstacles, self.meta, self.settings, allow_partial=True)
        self.assert_valid(partial, obstacles)
        self.assertFalse(partial['reached_goal'])
        self.assertTrue(partial['optimal_progress_proven'])
        self.assertLessEqual(partial['s'][-1], 4.0)

    def test_goal_future_collision_requires_later_arrival(self):
        obstacles = [[10.0, 5.0, -math.pi / 2, 0.5]]
        plan = plan_speed(self.path, obstacles, self.meta, self.settings)
        self.assert_valid(plan, obstacles)
        self.assertGreaterEqual(plan['arrival_time_s'], 12.0)

    def test_partial_window_keeps_actual_terminal_speed(self):
        settings = STSettings(v_max=2.0, a_min=-2.0, a_max=2.0, max_time=2.0)
        plan = plan_speed(self.path, [], self.meta, settings, allow_partial=True)
        self.assert_valid(plan, [], settings)
        self.assertFalse(plan['reached_goal'])
        self.assertEqual(plan['times'][-1], 2.0)
        self.assertGreater(plan['speeds'][-1], 0.0)
        self.assertEqual(float(sample_speed_plan(plan, [2.0])['speed'][0]), plan['speeds'][-1])
        with self.assertRaises(ValueError):
            sample_speed_plan(plan, [2.01])

    def test_narrow_speed_limit_is_not_skipped_between_time_samples(self):
        limits = np.array([[0.0, 2.0], [4.999, 2.0], [5.0, 0.2], [5.001, 2.0], [10.0, 2.0]])
        plan = plan_speed(self.path, [], self.meta, self.settings, speed_limits=limits, allow_partial=True)
        _, q = self.assert_valid(plan, [])
        cap = np.interp(q['s'], limits[:, 0], limits[:, 1])
        self.assertLessEqual(float(np.max(q['speed'] - cap)), 1e-10)

    def test_geometric_caps_use_floating_point_even_for_integer_speed_config(self):
        metadata = dict(self.meta, lf=3, lr=3, gamma_min=-1, gamma_max=1, gamma_dot_min=-0.15684989808486893, gamma_dot_max=0.15684989808486893, configured_speed=5)
        curved = np.array([[0.0, 0.0, 0.0], [2.0, 0.1, 0.2], [4.0, 0.5, 0.3]])
        profile = build_geometric_speed_profile(curved, metadata)
        self.assertTrue(np.all(profile['speed_limit'] > 0.0))
        self.assertTrue(np.any(profile['speed_limit'] < 5.0))
        self.assertTrue(np.any(profile['speed_limit'] % 1.0 != 0.0))
        bounds = profile['gamma_prime'] * profile['speed_limit']
        self.assertLessEqual(float(np.max(bounds)), metadata['gamma_dot_max'] + 1e-10)
        self.assertGreaterEqual(float(np.min(bounds)), metadata['gamma_dot_min'] - 1e-10)
if __name__ == '__main__':
    unittest.main()