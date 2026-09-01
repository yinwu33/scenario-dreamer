import unittest

import numpy as np

from sim.geometry import _corners, _sat_overlap
from sim.planners import PDMPlanner, build_planner
from sim.planners.pdm import _pairwise_overlap
from sim.world import SimConfig, SimScene


def _planner_cfg():
    return {
        "name": "pdm",
        "conditioning": None,
        "target_speed": 15.0,
        "min_gap": 1.0,
        "headway_time": 1.5,
        "max_accel": 2.0,
        "comfort_decel": 6.0,
        "accel_exponent": 4.0,
        "lateral_margin": 0.3,
        "lead_search_radius": 40.0,
        "lookahead_time": 0.9,
        "lookahead_min": 3.0,
        "lookahead_max": 12.0,
        "steer_preview_steps": 5,
        "route": {"spacing": 1.0, "max_depth": 12},
        "proposal": {
            "horizon_steps": 20,
            "ttc_steps": 10,
            "target_speeds": [8.0, 12.0, 15.0],
            "min_gaps": [1.0],
            "headway_times": [0.8, 1.5, 2.2],
            "progress_weight": 5.0,
            "ttc_weight": 5.0,
        },
    }


def _scene(obstacle_x=None):
    states = [[0.0, 0.0, 8.0, 1.0, 0.0, 4.5, 2.0, 40.0, 0.0]]
    if obstacle_x is not None:
        states.append([obstacle_x, 0.0, 0.0, 1.0, 0.0, 4.5, 2.0, obstacle_x, 0.0])
    lanes = np.array(
        [[[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0], [40.0, 0.0]]],
        dtype=np.float32,
    )
    sim = SimScene(
        np.asarray(states, dtype=np.float32),
        np.ones(len(states), dtype=np.int64),
        lanes,
        sim_cfg=SimConfig(
            dt=0.1,
            goal_radius=2.0,
            goal_speed=100.0,
            goal_behavior="continue",
            map_extent=100.0,
            max_controlled_agents=32,
            ego_crash_freeze=True,
        ),
    )
    sim.lane_graph = {
        "succ": np.empty((2, 0), dtype=np.int64),
        "lateral": np.empty((2, 0), dtype=np.int64),
    }
    return sim


class PDMPlannerTest(unittest.TestCase):
    def test_registry_builds_pdm(self):
        self.assertIsInstance(build_planner(_planner_cfg(), role="sut"), PDMPlanner)

    def test_clear_road_accelerates(self):
        planner = PDMPlanner(_planner_cfg(), role="sut")
        action = int(planner.plan([(_scene(), np.array([0]))])[0][0])
        self.assertGreater(action // 13, 3)

    def test_stopped_lead_causes_emergency_braking(self):
        planner = PDMPlanner(_planner_cfg(), role="sut")
        action = int(planner.plan([(_scene(obstacle_x=12.0), np.array([0]))])[0][0])
        self.assertEqual(action // 13, 0)

    def test_pairwise_sat_matches_scalar_sat(self):
        boxes_a = _corners([0.0, 10.0], [0.0, 0.0], [0.0, 0.2], [4.0, 4.0], [2.0, 2.0])
        boxes_b = _corners([1.0, 20.0], [0.0, 0.0], [0.1, 0.0], [4.0, 4.0], [2.0, 2.0])
        expected = np.stack([_sat_overlap(box, boxes_b) for box in boxes_a])
        np.testing.assert_array_equal(_pairwise_overlap(boxes_a, boxes_b), expected)


if __name__ == "__main__":
    unittest.main()
