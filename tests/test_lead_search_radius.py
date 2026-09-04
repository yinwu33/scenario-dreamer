"""The lead-vehicle broad phase must never hide a car the agent could stop for.

``lead_search_radius`` alone is a fixed distance, so above
``sqrt(2 * MAX_TABLE_DECEL * radius)`` (~17.9 m/s at the shipped 40 m) a lead
that a full-table brake still clears was filtered out before IDM saw it, and the
agent drove into it at undiminished speed. ``IDMPlanner._search_radius`` grows
the gate with ``v^2 / 2a``; PDM applies the same rule per proposal.
"""

import unittest

import numpy as np

from sim.geometry import _corners, _sat_overlap
from sim.planners import IDMPlanner
from sim.planners.idm import MAX_TABLE_DECEL
from sim.world import SimConfig, SimScene


def _cfg():
    return {
        "name": "idm",
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
    }


def _scene(v_ego, lead_x):
    states = [
        [0.0, 0.0, v_ego, 1.0, 0.0, 4.5, 2.0, 400.0, 0.0],
        [lead_x, 0.0, 0.0, 1.0, 0.0, 4.5, 2.0, lead_x, 0.0],
    ]
    lanes = np.array([[[float(x), 0.0] for x in range(0, 420, 10)]], dtype=np.float32)
    sim = SimScene(
        np.asarray(states, dtype=np.float32),
        np.ones(2, dtype=np.int64),
        lanes,
        sim_cfg=SimConfig(
            dt=0.1,
            goal_radius=2.0,
            goal_speed=100.0,
            goal_behavior="continue",
            map_extent=1000.0,
            max_controlled_agents=32,
            ego_crash_freeze=True,
        ),
    )
    sim.lane_graph = {
        "succ": np.empty((2, 0), dtype=np.int64),
        "lateral": np.empty((2, 0), dtype=np.int64),
    }
    return sim


class LeadSearchRadiusTest(unittest.TestCase):
    def test_radius_covers_the_braking_distance(self):
        planner = IDMPlanner(_cfg(), role="sut")
        sim = _scene(25.0, 150.0)
        # Below the configured radius the free-flow value wins; above it the
        # gate must reach at least as far as a full-table brake needs.
        self.assertEqual(planner._search_radius(sim, 0, 0.0), 40.0)
        for speed in (20.0, 25.0, 30.0):
            self.assertGreaterEqual(
                planner._search_radius(sim, 0, speed),
                speed * speed / (2.0 * MAX_TABLE_DECEL),
            )

    def test_fast_ego_stops_for_a_lead_beyond_the_configured_radius(self):
        # 22 m/s needs 60.5 m to stop, so the lead sits outside the 40 m gate
        # but well inside what the action table can still brake for.
        planner = IDMPlanner(_cfg(), role="sut")
        sim = _scene(22.0, 70.0)
        for _ in range(250):
            action = planner.plan([(sim, np.array([0]))])[0]
            sim.step_dynamics(np.asarray(action, dtype=np.int64), indices=np.array([0]))
            ego = _corners(
                sim.x[:1], sim.y[:1], sim.heading[:1], sim.length[:1], sim.width[:1]
            )
            lead = _corners(
                sim.x[1:], sim.y[1:], sim.heading[1:], sim.length[1:], sim.width[1:]
            )
            self.assertFalse(_sat_overlap(ego[0], lead).any(), "ego drove into the lead")


if __name__ == "__main__":
    unittest.main()
