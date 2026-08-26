import unittest

import numpy as np

from ddpo.reward import HierarchicalRewardConfig, build_reward


def _cfg(**overrides):
    values = dict(
        ttc_tau=3.0,
        lane_soft=1.75,
        lane_hard=2.75,
        lane_penalty=0.1,
        collision_warmup=0.75,
        collision_window=0.5,
        trivial_collision_t=0.75,
        invalid_grade_scale=10.0,
        tier0_lo=-1.0,
        tier0_hi=-0.82,
        h_fallback_lo=-0.75,
        h_fallback_hi=-0.45,
        h_park_lo=-0.44,
        h_park_hi=-0.40,
        h_close_lo=-0.35,
        h_close_hi=-0.05,
        h_ttc_lo=0.05,
        h_ttc_hi=0.65,
        h_collision_lo=0.75,
        h_collision_hi=1.35,
        h_ego_fault_extra=0.40,
        h_close_dist=8.0,
        h_close_floor=2.0,
        h_split_close=False,
        h_close_delta=1.0,
        h_close_gate=False,
        h_close_static_floor=0.30,
        h_trivial_collision_floor=False,
        h_fallback_mode="spawn",
        h_init_near=8.0,
        h_init_far=30.0,
        h_fb_dmin_near=0.0,
        h_fb_dmin_far=45.0,
        h_fb_clr_near=0.0,
        h_fb_clr_far=40.0,
    )
    values.update(overrides)
    return HierarchicalRewardConfig(**values)


def _scorer(cfg):
    return build_reward(cfg, True)


def _metrics():
    # invalid, skipped fallback, admitted fallback, close, TTC, collision,
    # ego-fault collision
    n = 7
    return {
        "path_conflict": np.array([1, 0, 1, 1, 1, 1, 1], dtype=np.float32),
        "path_conflict_dist": np.array([1, 20, 5, 5, 5, 5, 5], dtype=np.float32),
        "init_overlap_frac": np.array([0.2, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "gen_agent_is_parked": np.zeros(n, dtype=np.float32),
        "gen_agent_is_invalid": np.zeros(n, dtype=np.float32),
        "gen_agent_invalid_reason": np.full(n, "", dtype=object),
        "gen_agent_invalid_gap": np.zeros(n, dtype=np.float32),
        "ego_min_ttc": np.array([np.inf, np.inf, np.inf, np.inf, 1.0, 0.2, 0.2]),
        "ego_adv_min_dist_warmup": np.array(
            [np.inf, np.inf, 12.0, 4.0, 4.0, 1.0, 1.0], dtype=np.float32
        ),
        "ego_collision": np.array([0, 0, 0, 0, 0, 1, 1], dtype=np.float32),
        "ego_collision_time": np.array(
            [np.inf, np.inf, np.inf, np.inf, np.inf, 2.0, 2.0], dtype=np.float32
        ),
        "ego_fault_collision": np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
        "spawn_lane_dist": np.zeros(n, dtype=np.float32),
        "goal_lane_dist": np.zeros(n, dtype=np.float32),
        "ego_adv_spawn_dist": np.array([5, 10, 20, 20, 20, 20, 20], dtype=np.float32),
        # Only the v2 knobs read this; index 3 (the close sample) closed in by
        # 16 m, so it counts as attacking rather than parked.
        "ego_adv_init_dist": np.array(
            [np.inf, np.inf, 12.0, 20.0, 20.0, 20.0, 20.0], dtype=np.float32
        ),
    }


def _cfg_v2(**overrides):
    values = dict(
        h_fallback_mode="measured",
        h_fb_dmin_near=0.0,
        h_fb_dmin_far=45.0,
        h_fb_clr_near=0.0,
        h_fb_clr_far=40.0,
        h_split_close=True,
        h_close_delta=1.0,
        h_park_lo=-0.40,
        h_park_hi=-0.25,
        h_close_gate=True,
        h_close_static_floor=0.30,
        h_trivial_collision_floor=True,
        h_fallback_lo=-0.75,
        h_fallback_hi=-0.50,
        h_close_lo=-0.15,
        h_close_hi=0.0,
        h_ttc_lo=0.10,
        h_ttc_hi=0.70,
        h_collision_lo=0.80,
        h_collision_hi=1.40,
    )
    values.update(overrides)
    return _cfg(**values)


class HierarchicalRewardTest(unittest.TestCase):
    def test_strict_fallback_order(self):
        reward, comp = _scorer(_cfg()).assemble(_metrics())

        np.testing.assert_array_equal(comp["tier"], [0, 1, 1, 2, 3, 4, 4])
        self.assertTrue(
            reward[0] < reward[1] < reward[3] < reward[4] < reward[5] < reward[6]
        )
        self.assertGreater(reward[1], reward[2])

    def test_prefilter_admission_does_not_earn_a_higher_tier(self):
        m = _metrics()
        m["ego_adv_spawn_dist"][1:3] = 15.0
        reward, comp = _scorer(_cfg()).assemble(m)

        self.assertEqual(comp["tier"][1], 1)
        self.assertEqual(comp["tier"][2], 1)
        self.assertAlmostEqual(float(reward[1]), float(reward[2]))

    def test_invalid_band_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "strictly ordered"):
            _cfg(h_fallback_lo=-0.9)


class HierarchicalV2Test(unittest.TestCase):
    def test_six_levels_strictly_ordered(self):
        reward, comp = _scorer(_cfg_v2()).assemble(_metrics())

        # invalid, skipped, admitted-quiet, closing, TTC, collision, ego-fault
        np.testing.assert_array_equal(comp["tier"], [0, 1, 1, 3, 4, 5, 5])
        self.assertTrue(
            reward[0] < reward[1] < reward[3] < reward[4] < reward[5] < reward[6]
        )

    def test_parked_adversary_ranks_below_one_that_closed_in(self):
        m = _metrics()
        # same 4 m final distance, but this one was already there at spawn
        m["ego_adv_init_dist"][3] = 4.5
        reward, comp = _scorer(_cfg_v2()).assemble(m)

        self.assertEqual(comp["tier"][3], 2)  # parked, not closing
        closing, _ = _scorer(_cfg_v2()).assemble(_metrics())
        self.assertLess(reward[3], closing[3])
        self.assertGreater(reward[3], reward[1])  # still above the fallback band

    def test_fallback_ranks_measured_above_skipped(self):
        # index 1 was skipped (no path conflict), index 2 was rolled out and
        # measured a finite dmin: the measured one must rank higher.
        reward, comp = _scorer(_cfg_v2()).assemble(_metrics())

        self.assertEqual(comp["tier"][1], 1)
        self.assertEqual(comp["tier"][2], 1)
        self.assertGreater(reward[2], reward[1])

    def test_pre_warmup_collision_is_not_dropped_into_the_fallback_band(self):
        m = _metrics()
        # contact at 0.2 s, then the cars separate again -> large post-warmup
        # dmin. Without the floor this lands in the fallback band.
        m["ego_collision_time"][5] = 0.2
        m["ego_adv_min_dist_warmup"][5] = 25.0
        m["ego_min_ttc"][5] = np.inf
        reward, comp = _scorer(_cfg_v2()).assemble(m)

        self.assertEqual(comp["tier"][5], 2)
        self.assertGreater(reward[5], reward[1])

    def test_park_band_must_be_ordered_too(self):
        with self.assertRaisesRegex(ValueError, "parked"):
            _cfg_v2(h_park_lo=-0.9)


if __name__ == "__main__":
    unittest.main()
