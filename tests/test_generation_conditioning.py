import unittest

import torch
from torch_geometric.data import Batch, HeteroData

from critical_scene.ldm_adv_eval import set_generation_conditioning


def _scene(num_agents, adv_cond):
    data = HeteroData()
    data["agent"].x = torch.zeros(num_agents, 1)
    data["agent"].cond = torch.tensor([[2, 0, 0]] * num_agents)
    data["adv"].x = torch.zeros(1, 1)
    data["adv"].cond = torch.tensor([adv_cond])
    return data


class GenerationConditioningTest(unittest.TestCase):
    def setUp(self):
        self.batch = Batch.from_data_list([
            _scene(2, [0, 1, 1, 3]),
            _scene(3, [0, 1, 2, 3]),
        ])

    def test_ego_far_activates_only_ego_agent_condition(self):
        out = set_generation_conditioning(self.batch.clone(), "ego_far")

        self.assertEqual(out["agent"].cond_drop.tolist(), [0, 1, 0, 1, 1])
        self.assertEqual(out["agent"].cond[[0, 2]].tolist(), [[0, 1, 2], [0, 1, 2]])
        self.assertEqual(out["adv"].cond.tolist(), [[0, 1, 1, 3], [0, 1, 2, 3]])

    def test_all_null_drops_agents_and_nulls_adv_fields(self):
        out = set_generation_conditioning(self.batch.clone(), "all_null")

        self.assertEqual(out["agent"].cond_drop.tolist(), [1, 1, 1, 1, 1])
        self.assertEqual(out["adv"].cond.tolist(), [[3, 2, 3, 3], [3, 2, 3, 3]])


if __name__ == "__main__":
    unittest.main()
