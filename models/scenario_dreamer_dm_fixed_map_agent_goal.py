import glob
import os
import pickle

from torch_ema import ExponentialMovingAverage

from datasets.waymo.dataset_dm_fixed_map_agent_goal_waymo import WaymoDatasetDMFixedMapAgentGoal
from models.scenario_dreamer_dm import ScenarioDreamerDM
from nn_modules.dm_fixed_map_agent_goal import DMFixedMapAgentGoal


class ScenarioDreamerDMFixedMapAgentGoal(ScenarioDreamerDM):
    """Lightning wrapper for fixed-map agent init/goal diffusion."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.diff_model = DMFixedMapAgentGoal(self.cfg)
        self.ema = ExponentialMovingAverage(self.diff_model.parameters(), decay=self.cfg.train.ema_decay)

    def _initialize_pyg_dset(self, mode, num_samples, conditioning_path=None, nocturne_compatible_only=False):
        if mode != "lane_conditioned":
            return super()._initialize_pyg_dset(mode, num_samples, conditioning_path, nocturne_compatible_only)

        assert conditioning_path is not None
        conditioning_files = sorted(glob.glob(os.path.join(conditioning_path, "*.pkl")))[:num_samples]
        dset = WaymoDatasetDMFixedMapAgentGoal(self.cfg_dataset, split_name="val", mode="eval")
        data_list = []
        conditioning_filenames = []
        for i, conditioning_file in enumerate(conditioning_files):
            with open(conditioning_file, "rb") as f:
                cond_d = pickle.load(f)
            d = dset.get_data(cond_d, i, conditioning_file)
            if d is None:
                continue
            data_list.append(d)
            conditioning_filenames.append(os.path.splitext(os.path.basename(conditioning_file))[0])
        return data_list, conditioning_filenames
