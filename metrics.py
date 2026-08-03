import os 
from tqdm import tqdm 
import pickle 
import gzip
from utils.metrics_helpers import (
    convert_data_to_unified_format,
    compute_lane_metrics,
    compute_agent_metrics,
    compute_goal_metrics,
    has_goals,
)
from utils.goal_runtime import prepare_scene

class Metrics():

    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = cfg.dataset_name
        self.samples_path = cfg.eval.metrics.samples_path
        self.eval_set = cfg.eval.metrics.eval_set
        self.gt_format = cfg.eval.metrics.get('gt_format', 'standard')

    def _prepare_waymo_goal_gt(self, data):
        """Convert v2 goal records to the regular metric schema.

        Runs the *same* ``utils.goal_runtime.prepare_scene`` the goal dataset runs, so
        the reference agent set is identical to the one the goal autoencoder / LDM-Adv
        were trained on -- including any runtime off-road tightening or goal filter.
        The offline filters (closest-N cap, off-road vehicle removal) are already baked
        into the record. Goal columns are always appended, giving the same
        9-dimensional layout the goal-generating models produce.
        """
        scene = prepare_scene(data, self.cfg.dataset)

        data = dict(data)
        data['agent_states'] = scene['agent_states']
        data['agent_types'] = scene['agent_types']
        return data


    def compute_metrics(self):
        """Compute metrics given the generated samples and the ground truth samples."""
        sample_paths = [
            os.path.join(self.samples_path, file)
            for file in sorted(os.listdir(self.samples_path))
            if file.endswith('.pkl')
        ]

        with open(self.eval_set, 'rb') as f:
            gt_sample_filenames = pickle.load(f)['files']

        # Every metric compares the generated POOL against the real POOL (JSD /
        # Frechet), scene by scene into two independent accumulators -- nothing is
        # matched one-to-one. So the two counts are independent knobs: generate as
        # many scenes as you like (10k is plenty for stable JSDs) and keep the real
        # reference set as large as available for a low-variance reference.
        #   metrics.num_samples    -> cap on generated scenes (default: all cached)
        #   metrics.num_gt_samples -> cap on real scenes      (default: the whole eval set)
        max_samples = self.cfg.eval.metrics.get('num_samples')
        if max_samples is not None:
            sample_paths = sample_paths[:int(max_samples)]
        max_gt_samples = self.cfg.eval.metrics.get('num_gt_samples')
        if max_gt_samples is not None:
            gt_sample_filenames = gt_sample_filenames[:int(max_gt_samples)]

        if self.cfg.dataset_name == 'nuplan':
            gt_sample_ids = [os.path.splitext(file)[0] for file in gt_sample_filenames]

        num_samples = len(sample_paths)
        num_gt_samples = len(gt_sample_filenames)
        if num_samples == 0 or num_gt_samples == 0:
            raise ValueError(
                f"Nothing to compare: {num_samples} generated samples in "
                f"{self.samples_path}, {num_gt_samples} ground-truth samples."
            )

        print(f"Number of evaluated samples: {num_samples} generated / {num_gt_samples} real")
        samples = []
        gt_samples = []
        print("Converting generated samples to unified format for metrics computation...")
        for i in tqdm(range(num_samples)):
            with open(sample_paths[i], 'rb') as f:
                data = pickle.load(f)
            sample = convert_data_to_unified_format(data, dataset_name=f"{self.cfg.dataset_name}")
            if len(sample['G']) > 0:
                samples.append(sample)

        print("Converting ground-truth samples to unified format for metrics computation...")
        for i in tqdm(range(num_gt_samples)):
            if self.cfg.dataset_name == 'waymo':
                # agent and lane gt data are loaded from the preprocessed scenario dreamer waymo data
                with open(os.path.join(self.cfg.eval.metrics.gt_test_dir, gt_sample_filenames[i]), 'rb') as f:
                    gt_data = pickle.load(f)
                if self.gt_format == 'goal':
                    gt_data = self._prepare_waymo_goal_gt(gt_data)
            else:
                # the gt agent data comes from the preprocessed scenario dreamer nuplan data
                sample_id = gt_sample_ids[i]
                with open(os.path.join(self.cfg.eval.metrics.gt_agent_test_dir, f'{sample_id}_0.pkl'), 'rb') as f:
                    gt_agent_data = pickle.load(f)

                # As the lane graph is preprocessed slightly differently between SLEDGE and scenario dreamer,
                # for fairest comparison with SLEDGE we process the gt lane graphs following the SLEDGE preprocessing scheme (this requires
                # loading from the SLEDGE preprocessed nuplan data)
                # We could preprocess the gt lane graphs using the scenario dreamer preprocessing scheme,
                # but then we wouldn't know if performance improvement compared to SLEDGE is attributed to the GT lane graph preprocessing
                # being more aligned with scenario dreamer.
                # In practice, we find both preprocessing schemes yield very similar performance.
                with gzip.open(os.path.join(self.cfg.eval.metrics.gt_lane_test_dir, gt_sample_filenames[i]), 'rb') as f:
                    gt_lane_data = pickle.load(f)

                gt_data = gt_lane_data
                # add agent data to the gt lane data
                gt_data['agent_states'] = gt_agent_data['agent_states']
                gt_data['agent_types'] = gt_agent_data['agent_types']
                gt_data['lg_type'] = gt_agent_data['lg_type']

            gt_samples.append(convert_data_to_unified_format(gt_data, dataset_name=f'{self.cfg.dataset_name}_gt'))

        print(f"Usable scenes after filtering: {len(samples)} generated / {len(gt_samples)} real")
        if not samples or not gt_samples:
            raise ValueError(
                "Nothing usable after conversion: "
                f"{len(samples)} generated / {len(gt_samples)} real scenes."
            )

        generated_goal_flags = [has_goals(sample['vehicles']) for sample in samples]
        gt_goal_flags = [has_goals(sample['vehicles']) for sample in gt_samples]
        if any(generated_goal_flags) and not all(generated_goal_flags):
            raise ValueError("Generated samples mix 7D and 9D vehicle formats.")
        if any(gt_goal_flags) and not all(gt_goal_flags):
            raise ValueError("Ground-truth samples mix 7D and 9D vehicle formats.")
        generated_has_goals = all(generated_goal_flags)
        gt_has_goals = all(gt_goal_flags)
        if generated_has_goals and not gt_has_goals:
            raise ValueError(
                "Generated samples contain goals, but ground truth does not; "
                "use gt_format='goal'."
            )

        # The lane / agent metrics only read the first 7 unified-format columns, so
        # they are goal-agnostic: these are the numbers directly comparable to the
        # baseline LDM (the "w/o goal" table).
        lane_metrics = compute_lane_metrics(samples=samples, gt_samples=gt_samples)
        agent_metrics = compute_agent_metrics(samples=samples, gt_samples=gt_samples)
        # The goal metrics use the extra [goal_x, goal_y] columns and are only
        # defined for the goal pipeline (the "w goal" table). They are computed
        # automatically whenever both generated and real samples carry goals.
        goal_metrics = (compute_goal_metrics(samples=samples, gt_samples=gt_samples)
                        if generated_has_goals and gt_has_goals else None)

        print("--------------------------------------------------------------------------")
        print("[w/o goal] Lane metrics: ", ["{}: {:.2f}".format(k,v) for (k,v) in lane_metrics.items()])
        print("[w/o goal] Agent metrics: ", ["{}: {:.2f}".format(k,v) for (k,v) in agent_metrics.items()])
        if goal_metrics is not None:
            print("[w goal]   Goal metrics: ", ["{}: {:.2f}".format(k,v) for (k,v) in goal_metrics.items()])
        print("--------------------------------------------------------------------------")

        metrics = {
            'lane_metrics': lane_metrics,
            'agent_metrics': agent_metrics
        }
        if goal_metrics is not None:
            metrics['goal_metrics'] = goal_metrics
        # save metrics to file
        os.makedirs(self.cfg.eval.metrics.metrics_save_path, exist_ok=True)
        metrics_filename = self.cfg.eval.metrics.get('metrics_filename', 'metrics.pkl')
        metrics_path = os.path.join(self.cfg.eval.metrics.metrics_save_path, metrics_filename)
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)
        print(f"Metrics saved to {metrics_path}")
