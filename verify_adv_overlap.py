"""Read-only check that the in-context adv decode fixes the green-on-ego overlap.

For each val scene we deterministically reproduce the ldm_adv GT pipeline (eval
mode -> adv = first non-ego after reorder) and measure, in METERS:

  ego        : ego (scene's first agent) decoded with the proper scene graph
  adv_fix    : adv from the FIXED `_decode_scene_and_adv` (adv re-inserted into
               the full agent set, decoded in context) -- what is now drawn green
  adv_full   : adv from an INDEPENDENT full-set decode (adv re-inserted at its
               original middle index) -- reference for the fix
  adv_gt/ego_gt : raw ground-truth agent_states from the latent cache

All decodes use the latent MEAN (mu, no reparam noise) so they are deterministic.

Checks:
  1. adv_fix ~= adv_full and ego(fix) ~= ego(full)  -> the batched edge rebuild in
     `_decode_scene_and_adv` matches a per-scene-collated full decode (and confirms
     the set decoder is permutation-equivariant: middle-insert vs tail-append agree).
  2. D_symptom = ||adv_fix - ego|| now tracks the real separation D_gt, instead of
     collapsing to ~0 as the old isolated `_decode_adv` did.
"""
import os
import glob
import pickle

import numpy as np
import torch
from hydra import initialize_config_dir, compose
from torch_geometric.loader.dataloader import Collater

from model_registry import collapse_cfg
from models.scenario_dreamer_ldm_adv import ScenarioDreamerLDMAdv
from utils.data_container import ScenarioDreamerData
from utils.data_helpers import reorder_indices
from utils.pyg_helpers import get_edge_index_bipartite, get_edge_index_complete_graph
from utils.train_helpers import set_latent_stats
from utils.torch_helpers import from_numpy

N_SCENES = 400
OVERLAP_THRESH_M = 3.0  # cars are ~2x5 m; centers within 3 m visually overlap
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model():
    cfg_dir = os.environ["CONFIG_PATH"]
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="config_ldm_adv_train")
    _, cfg_node, cfg_ae = collapse_cfg(cfg, "ldm_adv")
    cfg_node = set_latent_stats(cfg_node)
    model = ScenarioDreamerLDMAdv(cfg_node, cfg_ae).to(DEVICE).eval()
    return model, cfg_node


def reorder_scene(data):
    """Reproduce WaymoDatasetLDMAdv ordering: return reordered mu + raw GT states."""
    agent_mu, agent_log_var = data["agent_mu"], data["agent_log_var"]
    lane_mu, lane_log_var = data["lane_mu"], data["lane_log_var"]
    e_l2l, states = data["edge_index_lane_to_lane"], data["agent_states"]
    road_points, scene_type = data["road_points"], data["scene_type"]
    agent_mu_r, _, lane_mu_r, _, e_l2l_r, _, _ = reorder_indices(
        agent_mu, agent_log_var, lane_mu, lane_log_var, e_l2l,
        states, road_points, scene_type, dataset="waymo")
    states_r = reorder_indices(
        states, states.copy(), lane_mu, lane_log_var, e_l2l,
        states, road_points, scene_type, dataset="waymo")[0]
    return agent_mu_r, lane_mu_r, e_l2l_r, states_r


def make_full_data(agent_mu, lane_mu, e_l2l):
    """Independent full-set decode graph (proper complete a2a + bipartite l2a)."""
    n_a, n_l = agent_mu.shape[0], lane_mu.shape[0]
    d = ScenarioDreamerData()
    d["agent"].x = from_numpy(agent_mu.astype(np.float32))
    d["lane"].x = from_numpy(lane_mu.astype(np.float32))
    d["lane", "to", "lane"].edge_index = torch.from_numpy(np.asarray(e_l2l)).long()
    d["agent", "to", "agent"].edge_index = get_edge_index_complete_graph(n_a).long()
    d["lane", "to", "agent"].edge_index = get_edge_index_bipartite(n_l, n_a).long()
    return d


def make_split_data(agent_mu, adv_mu, lane_mu, e_l2l, cfg):
    """Split scene exactly as WaymoDatasetLDMAdv (ego+others / adv), normalized
    mu latents -- the input `_decode_scene_and_adv` consumes."""
    am, as_, lm, ls = (cfg.dataset.agent_latents_mean, cfg.dataset.agent_latents_std,
                       cfg.dataset.lane_latents_mean, cfg.dataset.lane_latents_std)
    d = ScenarioDreamerData()
    d["agent"].x = from_numpy(agent_mu.astype(np.float32))
    d["agent"].latents = from_numpy(((agent_mu - am) / as_).astype(np.float32))
    d["lane"].x = from_numpy(lane_mu.astype(np.float32))
    d["lane"].latents = from_numpy(((lane_mu - lm) / ls).astype(np.float32))
    d["adv"].x = from_numpy(adv_mu.astype(np.float32))
    d["adv"].latents = from_numpy(((adv_mu - am) / as_).astype(np.float32))
    d["lane", "to", "lane"].edge_index = torch.from_numpy(np.asarray(e_l2l)).long()
    return d


def first_index_per_scene(batch_vec, n):
    return [int((batch_vec == s).nonzero(as_tuple=True)[0][0]) for s in range(n)]


def main():
    torch.manual_seed(0)
    model, cfg = build_model()
    ae = model.autoencoder.model
    collate = Collater(dataset=None)

    files = sorted(glob.glob(os.path.join(cfg.dataset.dataset_path, "val", "*.pkl")))
    full_list, split_list, gt_ego, gt_adv = [], [], [], []
    used = 0
    for f in files:
        if used >= N_SCENES:
            break
        with open(f, "rb") as fh:
            data = pickle.load(fh)
        if data["agent_mu"].shape[0] < 2:
            continue
        agent_mu_r, lane_mu_r, e_l2l_r, states_r = reorder_scene(data)
        adv_idx = 1  # eval mode: first non-ego after reorder
        keep = np.ones(agent_mu_r.shape[0], dtype=bool)
        keep[adv_idx] = False

        full_list.append(make_full_data(agent_mu_r, lane_mu_r, e_l2l_r))
        split_list.append(make_split_data(agent_mu_r[keep], agent_mu_r[adv_idx:adv_idx + 1],
                                          lane_mu_r, e_l2l_r, cfg))
        gt_ego.append(states_r[0, :2])
        gt_adv.append(states_r[adv_idx, :2])
        used += 1

    print(f"scenes used: {used}  (device={DEVICE})\n")

    batch_full = collate(full_list).to(DEVICE)
    batch_split = collate(split_list).to(DEVICE)

    with torch.no_grad():
        # independent full-set decode (reference)
        full_states, full_lane, *_ = ae.forward_decoder(
            batch_full["agent"].x, batch_full["lane"].x, batch_full)
        full_states, _ = model._unnormalize_agent_like(full_states, full_lane)
        # the FIXED pipeline decode (real code path under test)
        ego_states, _, _, _, _, adv_fix, _ = model._decode_scene_and_adv(
            batch_split["agent"].latents, batch_split["lane"].latents,
            batch_split["adv"].latents, batch_split)

    full_states = full_states.cpu().numpy()
    ego_states = ego_states.cpu().numpy()
    adv_fix = adv_fix.cpu().numpy()

    off_full = first_index_per_scene(batch_full["agent"].batch.cpu(), used)
    off_split = first_index_per_scene(batch_split["agent"].batch.cpu(), used)

    ego_full = np.stack([full_states[off_full[s], :2] for s in range(used)])
    adv_full = np.stack([full_states[off_full[s] + 1, :2] for s in range(used)])  # adv_idx=1
    ego_fix = np.stack([ego_states[off_split[s], :2] for s in range(used)])
    adv_fix_xy = adv_fix[:, :2]

    # GT raw states are stored normalized -> unnormalize the same way (x,y use fov)
    gt_ego, gt_adv = np.stack(gt_ego), np.stack(gt_adv)
    pad = lambda xy: np.concatenate([xy, np.zeros((used, 7), np.float32)], 1).astype(np.float32)
    dl = torch.zeros((1, 1, 2), device=DEVICE)
    gt_ego = model._unnormalize_agent_like(torch.from_numpy(pad(gt_ego)).to(DEVICE), dl)[0].cpu().numpy()[:, :2]
    gt_adv = model._unnormalize_agent_like(torch.from_numpy(pad(gt_adv)).to(DEVICE), dl)[0].cpu().numpy()[:, :2]

    dist = lambda a, b: np.linalg.norm(a - b, axis=1)

    # --- check 1: fixed method matches the independent full-set decode ---
    err_adv = dist(adv_fix_xy, adv_full)
    err_ego = dist(ego_fix, ego_full)
    print("=== check 1: fixed `_decode_scene_and_adv` vs independent full decode (m) ===")
    print(f"  adv  max err={err_adv.max():.4f}  mean={err_adv.mean():.4f}")
    print(f"  ego  max err={err_ego.max():.4f}  mean={err_ego.mean():.4f}")
    print(f"  -> {'PASS' if max(err_adv.max(), err_ego.max()) < 1e-2 else 'FAIL'}"
          f" (permutation-equivariant; batched edge rebuild correct)\n")

    # --- check 2: overlap gone; green now tracks the real separation ---
    D_symptom = dist(adv_fix_xy, ego_fix)
    D_gt = dist(gt_adv, gt_ego)
    D_recon = dist(adv_fix_xy, gt_adv)

    def stats(name, d):
        print(f"  {name:22s} median={np.median(d):6.2f}  mean={d.mean():6.2f}  "
              f"p10={np.percentile(d,10):6.2f}")

    print("=== check 2: distances over all scenes (m) ===")
    stats("D_symptom adv-ego", D_symptom)
    stats("D_gt     advGT-egoGT", D_gt)
    stats("D_recon  adv-advGT", D_recon)
    overlap = (D_symptom < OVERLAP_THRESH_M)
    gt_close = (D_gt < OVERLAP_THRESH_M)
    print(f"\n  green within {OVERLAP_THRESH_M} m of ego: {overlap.sum()}/{used} "
          f"({100*overlap.mean():.1f}%)   [was 9.2% with the old isolated decode]")
    print(f"  real adv within {OVERLAP_THRESH_M} m of ego (GT): {gt_close.sum()}/{used} "
          f"({100*gt_close.mean():.1f}%)  <- overlaps should match these genuine cases")
    print(f"  of green-overlap scenes, real adv ALSO close in GT: "
          f"{100*(gt_close[overlap].mean() if overlap.any() else 1.0):.1f}%")


if __name__ == "__main__":
    main()
