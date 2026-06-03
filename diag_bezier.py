"""Read-only diagnostic for the bezier autoencoder. Loads last.ckpt, runs one
val batch, and prints node-position spread, junction matching stats, edge ctrl
magnitudes, per-lane reconstruction error, and existence-head probabilities.

Does not modify training logic or write anything to disk.
"""
import os
os.environ.setdefault("PROJECT_ROOT", os.getcwd())
os.environ.setdefault("SCRATCH_ROOT", "data")
os.environ.setdefault("DATASET_ROOT", "data")
os.environ.setdefault("CONFIG_PATH", os.path.join(os.getcwd(), "cfgs"))

import numpy as np
import torch
from hydra import initialize_config_dir, compose
from torch_geometric.loader import DataLoader

from models.scenario_dreamer_autoencoder_bezier import ScenarioDreamerAutoEncoderBezier
from datasets.waymo.dataset_autoencoder_waymo import WaymoDatasetAutoEncoder
from utils.data_container import get_batches, get_features, get_edge_indices

torch.set_printoptions(sci_mode=False, precision=4)
CKPT = "data/checkpoints/scenario_dreamer_autoencoder_bezier_waymo/last.ckpt"


def main():
    with initialize_config_dir(version_base=None, config_dir=os.path.join(os.getcwd(), "cfgs")):
        cfg = compose(config_name="config_autoencoder_bezier_train")
    cfg = cfg.ae
    from omegaconf import OmegaConf
    OmegaConf.set_struct(cfg, False)
    cfg.dataset_name = "waymo"
    OmegaConf.set_struct(cfg, True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ScenarioDreamerAutoEncoderBezier.load_from_checkpoint(CKPT, cfg=cfg, map_location="cpu")
    model = model.to(device).eval()
    m = model.model  # AutoEncoderBezier

    ds = WaymoDatasetAutoEncoder(cfg.dataset, split_name="val", mode="eval")
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    data = next(iter(dl)).to(device)

    with torch.no_grad():
        _, lane_batch, _ = get_batches(data)
        feats = get_features(data)
        x_lane_states, x_lane_conn = feats[4], feats[6]
        _, l2l_edge_index, _ = get_edge_indices(data)
        dec = m.forward(data)

        node_xy = dec["node_xy"]              # (B, N, 2)
        edge_ctrl = dec["edge_ctrl"]          # (B, N, N, 4)
        edge_logits = dec["edge_exist_logits"]
        node_logits = dec["node_exist_logits"]
        B, N = node_logits.shape

        print(f"device={device} B={B} N={N}  total GT lanes={x_lane_states.shape[0]}")
        print("\n--- node_xy spread (per scene) ---")
        for b in range(B):
            nx = node_xy[b]
            print(f" scene {b}: x[{nx[:,0].min():.3f},{nx[:,0].max():.3f}] "
                  f"y[{nx[:,1].min():.3f},{nx[:,1].max():.3f}] "
                  f"std=({nx[:,0].std():.3f},{nx[:,1].std():.3f})")

        print("\n--- edge_ctrl magnitude (offsets, scale=%.2f) ---" % m.cfg.bezier_ctrl_scale)
        ac = edge_ctrl.abs()
        print(f" |ctrl| mean={ac.mean():.4f} max={ac.max():.4f} "
              f"frac>0.9*scale={(ac > 0.9*m.cfg.bezier_ctrl_scale).float().mean():.4f}")

        print("\n--- existence head probs ---")
        ep = torch.sigmoid(edge_logits)
        npb = torch.sigmoid(node_logits)
        offdiag = ~torch.eye(N, dtype=torch.bool, device=device)
        print(f" edge prob: mean={ep[:,offdiag].mean():.4f} max={ep[:,offdiag].max():.4f} "
              f"frac>0.5={(ep[:,offdiag]>0.5).float().mean():.6f} "
              f"#>0.5/scene={(ep[:,offdiag]>0.5).float().sum()/B:.1f}")
        print(f" node prob: mean={npb.mean():.4f} max={npb.max():.4f} "
              f"#>0.5/scene={(npb>0.5).float().sum()/B:.1f}")

        # --- replicate _match_nodes stats ---
        mt = m._match_nodes(dec, x_lane_states, lane_batch, l2l_edge_index, x_lane_conn)
        counts = torch.bincount(lane_batch, minlength=B).cpu().numpy()
        print("\n--- junction matching (_match_nodes) ---")
        if mt["lane_b"] is not None:
            lane_b = mt["lane_b"].cpu().numpy()
            for b in range(B):
                n_lanes = int(counts[b])
                n_repr = int((lane_b == b).sum())  # lanes that produced a target edge
                print(f" scene {b}: GT_lanes={n_lanes} representable(edge target)={n_repr} "
                      f"dropped(sj<0/ej<0/sj==ej)={n_lanes - n_repr}")
            # node<->junction matched count
            print(f" total matched junctions={mt['nm_b'].shape[0]} "
                  f"total representable lanes={mt['lane_b'].shape[0]} "
                  f"(of {x_lane_states.shape[0]} GT lanes)")
        else:
            print(" NO representable lanes at all!")

        # --- per-lane reconstruction error (matched beziers vs GT) ---
        lane_samples = m.reconstruct_lanes(dec, x_lane_states, lane_batch, l2l_edge_index, x_lane_conn)
        per_lane_l1 = (lane_samples - x_lane_states).abs().mean(dim=(1, 2))  # (N_lanes,)
        replaced = per_lane_l1 > 1e-9  # lanes that were NOT left as GT clone
        print("\n--- reconstruction (reconstruct_lanes) ---")
        print(f" lanes replaced by bezier={int(replaced.sum())}/{lane_samples.shape[0]} "
              f"(rest fall back to GT)")
        if replaced.any():
            print(f" replaced-lane L1: mean={per_lane_l1[replaced].mean():.4f} "
                  f"max={per_lane_l1[replaced].max():.4f}")
            # endpoint error vs interior error (per-lane, against the lane's OWN GT)
            ep_err = ((lane_samples[replaced][:, 0] - x_lane_states[replaced][:, 0]).abs().mean()
                      + (lane_samples[replaced][:, -1] - x_lane_states[replaced][:, -1]).abs().mean()) / 2
            mid = lane_samples.shape[1] // 2
            mid_err = (lane_samples[replaced][:, mid] - x_lane_states[replaced][:, mid]).abs().mean()
            print(f" per-lane endpoint L1={ep_err:.4f}  midpoint L1={mid_err:.4f}")

        # --- how many DISTINCT junctions / node-pairs do GT lanes map to? ---
        from omegaconf import OmegaConf
        def collapse_report(tag, eps_override=None):
            old = m.cfg.junction_merge_eps
            if eps_override is not None:
                OmegaConf.set_struct(m.cfg, False)
                m.cfg.junction_merge_eps = eps_override
                OmegaConf.set_struct(m.cfg, True)
            mt2 = m._match_nodes(dec, x_lane_states, lane_batch, l2l_edge_index, x_lane_conn)
            lb = mt2["lane_b"].cpu().numpy(); ls = mt2["lane_snode"].cpu().numpy(); ld = mt2["lane_dnode"].cpu().numpy()
            print(f"\n--- lane->node collapse  [{tag}] ---")
            for b in range(B):
                sel = lb == b
                pairs = set(zip(ls[sel].tolist(), ld[sel].tolist()))
                junc = set(ls[sel].tolist()) | set(ld[sel].tolist())
                print(f" scene {b}: GT_lanes={int(sel.sum())} junctions={len(junc)} "
                      f"distinct-pairs={len(pairs)} collapse={int(sel.sum())/max(len(pairs),1):.1f}x")
            OmegaConf.set_struct(m.cfg, False); m.cfg.junction_merge_eps = old; OmegaConf.set_struct(m.cfg, True)

        collapse_report(f"eps={m.cfg.junction_merge_eps} (succ + dist)")
        collapse_report("eps=0 (succ-connectivity ONLY)", eps_override=0.0)


if __name__ == "__main__":
    main()
