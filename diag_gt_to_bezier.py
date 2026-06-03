"""Visualize the GROUND-TRUTH vector-lane -> bezier-lane conversion (no model).

For each scene we reproduce exactly the GT-junction construction used in
AutoEncoderBezier._match_nodes (succ-connectivity unions + distance fallback
single-linkage merging), then represent each GT lane as a cubic bezier whose
endpoints are the *junction centroids* of its start/end (this is the shared-node
target the model is trained toward) and whose two inner control points are the
LEAST-SQUARES best fit to the GT polyline. We then re-interpolate the bezier and
plot:  left = GT vector lanes,  right = bezier re-interpolated (+ junction nodes).

This isolates whether the GT->bezier *representation* is faithful, independent of
the trained weights.  Set DIAG_EPS to override junction_merge_eps (e.g. 0 to use
succ-connectivity only).

Read-only: writes PNGs to diag_gt_to_bezier_out/ only.
"""
import os
os.environ.setdefault("PROJECT_ROOT", os.getcwd())
os.environ.setdefault("SCRATCH_ROOT", "data")
os.environ.setdefault("DATASET_ROOT", "data")
os.environ.setdefault("CONFIG_PATH", os.path.join(os.getcwd(), "cfgs"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader

from datasets.waymo.dataset_autoencoder_waymo import WaymoDatasetAutoEncoder
from utils.data_container import get_batches, get_features, get_edge_indices

SUCC_CONN_INDEX = 2
OUT = "diag_gt_to_bezier_out"
NUM_FIT_POINTS = None  # set from data
EPS = float(os.environ.get("DIAG_EPS", "0.02"))
USE_SUCC = os.environ.get("DIAG_USE_SUCC", "0") == "1"  # default: distance-only
NUM_SCENES = 8


def cubic_basis(num_points):
    t = np.linspace(0.0, 1.0, num_points)
    omt = 1.0 - t
    return np.stack([omt**3, 3*t*omt**2, 3*t**2*omt, t**3], axis=-1)  # (P,4)


def build_junctions(starts, ends, succ_pairs, eps, use_succ=False):
    """Junction construction for ONE scene.

    starts/ends: (m,2) GT lane endpoints. Nodes (junctions) are determined by
    DISTANCE ONLY (endpoints within ``eps`` L1 are the same junction); the
    succ/pred topological union is only added when ``use_succ`` is True.
    A lane's own two endpoints are never merged together.
    Returns (labels (2m,), centroids (J,2), degree (J,)).
    """
    m = starts.shape[0]
    coords = np.empty((2 * m, 2), dtype=np.float64)
    coords[0::2] = starts   # slot 2l   = start of lane l
    coords[1::2] = ends     # slot 2l+1 = end of lane l

    rows, cols = [], []
    if use_succ:
        for (a, b) in succ_pairs:
            rows.append(2 * a + 1)
            cols.append(2 * b)
    if eps > 0 and m > 1:
        d = np.abs(coords[:, None, :] - coords[None, :, :]).sum(-1)
        lane_of = np.arange(2 * m) // 2
        iu = np.triu_indices(2 * m, 1)
        close = (d[iu] < eps) & (lane_of[iu[0]] != lane_of[iu[1]])
        rows += iu[0][close].tolist()
        cols += iu[1][close].tolist()

    if rows:
        adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(2 * m, 2 * m))
    else:
        adj = sp.coo_matrix(([], ([], [])), shape=(2 * m, 2 * m))
    num_j, labels = connected_components(adj, directed=False)
    centroids = np.zeros((num_j, 2))
    degree = np.bincount(labels, minlength=num_j)  # #endpoint slots in each junction
    for j in range(num_j):
        centroids[j] = coords[labels == j].mean(0)
    return labels, centroids, degree


def fit_bezier(gt_poly, p0, p3, basis):
    """Least-squares inner control points P1,P2 for a cubic bezier with fixed
    endpoints p0,p3 fitting gt_poly (P,2). Returns reinterpolated poly (P,2)."""
    rhs = gt_poly - basis[:, [0]] * p0[None, :] - basis[:, [3]] * p3[None, :]  # (P,2)
    A = basis[:, 1:3]  # (P,2)
    pc, *_ = np.linalg.lstsq(A, rhs, rcond=None)  # (2,2) -> P1,P2
    p1, p2 = pc[0], pc[1]
    ctrl = np.stack([p0, p1, p2, p3], axis=0)  # (4,2)
    return basis @ ctrl  # (P,2)


def plot_pair(gt_lanes, bez_lanes, bez_own, centroids, degree, title, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    panels = ((axes[0], gt_lanes, "GT vector", False),
              (axes[1], bez_lanes, "bezier (shared junction endpoints)", True),
              (axes[2], bez_own, "bezier (lane's OWN endpoints)", False))
    for ax, lanes, name, show_j in panels:
        for ln in lanes:
            ax.plot(ln[:, 0], ln[:, 1], color="grey", lw=1.0, ls="--")
            ax.scatter(ln[0, 0], ln[0, 1], color="tab:blue", s=8, zorder=3)
            ax.scatter(ln[-1, 0], ln[-1, 1], color="tab:blue", s=8, zorder=3)
        if show_j and centroids is not None:
            deg1 = degree == 1
            ax.scatter(centroids[deg1, 0], centroids[deg1, 1], color="limegreen", s=60,
                       marker="o", facecolors="none", linewidths=1.5, zorder=5,
                       label=f"degree-1 junctions ({int(deg1.sum())})")
            ax.scatter(centroids[~deg1, 0], centroids[~deg1, 1], color="red", s=40,
                       marker="x", zorder=4, label=f"degree>=2 junctions ({int((~deg1).sum())})")
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title(name)
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal"); ax.grid(alpha=0.2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    with initialize_config_dir(version_base=None, config_dir=os.path.join(os.getcwd(), "cfgs")):
        cfg = compose(config_name="config_autoencoder_bezier_train")
    cfg = cfg.ae
    OmegaConf.set_struct(cfg, False); cfg.dataset_name = "waymo"; OmegaConf.set_struct(cfg, True)

    ds = WaymoDatasetAutoEncoder(cfg.dataset, split_name="val", mode="eval")
    dl = DataLoader(ds, batch_size=NUM_SCENES, shuffle=False, num_workers=0)
    data = next(iter(dl))

    _, lane_batch, _ = get_batches(data)
    feats = get_features(data)
    x_lane_states, x_lane_conn = feats[4], feats[6]   # (N,P,2), (E,conn)
    _, l2l_edge_index, _ = get_edge_indices(data)

    P = x_lane_states.shape[1]
    basis = cubic_basis(P)
    xls = x_lane_states.cpu().numpy()
    lb = lane_batch.cpu().numpy()
    l2l = l2l_edge_index.cpu().numpy()
    if x_lane_conn.shape[1] > SUCC_CONN_INDEX:
        succ_mask = (x_lane_conn[:, SUCC_CONN_INDEX] > 0.5).cpu().numpy()
    else:
        succ_mask = np.zeros(l2l.shape[1], dtype=bool)

    B = NUM_SCENES
    counts = np.bincount(lb, minlength=B)
    offsets = np.zeros(B, dtype=np.int64); offsets[1:] = np.cumsum(counts)[:-1]

    print(f"eps={EPS}  use_succ={USE_SUCC}  P(points/lane)={P}")
    for b in range(B):
        m = int(counts[b]); g0 = int(offsets[b])
        if m == 0:
            continue
        starts = xls[g0:g0 + m, 0, :]
        ends = xls[g0:g0 + m, -1, :]
        # succ pairs (local)
        sm = succ_mask & (lb[l2l[0]] == b)
        succ_pairs = [(int(s - g0), int(d - g0)) for s, d in zip(l2l[0][sm], l2l[1][sm])
                      if 0 <= s - g0 < m and 0 <= d - g0 < m]
        labels, centroids, degree = build_junctions(starts, ends, succ_pairs, EPS, use_succ=USE_SUCC)

        gt_lanes, bez_lanes, bez_own = [], [], []
        pair_set = set()
        for l in range(m):
            sj, ej = int(labels[2 * l]), int(labels[2 * l + 1])
            gt = xls[g0 + l]
            gt_lanes.append(gt)
            # variant: fit using the lane's OWN endpoints (no junction sharing)
            bez_own.append(fit_bezier(gt, gt[0], gt[-1], basis))
            if sj == ej:
                bez_lanes.append(gt)  # degenerate (skipped in training) -> show GT
                continue
            pair_set.add((sj, ej))
            p0, p3 = centroids[sj], centroids[ej]   # shared-node endpoints
            bez_lanes.append(fit_bezier(gt, p0, p3, basis))

        deg_hist = np.bincount(degree)
        title = (f"scene {b}: {m} GT lanes -> {len(centroids)} junctions, "
                 f"{len(pair_set)} distinct node-pairs (collapse {m/max(len(pair_set),1):.1f}x)")
        plot_pair(gt_lanes, bez_lanes, bez_own, centroids, degree, title,
                  os.path.join(OUT, f"scene_{b}_eps{EPS}_succ{int(USE_SUCC)}.png"))
        print(f" {title}\n   degree hist (idx=degree, val=#junctions): {deg_hist.tolist()}")
    print(f"\nwrote PNGs to {OUT}/")


if __name__ == "__main__":
    main()
