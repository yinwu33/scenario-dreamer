#!/usr/bin/env python
"""Analyze the preprocessed Waymo AE dataset to check whether the design
concerns about ``_graph_loss_gt_nodes`` / ``_match_nodes`` (in
``nn_modules/autoencoder_bezier.py``) are *real* in the data, or whether the
data makes them moot (so they don't actually hurt training).

The function builds GT junctions by merging lane endpoints that are within
``junction_merge_eps`` (L1, normalized coords), exactly replicating
``_match_nodes``. We then measure, per scene:

  C1  Multigraph collapse:  two GT lanes mapped to the SAME directed
      (start_node, end_node) pair -> only one decoder edge slot can represent
      them, so the surplus lane's geometry is lost / averaged.

  C2  Node-slot truncation:  #junctions > num_graph_nodes (n) -> low-degree
      junctions are dropped and their lanes get NO supervision.

  C3  Conflicting node supervision:  spread (L1 diameter) of the endpoints
      merged into one junction. node_xy is pulled to the junction *mean*
      (node_pos_loss) but also to each lane's individual endpoint (reg_loss).
      If spread ~ 0 the two targets agree and there is no conflict.

  C6  Self-loop lanes:  a lane whose start & end fall in the same junction
      (sj == ej) -> dropped (no edge target, no regression).

  C4  Distance-only merging vs. real connectivity (pred/succ):
      C4a false-merge: endpoints merged by distance with NO pred/succ link.
      C4b missed-merge: pred/succ-connected endpoints that are >= eps apart
          (so they are NOT merged -> structural sharing fails).

Usage:
    python analyze_map.py [--split train] [--num 5000] [--eps 0.02] [--n 100]
"""
import argparse
import glob
import os
import pickle
import random

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

# connection-type one-hot column order (utils/data_helpers.py)
CT = {"none": 0, "pred": 1, "succ": 2, "left": 3, "right": 4, "self": 5}


def build_junctions(starts, ends, eps):
    """Replicate _match_nodes junction merging. Returns (num_j, labels, coords,
    merged_pairs) where labels has length 2*m (slot 2l=start, 2l+1=end) and
    merged_pairs is the list of (slot_i, slot_j) cross-lane pairs that merged."""
    m = len(starts)
    coords = np.empty((2 * m, 2), dtype=np.float64)
    coords[0::2] = starts
    coords[1::2] = ends
    rows, cols = [], []
    if eps > 0 and m > 1:
        d = np.abs(coords[:, None, :] - coords[None, :, :]).sum(-1)
        lane_of = np.arange(2 * m) // 2
        iu = np.triu_indices(2 * m, 1)
        close = (d[iu] < eps) & (lane_of[iu[0]] != lane_of[iu[1]])
        rows = iu[0][close].tolist()
        cols = iu[1][close].tolist()
    if rows:
        adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(2 * m, 2 * m))
    else:
        adj = sp.coo_matrix(([], ([], [])), shape=(2 * m, 2 * m))
    num_j, labels = connected_components(adj, directed=False)
    merged_pairs = list(zip(rows, cols))
    return num_j, labels, coords, merged_pairs


def connection_endpoints(edge_index, ct_onehot):
    """Return a set of expected endpoint coincidences from pred/succ edges.

    Edge (src=a, dst=b, type='pred') means a is predecessor of b -> a.end ~
    b.start, encoded as frozenset({('e',a),('s',b)}).
    Edge type='succ' (src=a,dst=b): a is successor of b -> a.start ~ b.end,
    encoded as frozenset({('s',a),('e',b)}).
    """
    ei = edge_index.numpy() if hasattr(edge_index, "numpy") else np.asarray(edge_index)
    types = ct_onehot.argmax(1)
    expected = set()
    pred_count = succ_count = 0
    for i in range(ei.shape[1]):
        a, b = int(ei[0, i]), int(ei[1, i])
        t = int(types[i])
        if t == CT["pred"]:
            expected.add(frozenset({("e", a), ("s", b)}))
            pred_count += 1
        elif t == CT["succ"]:
            expected.add(frozenset({("s", a), ("e", b)}))
            succ_count += 1
    return expected, pred_count, succ_count


def analyze_scene(d, eps, n):
    rp = np.asarray(d["road_points"])[:, :, :2].astype(np.float64)  # (L,P,2)
    L = rp.shape[0]
    starts, ends = rp[:, 0, :], rp[:, -1, :]
    num_j, labels, coords, merged_pairs = build_junctions(starts, ends, eps)

    res = {"L": L, "J": num_j, "truncated": int(num_j > n)}

    # per-lane (sj, ej) after merge
    sj = labels[0::2]
    ej = labels[1::2]
    self_loops = int((sj == ej).sum())

    # C1 multigraph collapse: duplicate directed (sj,ej) among non-self lanes
    pair_count = {}
    for l in range(L):
        if sj[l] == ej[l]:
            continue
        key = (int(sj[l]), int(ej[l]))
        pair_count[key] = pair_count.get(key, 0) + 1
    collapsed_lanes = sum(c - 1 for c in pair_count.values() if c > 1)  # surplus lanes losing a slot
    res["self_loops"] = self_loops
    res["collapsed_lanes"] = collapsed_lanes
    res["valid_lanes"] = L - self_loops  # lanes that get an edge target

    # C3 junction endpoint spread (L1 diameter) for multi-member junctions
    spreads = []
    for j in range(num_j):
        mem = coords[labels == j]
        if len(mem) >= 2:
            diam = (mem.max(0) - mem.min(0)).sum()  # L1 bbox diameter
            spreads.append(diam)
    res["spreads"] = spreads
    res["multi_junctions"] = len(spreads)

    # C4 distance-only merge vs connectivity
    expected, pred_count, succ_count = connection_endpoints(
        d["edge_index_lane_to_lane"], np.asarray(d["road_connection_types"]))
    res["pred_succ_edges"] = pred_count + succ_count

    # C4a: merged cross-lane pairs broken down by role + backed-by-connection
    role_combo = {"s-s": 0, "e-e": 0, "s-e": 0}  # symmetric: s-e == e-s
    es_pairs = 0          # end<->start (directly connectable orientation)
    es_backed = 0         # of those, backed by a pred/succ edge
    for (pi, pj) in merged_pairs:
        li, ri = pi // 2, pi % 2      # ri: 0=start,1=end
        lj, rj = pj // 2, pj % 2
        roli = "s" if ri == 0 else "e"
        rolj = "s" if rj == 0 else "e"
        combo = "".join(sorted([roli + "-" , rolj])) if False else None
        # classify
        if roli == rolj == "s":
            role_combo["s-s"] += 1
        elif roli == rolj == "e":
            role_combo["e-e"] += 1
        else:
            role_combo["s-e"] += 1
            es_pairs += 1
            key = frozenset({(roli, li), (rolj, lj)})
            if key in expected:
                es_backed += 1
    res["role_combo"] = role_combo
    res["es_pairs"] = es_pairs
    res["es_backed"] = es_backed

    # C4b: pred/succ connections whose endpoints are within eps (would merge)
    near = 0
    for key in expected:
        (r1, l1), (r2, l2) = tuple(key)
        p1 = starts[l1] if r1 == "s" else ends[l1]
        p2 = starts[l2] if r2 == "s" else ends[l2]
        if np.abs(p1 - p2).sum() < eps:
            near += 1
    res["conn_expected"] = len(expected)
    res["conn_near"] = near  # within eps -> actually merged
    return res


def pct(x, p):
    return float(np.percentile(x, p)) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/scenario_dreamer_ae_preprocess_waymo")
    ap.add_argument("--split", default="train")
    ap.add_argument("--num", type=int, default=5000, help="random files to sample")
    ap.add_argument("--eps", type=float, default=0.02, help="junction_merge_eps")
    ap.add_argument("--n", type=int, default=100, help="num_graph_nodes (node slots)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.root, args.split, "*.pkl"))
    random.seed(args.seed)
    if args.num and args.num < len(files):
        files = random.sample(files, args.num)
    print(f"Analyzing {len(files)} scenes from split='{args.split}' "
          f"(eps={args.eps}, n={args.n})\n")

    Ls, Js = [], []
    n_trunc = 0
    tot_lanes = tot_self = tot_collapse = tot_valid = 0
    all_spreads = []
    role_tot = {"s-s": 0, "e-e": 0, "s-e": 0}
    es_pairs = es_backed = 0
    conn_expected = conn_near = 0
    scenes_with_collapse = 0
    scenes_with_self = 0

    for f in files:
        try:
            d = pickle.load(open(f, "rb"))
        except Exception as e:
            print("skip", f, e)
            continue
        r = analyze_scene(d, args.eps, args.n)
        Ls.append(r["L"]); Js.append(r["J"])
        n_trunc += r["truncated"]
        tot_lanes += r["L"]; tot_self += r["self_loops"]
        tot_collapse += r["collapsed_lanes"]; tot_valid += r["valid_lanes"]
        all_spreads.extend(r["spreads"])
        for k in role_tot:
            role_tot[k] += r["role_combo"][k]
        es_pairs += r["es_pairs"]; es_backed += r["es_backed"]
        conn_expected += r["conn_expected"]; conn_near += r["conn_near"]
        scenes_with_collapse += int(r["collapsed_lanes"] > 0)
        scenes_with_self += int(r["self_loops"] > 0)

    S = len(Ls)
    print("=" * 70)
    print("SCENE SIZE")
    print(f"  lanes/scene:      mean={np.mean(Ls):.1f}  p50={pct(Ls,50):.0f}  "
          f"p99={pct(Ls,99):.0f}  max={max(Ls)}")
    print(f"  junctions/scene:  mean={np.mean(Js):.1f}  p50={pct(Js,50):.0f}  "
          f"p99={pct(Js,99):.0f}  max={max(Js)}")

    print("\n" + "=" * 70)
    print(f"C2  NODE-SLOT TRUNCATION  (junctions > n={args.n})")
    print(f"  scenes truncated: {n_trunc}/{S} ({100*n_trunc/S:.3f}%)   "
          f"max junctions seen: {max(Js)}")
    verdict = "NOT A PROBLEM" if n_trunc == 0 else "OCCURS"
    print(f"  --> {verdict} (n=100 vs max {max(Js)} junctions)")

    print("\n" + "=" * 70)
    print("C1  MULTIGRAPH COLLAPSE  (>=2 lanes share one directed node pair)")
    print(f"  surplus lanes collapsed: {tot_collapse}/{tot_lanes} lanes "
          f"({100*tot_collapse/tot_lanes:.4f}%)")
    print(f"  scenes affected:         {scenes_with_collapse}/{S} "
          f"({100*scenes_with_collapse/S:.3f}%)")
    verdict = "NEGLIGIBLE" if tot_collapse / tot_lanes < 1e-3 else "REAL"
    print(f"  --> {verdict}")

    print("\n" + "=" * 70)
    print("C6  SELF-LOOP LANES  (start & end in same junction -> dropped)")
    print(f"  self-loop lanes: {tot_self}/{tot_lanes} ({100*tot_self/tot_lanes:.4f}%)")
    print(f"  scenes affected: {scenes_with_self}/{S} ({100*scenes_with_self/S:.3f}%)")
    verdict = "NEGLIGIBLE" if tot_self / tot_lanes < 1e-3 else "REAL"
    print(f"  --> {verdict}")

    print("\n" + "=" * 70)
    print("C3  JUNCTION ENDPOINT SPREAD  (L1 diameter of merged endpoints)")
    print(f"  multi-endpoint junctions: {len(all_spreads)}")
    if all_spreads:
        print(f"  spread:  mean={np.mean(all_spreads):.5f}  p50={pct(all_spreads,50):.5f} "
              f" p99={pct(all_spreads,99):.5f}  max={max(all_spreads):.5f}")
        print(f"  (eps={args.eps}; spread is bounded by transitive chains of eps)")
        verdict = ("SMALL vs lane scale (~2.0 fov span)"
                   if np.mean(all_spreads) < 0.05 else "NON-TRIVIAL")
        print(f"  --> conflict magnitude {verdict}")

    print("\n" + "=" * 70)
    print("C4  DISTANCE-ONLY MERGE vs REAL pred/succ CONNECTIVITY")
    merged_total = sum(role_tot.values())
    print(f"  merged cross-lane endpoint pairs: {merged_total}")
    print(f"    by role:  start-start={role_tot['s-s']}  end-end={role_tot['e-e']}  "
          f"start-end={role_tot['s-e']}")
    print("  C4a FALSE-MERGE (end<->start pairs not backed by a pred/succ edge):")
    if es_pairs:
        print(f"    {es_pairs - es_backed}/{es_pairs} "
              f"({100*(es_pairs-es_backed)/es_pairs:.2f}%) end<->start merges have NO pred/succ link")
    print("  C4b MISSED-MERGE (pred/succ connections NOT merged, endpoints >= eps):")
    if conn_expected:
        missed = conn_expected - conn_near
        print(f"    {missed}/{conn_expected} ({100*missed/conn_expected:.2f}%) "
              f"pred/succ links have endpoints >= eps apart -> NOT merged")
        verdict = "GOOD: connectivity == proximity" if missed / conn_expected < 0.02 else \
                  "MISMATCH: proximity misses real connections"
        print(f"    --> {verdict}")
    print()


if __name__ == "__main__":
    main()
