"""Bezier lane-graph variant of the Scenario Dreamer AutoEncoder.

This module implements a *query-based graph decoder* for lanes, while keeping the
encoder and the agent branch identical to ``nn_modules/autoencoder.py``.

Design (see ``TASK_bezier_lane_graph_decoder.md``)
--------------------------------------------------
The encoder is unchanged: it still produces per-lane-token latents
(``lane_mu/lane_log_var``), so the latent space / LDM compatibility is preserved.

The decoder replaces the per-lane-token geometry head with a DETR-style
node-edge graph decoder:

* ``N = cfg.num_graph_nodes`` learnable node queries cross-attend to the
  (per-scene) lane latents -> node embeddings.
* A node head predicts ``(B, N, 3)``:
    - ``[..., 0]``    node existence logit
    - ``[..., 1:3]``  node xy position (tanh -> normalized [-1, 1] space)
* An edge head predicts ``(B, N, N, 5)`` for directed edge ``i -> j``:
    - ``[..., 0]``    edge existence logit
    - ``[..., 1:5]``  two inner cubic-Bezier control points P1, P2

Each directed edge ``i -> j`` resolves to a cubic Bezier with
``P0 = node_xy[i]``, ``P3 = node_xy[j]`` and the predicted inner control points,
sampled into a polyline of ``cfg.num_points_per_lane`` points. Endpoint
continuity between connected lanes is therefore *structural* (shared nodes).

Supervision (no GT shared nodes required)
-----------------------------------------
Each GT lane is itself a polyline. Each predicted directed edge is resolved into
a polyline of the same length, so predicted edges and GT lanes are directly
comparable. We Hungarian-match predicted-edge polylines to GT lane polylines
(STSU / MapTR style) and supervise:

* edge existence (matched -> 1, else 0),
* matched-edge polyline regression (L1) to the GT lane,
* node existence (endpoint of a matched edge -> 1),
* (optional) succ/pred endpoint-continuity loss that encourages connected GT
  lanes' matched edges to share a node coordinate.

``left/right`` lane relationships are intentionally dropped (not representable in
an endpoint node graph).
"""

import numpy as np
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from scipy.optimize import linear_sum_assignment
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from utils.layers import ResidualMLP, AutoEncoderFactorizedAttentionBlock
from utils.train_helpers import weight_init
from utils.losses import GeometricLosses
from utils.data_container import get_batches, get_features, get_edge_indices, get_encoder_edge_indices
from utils.data_helpers import reparameterize
from cfgs.config import NON_PARTITIONED

# reuse the (unchanged) encoder so the latent space stays identical to the
# original autoencoder (keeps LDM / latent-cache compatibility).
from nn_modules.autoencoder import ScenarioDreamerEncoder

# index of the "succ" lane-connection type in the one-hot encoding (same for
# waymo and nuplan: {none:0, pred:1, succ:2, ...}).
SUCC_CONN_INDEX = 2


def cubic_bezier_basis(num_points: int, device, dtype) -> torch.Tensor:
    """Bernstein basis matrix for a cubic Bezier sampled at ``num_points``.

    Returns:
        Tensor of shape ``(num_points, 4)`` such that
        ``polyline = basis @ control_points``.
    """
    t = torch.linspace(0.0, 1.0, num_points, device=device, dtype=dtype)
    one_minus_t = 1.0 - t
    basis = torch.stack(
        [
            one_minus_t ** 3,
            3.0 * t * one_minus_t ** 2,
            3.0 * t ** 2 * one_minus_t,
            t ** 3,
        ],
        dim=-1,
    )  # (num_points, 4)
    return basis


def resolve_bezier_edges(
    node_xy: torch.Tensor,
    edge_ctrl: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Resolve directed-edge Bezier control points into polylines.

    The inner control points are parameterized as *offsets relative to the
    endpoints* (``P1 = P0 + delta1``, ``P2 = P3 + delta2``). Anchoring the
    control points to the node endpoints forces the curve geometry to be carried
    by the node positions, preventing the degenerate "all lanes loop out of one
    collapsed node" solution.

    Args:
        node_xy: ``(N, 2)`` node positions for one scene.
        edge_ctrl: ``(N, N, 4)`` control-point offsets per directed edge
            ``i -> j`` (``[..., 0:2] = delta1`` for P1, ``[..., 2:4] = delta2``
            for P2). Expected to be pre-bounded (e.g. ``tanh * scale``).
        basis: ``(P, 4)`` cubic Bernstein basis.

    Returns:
        ``(N, N, P, 2)`` polylines, where ``poly[i, j]`` is the curve from node
        ``i`` to node ``j``.
    """
    n = node_xy.shape[0]
    p0 = node_xy.unsqueeze(1).expand(n, n, 2)   # source i
    p3 = node_xy.unsqueeze(0).expand(n, n, 2)   # dest j
    p1 = p0 + edge_ctrl[..., 0:2]
    p2 = p3 + edge_ctrl[..., 2:4]
    ctrl = torch.stack([p0, p1, p2, p3], dim=-2)  # (N, N, 4, 2)
    poly = torch.einsum('pk,ijkd->ijpd', basis, ctrl)  # (N, N, P, 2)
    return poly


class BezierLaneGraphDecoder(nn.Module):
    """Query-based node-edge graph decoder with cubic-Bezier lane edges."""

    def __init__(self, cfg):
        super(BezierLaneGraphDecoder, self).__init__()
        self.cfg = cfg

        # ------------------- latent -> hidden projections ---------------- #
        self.lane_mlp = nn.Linear(self.cfg.lane_latent_dim, self.cfg.hidden_dim)
        self.agent_mlp = nn.Linear(self.cfg.agent_latent_dim, self.cfg.agent_hidden_dim)
        self.downsample_lane_mlp = nn.Linear(self.cfg.hidden_dim, self.cfg.lane_conn_hidden_dim)
        self.lane_conn_mlp = nn.Linear(self.cfg.lane_conn_hidden_dim * 2, self.cfg.lane_conn_hidden_dim)

        # ------------------- factorized attention (agent <-> lane) ------- #
        # Used to contextualize lane/agent embeddings before the graph head.
        self.decoder_transformer_blocks = nn.ModuleList([
            AutoEncoderFactorizedAttentionBlock(
                lane_hidden_dim=self.cfg.hidden_dim,
                lane_feedforward_dim=self.cfg.dim_f,
                lane_num_heads=self.cfg.num_heads,
                agent_hidden_dim=self.cfg.agent_hidden_dim,
                agent_feedforward_dim=self.cfg.agent_dim_f,
                agent_num_heads=self.cfg.agent_num_heads,
                lane_conn_hidden_dim=self.cfg.lane_conn_hidden_dim,
                dropout=self.cfg.dropout)
            for _ in range(self.cfg.num_decoder_blocks)
        ])

        # ------------------- agent heads (unchanged) --------------------- #
        self.pred_agent_states = ResidualMLP(input_dim=self.cfg.agent_hidden_dim,
                                             hidden_dim=self.cfg.agent_hidden_dim,
                                             n_hidden=3,
                                             output_dim=self.cfg.state_dim)
        self.pred_agent_types = ResidualMLP(input_dim=self.cfg.agent_hidden_dim,
                                            hidden_dim=self.cfg.agent_hidden_dim,
                                            n_hidden=2,
                                            output_dim=self.cfg.num_agent_types)

        # ------------------- node-graph branch --------------------------- #
        self.num_nodes = self.cfg.num_graph_nodes
        self.node_queries = nn.Parameter(torch.empty(self.num_nodes, self.cfg.hidden_dim))
        nn.init.normal_(self.node_queries, std=0.02)

        node_layer = nn.TransformerDecoderLayer(
            d_model=self.cfg.hidden_dim,
            nhead=self.cfg.num_heads,
            dim_feedforward=self.cfg.dim_f,
            dropout=self.cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.node_decoder = nn.TransformerDecoder(node_layer, num_layers=self.cfg.num_node_decoder_layers)

        # node head: existence logit + xy
        self.pred_node = ResidualMLP(input_dim=self.cfg.hidden_dim,
                                     hidden_dim=self.cfg.hidden_dim,
                                     n_hidden=3,
                                     output_dim=3)
        # edge head operates on a downsampled pairwise representation to bound
        # the O(N^2) memory of the dense adjacency tensor.
        self.edge_hidden_dim = self.cfg.edge_hidden_dim
        self.edge_node_proj = nn.Linear(self.cfg.hidden_dim, self.edge_hidden_dim)
        self.pred_edge = ResidualMLP(input_dim=self.edge_hidden_dim * 2,
                                     hidden_dim=self.edge_hidden_dim,
                                     n_hidden=3,
                                     output_dim=5)
        # max magnitude of the (relative) inner control-point offsets, in the
        # normalized [-1, 1] coordinate space.
        self.ctrl_scale = self.cfg.bezier_ctrl_scale

        self.apply(weight_init)

    def forward(
        self,
        agent_latents: torch.Tensor,
        lane_latents: torch.Tensor,
        lane_batch: torch.Tensor,
        a2a_edge_index: torch.Tensor,
        l2l_edge_index: torch.Tensor,
        l2a_edge_index: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Decode latents into agent states and a Bezier lane graph.

        Returns a dict with:
            agent_states_pred:  (N_agents, state_dim)
            agent_types_logits: (N_agents, num_agent_types)
            agent_types_pred:   (N_agents,)
            node_exist_logits:  (B, N)
            node_xy:            (B, N, 2)
            node_mask:          (B, N) bool, valid query slots (always all True;
                                kept for symmetry with lane_mask handling)
            edge_exist_logits:  (B, N, N)
            edge_ctrl:          (B, N, N, 4)
            lane_mask:          (B, L) bool, valid (non-padded) lane tokens
        """
        # ----------- latent -> hidden-dim projections -------------------- #
        agent_embeddings = self.agent_mlp(agent_latents)
        lane_embeddings = self.lane_mlp(lane_latents)

        # ----------- build lane-connection embeddings -------------------- #
        lane_embeddings_downsampled = self.downsample_lane_mlp(lane_embeddings)
        src_lane_conn_embedding = lane_embeddings_downsampled[l2l_edge_index[0]]
        dst_lane_conn_embedding = lane_embeddings_downsampled[l2l_edge_index[1]]
        lane_conn_embeddings = self.lane_conn_mlp(
            torch.cat([src_lane_conn_embedding, dst_lane_conn_embedding], dim=-1))

        # ----------- factorized attention (contextualize) --------------- #
        for block in self.decoder_transformer_blocks:
            agent_embeddings, lane_embeddings, lane_conn_embeddings = block(
                agent_embeddings,
                lane_embeddings,
                lane_conn_embeddings,
                lane_conn_embeddings,
                a2a_edge_index,
                l2l_edge_index,
                l2a_edge_index)

        # ----------- agent heads ----------------------------------------- #
        agent_states_pred = self.pred_agent_states(agent_embeddings)
        agent_types_logits = self.pred_agent_types(agent_embeddings)
        agent_types_pred = torch.argmax(agent_types_logits, dim=1)

        # ----------- node-graph branch ----------------------------------- #
        # dense per-scene lane memory + padding mask
        lane_dense, lane_mask = to_dense_batch(lane_embeddings, lane_batch)  # (B, L, H), (B, L)
        batch_size = lane_dense.shape[0]

        tgt = self.node_queries.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, H)
        # nn.Transformer uses True == "mask out" for key_padding_mask
        node_emb = self.node_decoder(
            tgt, lane_dense, memory_key_padding_mask=~lane_mask)  # (B, N, H)

        node_out = self.pred_node(node_emb)  # (B, N, 3)
        node_exist_logits = node_out[..., 0]
        node_xy = torch.tanh(node_out[..., 1:3])

        # pairwise edge head
        ne = self.edge_node_proj(node_emb)  # (B, N, edge_hidden)
        n = ne.shape[1]
        ni = ne.unsqueeze(2).expand(-1, -1, n, -1)  # source i
        nj = ne.unsqueeze(1).expand(-1, n, -1, -1)  # dest j
        edge_out = self.pred_edge(torch.cat([ni, nj], dim=-1))  # (B, N, N, 5)
        edge_exist_logits = edge_out[..., 0]
        # inner control points as bounded offsets relative to the node endpoints
        # (P1 = P0 + edge_ctrl[:2], P2 = P3 + edge_ctrl[2:]); see resolve_bezier_edges.
        edge_ctrl = torch.tanh(edge_out[..., 1:5]) * self.ctrl_scale

        return {
            'agent_states_pred': agent_states_pred,
            'agent_types_logits': agent_types_logits,
            'agent_types_pred': agent_types_pred,
            'node_exist_logits': node_exist_logits,
            'node_xy': node_xy,
            'node_mask': lane_mask.new_ones((batch_size, n), dtype=torch.bool),
            'edge_exist_logits': edge_exist_logits,
            'edge_ctrl': edge_ctrl,
            'lane_mask': lane_mask,
        }


class AutoEncoderBezier(nn.Module):
    """Scenario Dreamer AutoEncoder with a Bezier lane-graph decoder."""

    def __init__(self, cfg):
        super(AutoEncoderBezier, self).__init__()
        self.cfg = cfg
        self.encoder = ScenarioDreamerEncoder(self.cfg)
        self.decoder = BezierLaneGraphDecoder(self.cfg)

        # agent + kl losses reuse the original geometric losses
        self.agent_loss_fn = GeometricLosses['l1']()
        self.agent_type_loss_fn = GeometricLosses['cross_entropy'](apply_mean=False)
        self.kl_loss_fn = GeometricLosses['kl']()

        self.apply(weight_init)

    # ------------------------------------------------------------------ #
    # GT junction construction + node matching (method B)                #
    # ------------------------------------------------------------------ #
    def _match_nodes(self, dec, x_lane_states, lane_batch, l2l_edge_index, x_lane_conn):
        """Build GT junction nodes and match predicted node slots to them.

        For each scene:
          1. Each GT lane contributes a *start* and *end* endpoint slot.
          2. Slots are merged into junctions by (a) succ/pred connectivity
             (``a`` succ ``b`` -> a.end and b.start are the same junction) and
             (b) a distance fallback (slots within ``junction_merge_eps``,
             excluding a lane's own two endpoints).
          3. If #junctions > N, keep the top-N by degree (#incident endpoints).
          4. Hungarian-match the N predicted node slots to the kept junctions
             on position (L1).

        Each GT lane's target edge is then ``(node(start_junction),
        node(end_junction))`` -- so lanes meeting at a junction *share the same
        predicted node index* (structural sharing), not just coincide.

        Returns a dict of index tensors (``None`` where empty):
            lane_b, lane_global, lane_snode, lane_dnode  (representable lanes)
            nm_b, nm_node, nm_jpos                        (node<->junction matches)
        """
        device = x_lane_states.device
        node_xy = dec['node_xy']
        batch_size, n = dec['node_exist_logits'].shape
        eps = float(self.cfg.junction_merge_eps)

        starts = x_lane_states[:, 0, :].detach().cpu().numpy()    # (N_lanes, 2)
        ends = x_lane_states[:, -1, :].detach().cpu().numpy()
        lb = lane_batch.detach().cpu().numpy()
        node_xy_cpu = node_xy.detach().cpu().numpy()             # (B, N, 2)

        counts = np.bincount(lb, minlength=batch_size)
        offsets = np.zeros(batch_size, dtype=np.int64)
        offsets[1:] = np.cumsum(counts)[:-1]

        if x_lane_conn.shape[1] > SUCC_CONN_INDEX:
            succ_mask = (x_lane_conn[:, SUCC_CONN_INDEX] > 0.5).detach().cpu().numpy()
        else:
            succ_mask = np.zeros(l2l_edge_index.shape[1], dtype=bool)
        l2l_cpu = l2l_edge_index.detach().cpu().numpy()
        succ_src = l2l_cpu[0][succ_mask]
        succ_dst = l2l_cpu[1][succ_mask]
        succ_src_scene = lb[succ_src] if succ_src.shape[0] > 0 else succ_src

        lane_b, lane_global, lane_s, lane_d = [], [], [], []
        nm_b, nm_node, nm_jpos = [], [], []

        for b in range(batch_size):
            m = int(counts[b])
            if m == 0:
                continue
            g0 = int(offsets[b])
            coords = np.empty((2 * m, 2), dtype=starts.dtype)
            coords[0::2] = starts[g0:g0 + m]   # slot 2l   = start of lane l
            coords[1::2] = ends[g0:g0 + m]     # slot 2l+1 = end of lane l

            rows, cols = [], []
            # (a) succ connectivity unions: end(a) <-> start(b)
            sc = succ_src_scene == b
            a_loc = succ_src[sc] - g0
            d_loc = succ_dst[sc] - g0
            for ai, di in zip(a_loc.tolist(), d_loc.tolist()):
                if 0 <= ai < m and 0 <= di < m:
                    rows.append(2 * ai + 1)
                    cols.append(2 * di)
            # (b) distance fallback unions (exclude a lane's own two endpoints)
            if eps > 0 and m > 1:
                d = np.abs(coords[:, None, :] - coords[None, :, :]).sum(-1)  # (2m, 2m) L1
                lane_of = np.arange(2 * m) // 2
                iu = np.triu_indices(2 * m, 1)
                close = (d[iu] < eps) & (lane_of[iu[0]] != lane_of[iu[1]])
                rows += iu[0][close].tolist()
                cols += iu[1][close].tolist()

            if len(rows) > 0:
                adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(2 * m, 2 * m))
            else:
                adj = sp.coo_matrix(([], ([], [])), shape=(2 * m, 2 * m))
            num_j, labels = connected_components(adj, directed=False)

            jpos = np.zeros((num_j, 2), dtype=np.float64)
            deg = np.bincount(labels, minlength=num_j)
            for j in range(num_j):
                jpos[j] = coords[labels == j].mean(0)

            # truncate to top-N junctions by degree if needed
            if num_j > n:
                keep = np.argsort(-deg)[:n]
                remap = -np.ones(num_j, dtype=np.int64)
                remap[keep] = np.arange(n)
                labels = remap[labels]
                jpos = jpos[keep]
                num_kept = n
            else:
                num_kept = num_j

            # node slot <-> junction Hungarian on position (L1)
            nx = node_xy_cpu[b]                                 # (N, 2)
            cost = np.abs(nx[:, None, :] - jpos[None, :, :]).sum(-1)  # (N, num_kept)
            r_node, c_jct = linear_sum_assignment(cost)
            node_for_jct = -np.ones(num_kept, dtype=np.int64)
            node_for_jct[c_jct] = r_node
            for j in range(num_kept):
                nm_b.append(b)
                nm_node.append(int(node_for_jct[j]))
                nm_jpos.append(jpos[j])

            # lane -> target edge nodes (skip dropped or self-loop junctions)
            for l in range(m):
                sj = int(labels[2 * l])
                ej = int(labels[2 * l + 1])
                if sj < 0 or ej < 0 or sj == ej:
                    continue
                lane_b.append(b)
                lane_global.append(g0 + l)
                lane_s.append(int(node_for_jct[sj]))
                lane_d.append(int(node_for_jct[ej]))

        def _t(lst, dt=torch.long):
            if len(lst) == 0:
                return None
            return torch.as_tensor(np.asarray(lst), device=device, dtype=dt)

        return {
            'lane_b': _t(lane_b), 'lane_global': _t(lane_global),
            'lane_snode': _t(lane_s), 'lane_dnode': _t(lane_d),
            'nm_b': _t(nm_b), 'nm_node': _t(nm_node),
            'nm_jpos': _t(nm_jpos, dt=node_xy.dtype),
        }

    def _graph_loss_gt_nodes(self, dec, x_lane_states, lane_batch, l2l_edge_index, x_lane_conn):
        """Node-supervised graph loss (method B), enforcing structural node sharing.

        Differs from :meth:`_graph_loss` in that nodes are matched to GT
        *junctions* (so connected lanes share a node index) and each lane's
        target edge is determined by that node matching rather than an
        independent per-lane edge assignment.
        """
        device = x_lane_states.device
        dtype = x_lane_states.dtype
        num_points = x_lane_states.shape[1]
        basis = cubic_bezier_basis(num_points, device, dtype)

        node_xy = dec['node_xy']
        edge_ctrl = dec['edge_ctrl']
        edge_logits = dec['edge_exist_logits']
        node_logits = dec['node_exist_logits']
        batch_size, n = node_logits.shape

        mt = self._match_nodes(dec, x_lane_states, lane_batch, l2l_edge_index, x_lane_conn)

        # ---- node position loss + node existence ------------------------ #
        node_target = torch.zeros(batch_size, n, device=device, dtype=dtype)
        if mt['nm_b'] is not None:
            node_target[mt['nm_b'], mt['nm_node']] = 1.0
            node_pos_loss = (node_xy[mt['nm_b'], mt['nm_node']] - mt['nm_jpos']).abs().mean()
            n_pos = mt['nm_b'].shape[0]
        else:
            node_pos_loss = torch.tensor(0.0, device=device, dtype=dtype)
            n_pos = 0
        node_pos_weight = torch.tensor(
            max(batch_size * n - n_pos, 1) / max(n_pos, 1), device=device, dtype=dtype)
        node_exist_loss = F.binary_cross_entropy_with_logits(
            node_logits, node_target, pos_weight=node_pos_weight)

        # ---- edge target (node-indexed) + polyline regression ----------- #
        offdiag = ~torch.eye(n, dtype=torch.bool, device=device)
        edge_target = torch.zeros(batch_size, n, n, device=device, dtype=dtype)
        if mt['lane_b'] is not None:
            lb_, ls_, ld_ = mt['lane_b'], mt['lane_snode'], mt['lane_dnode']
            edge_target[lb_, ls_, ld_] = 1.0
            p0 = node_xy[lb_, ls_]
            p3 = node_xy[lb_, ld_]
            ctrl = edge_ctrl[lb_, ls_, ld_]            # (T, 4) offsets
            p1 = p0 + ctrl[:, 0:2]
            p2 = p3 + ctrl[:, 2:4]
            ctrl4 = torch.stack([p0, p1, p2, p3], dim=1)  # (T, 4, 2)
            poly = torch.einsum('pc,tcd->tpd', basis, ctrl4)  # (T, P, 2)
            reg_loss = (poly - x_lane_states[mt['lane_global']]).abs().mean()
            num_match = lb_.shape[0]
        else:
            reg_loss = torch.tensor(0.0, device=device, dtype=dtype)
            num_match = 0

        edge_logits_off = edge_logits[:, offdiag]   # (B, E)
        edge_target_off = edge_target[:, offdiag]   # (B, E)
        edge_pos_weight = torch.tensor(
            max(batch_size * edge_logits_off.shape[1] - num_match, 1) / max(num_match, 1),
            device=device, dtype=dtype)
        edge_exist_loss = F.binary_cross_entropy_with_logits(
            edge_logits_off, edge_target_off, pos_weight=edge_pos_weight)

        return {
            'lane_reg_loss': reg_loss,
            'endpoint_loss': node_pos_loss,   # node<->junction position supervision
            'edge_exist_loss': edge_exist_loss,
            'node_exist_loss': node_exist_loss,
            'continuity_loss': torch.tensor(0.0, device=device, dtype=dtype),
        }

    # ------------------------------------------------------------------ #
    # Hungarian matching (shared by the graph loss and reconstruction)   #
    # ------------------------------------------------------------------ #
    def _match(self, dec, x_lane_states, lane_batch):
        """Match predicted directed-edge bezier polylines to GT lane polylines.

        Returns a dict of intermediate tensors shared by :meth:`_graph_loss` and
        :meth:`reconstruct_lanes`. The match indices (``mb_idx``/``msel``/``mgt``)
        are ``None`` when the batch has no GT lanes.

        Args:
            dec: decoder output dict.
            x_lane_states: ``(N_lanes, P, 2)`` GT lane polylines.
            lane_batch: ``(N_lanes,)`` scene index per lane.
        """
        device = x_lane_states.device
        num_points = x_lane_states.shape[1]
        basis = cubic_bezier_basis(num_points, device, x_lane_states.dtype)

        node_xy = dec['node_xy']                  # (B, N, 2)
        edge_ctrl = dec['edge_ctrl']              # (B, N, N, 4)
        edge_exist_logits = dec['edge_exist_logits']  # (B, N, N)
        batch_size, n = dec['node_exist_logits'].shape

        # ---- shared off-diagonal candidate-edge indexing ---------------- #
        offdiag = ~torch.eye(n, dtype=torch.bool, device=device)  # (N, N)
        cand_idx = offdiag.flatten().nonzero(as_tuple=False).squeeze(1)  # (E,)
        E = cand_idx.shape[0]
        row_idx = torch.div(cand_idx, n, rounding_mode='floor')  # (E,) source node i
        col_idx = cand_idx % n                                   # (E,) dest node j

        # per-candidate-edge tensors, batched: (B, E, ...)
        b_ar = torch.arange(batch_size, device=device).unsqueeze(1)  # (B, 1)
        start_xy = node_xy[:, row_idx, :]   # (B, E, 2) == P0
        end_xy = node_xy[:, col_idx, :]     # (B, E, 2) == P3
        ctrl_flat = edge_ctrl.reshape(batch_size, n * n, 4)[:, cand_idx, :]  # (B, E, 4)
        logits_flat = edge_exist_logits.reshape(batch_size, n * n)[:, cand_idx]  # (B, E)

        # ---- pad GT lanes to a dense (B, M, P, 2) tensor ---------------- #
        gt_dense, gt_mask = to_dense_batch(x_lane_states, lane_batch, batch_size=batch_size)  # (B,M,P,2),(B,M)
        m_max = gt_dense.shape[1]
        counts = gt_mask.sum(dim=1)  # (B,) GT lanes per scene

        gt_start = gt_dense[:, :, 0, :]   # (B, M, 2)
        gt_end = gt_dense[:, :, -1, :]    # (B, M, 2)

        # ---- cheap endpoint pre-filter (batched) ------------------------ #
        # keep the top-k candidate edges (per scene) whose endpoints are
        # closest to some GT lane's endpoints, avoiding the O(N^2 * P) tensor.
        proxy = torch.cdist(start_xy, gt_start, p=1) + torch.cdist(end_xy, gt_end, p=1)  # (B, E, M)
        proxy = proxy.masked_fill(~gt_mask.unsqueeze(1), float('inf'))
        topk = min(E, max(self.cfg.matching_topk, 4 * int(m_max)))
        sel = torch.topk(proxy.amin(dim=2), k=topk, largest=False, dim=1).indices  # (B, k)

        # ---- resolve bezier polylines for selected edges (batched) ------ #
        sel_start = start_xy[b_ar, sel]            # (B, k, 2)  P0
        sel_end = end_xy[b_ar, sel]                # (B, k, 2)  P3
        sel_ctrl = ctrl_flat[b_ar, sel]            # (B, k, 4)  offsets (delta1, delta2)
        # inner control points are offsets relative to the endpoints
        p1 = sel_start + sel_ctrl[..., 0:2]
        p2 = sel_end + sel_ctrl[..., 2:4]
        ctrl4 = torch.stack([sel_start, p1, p2, sel_end], dim=2)  # (B, k, 4, 2)
        poly_sel = torch.einsum('pc,bkcd->bkpd', basis, ctrl4)  # (B, k, P, 2)

        # ---- matching cost on the reduced candidate set (batched) ------- #
        poly_flat2 = poly_sel.reshape(batch_size, topk, num_points * 2)
        gt_flat2 = gt_dense.reshape(batch_size, m_max, num_points * 2)
        reg_cost = torch.cdist(poly_flat2, gt_flat2, p=1) / (num_points * 2)  # (B, k, M)
        cls_cost = -torch.sigmoid(logits_flat[b_ar, sel]).unsqueeze(2)        # (B, k, 1)
        cost = self.cfg.cost_reg_weight * reg_cost + self.cfg.cost_cls_weight * cls_cost  # (B, k, M)
        cost = cost.masked_fill(~gt_mask.unsqueeze(1), 1e6)

        # ---- Hungarian assignment per scene (cheap CPU loop) ------------ #
        cost_cpu = cost.detach().cpu().numpy()
        counts_cpu = counts.cpu().tolist()
        mb_list, msel_list, mgt_list = [], [], []
        for b in range(batch_size):
            mb = int(counts_cpu[b])
            if mb == 0:
                continue
            row, gcol = linear_sum_assignment(cost_cpu[b][:, :mb])
            mb_list.append(np.full(row.shape[0], b))
            msel_list.append(row)
            mgt_list.append(gcol)

        if len(mb_list) > 0:
            mb_idx = torch.as_tensor(np.concatenate(mb_list), device=device, dtype=torch.long)
            msel = torch.as_tensor(np.concatenate(msel_list), device=device, dtype=torch.long)  # local in k
            mgt = torch.as_tensor(np.concatenate(mgt_list), device=device, dtype=torch.long)     # local in M
            matched_cand = sel[mb_idx, msel]  # (T,) index into E
        else:
            mb_idx = msel = mgt = matched_cand = None

        return {
            'n': n, 'E': E, 'batch_size': batch_size, 'm_max': m_max, 'counts': counts,
            'row_idx': row_idx, 'col_idx': col_idx, 'logits_flat': logits_flat,
            'poly_sel': poly_sel, 'gt_dense': gt_dense,
            'mb_idx': mb_idx, 'msel': msel, 'mgt': mgt, 'matched_cand': matched_cand,
        }

    # ------------------------------------------------------------------ #
    # graph (node/edge) matching loss                                    #
    # ------------------------------------------------------------------ #
    def _graph_loss(self, dec, x_lane_states, lane_batch, l2l_edge_index, x_lane_conn):
        """Hungarian-matched node-edge graph loss.

        Args:
            dec: decoder output dict.
            x_lane_states: ``(N_lanes, P, 2)`` GT lane polylines.
            lane_batch: ``(N_lanes,)`` scene index per lane.
            l2l_edge_index: ``(2, E_lane)`` decoder lane-to-lane edges (complete
                graph per scene).
            x_lane_conn: ``(E_lane, num_conn_types)`` one-hot connection types.
        """
        device = x_lane_states.device
        dtype = x_lane_states.dtype

        mt = self._match(dec, x_lane_states, lane_batch)
        n, E, batch_size, m_max = mt['n'], mt['E'], mt['batch_size'], mt['m_max']
        counts, row_idx, col_idx = mt['counts'], mt['row_idx'], mt['col_idx']
        logits_flat = mt['logits_flat']
        mb_idx, msel, mgt, matched_cand = mt['mb_idx'], mt['msel'], mt['mgt'], mt['matched_cand']
        node_xy = dec['node_xy']
        node_exist_logits = dec['node_exist_logits']

        # regression loss on matched polylines
        if matched_cand is not None:
            reg_loss = (mt['poly_sel'][mb_idx, msel] - mt['gt_dense'][mb_idx, mgt]).abs().mean()
            num_match = mb_idx.shape[0]

            # ---- explicit endpoint loss --------------------------------- #
            # Directly pull each matched edge's source/dest node onto the GT
            # lane's start/end point. This is NOT diluted by the interior
            # polyline points, giving the node positions a strong gradient.
            gt_match = mt['gt_dense'][mb_idx, mgt]              # (T, P, 2)
            node_src = node_xy[mb_idx, row_idx[matched_cand]]  # (T, 2) predicted P0
            node_dst = node_xy[mb_idx, col_idx[matched_cand]]  # (T, 2) predicted P3
            endpoint_loss = ((node_src - gt_match[:, 0, :]).abs().mean()
                             + (node_dst - gt_match[:, -1, :]).abs().mean())
        else:
            reg_loss = torch.tensor(0.0, device=device, dtype=dtype)
            endpoint_loss = torch.tensor(0.0, device=device, dtype=dtype)
            num_match = 0

        # ---- edge existence BCE (batched, balanced) --------------------- #
        edge_target = torch.zeros(batch_size, E, device=device, dtype=dtype)
        if matched_cand is not None:
            edge_target[mb_idx, matched_cand] = 1.0
        edge_pos_weight = torch.tensor(
            max(batch_size * E - num_match, 1) / max(num_match, 1), device=device, dtype=dtype)
        edge_exist_loss = F.binary_cross_entropy_with_logits(
            logits_flat, edge_target, pos_weight=edge_pos_weight)

        # ---- node existence BCE (active iff endpoint of a matched edge) -- #
        node_target = torch.zeros(batch_size, n, device=device, dtype=dtype)
        if matched_cand is not None:
            node_target[mb_idx, row_idx[matched_cand]] = 1.0
            node_target[mb_idx, col_idx[matched_cand]] = 1.0
        n_match = int(node_target.sum().item())
        node_pos_weight = torch.tensor(
            max(batch_size * n - n_match, 1) / max(n_match, 1), device=device, dtype=dtype)
        node_exist_loss = F.binary_cross_entropy_with_logits(
            node_exist_logits, node_target, pos_weight=node_pos_weight)

        # ---- optional succ/pred endpoint-continuity (node sharing) ------ #
        continuity_loss = torch.tensor(0.0, device=device, dtype=dtype)
        succ_conn = (x_lane_conn[:, SUCC_CONN_INDEX] > 0.5
                     if x_lane_conn.shape[1] > SUCC_CONN_INDEX else None)
        if (self.cfg.continuity_weight > 0 and matched_cand is not None and succ_conn is not None):
            # GT lane -> matched candidate edge id (per scene, dense)
            matched_edge_full = torch.full((batch_size, m_max), -1, device=device, dtype=torch.long)
            matched_edge_full[mb_idx, mgt] = matched_cand

            # global lane id -> local index within its scene
            offsets = torch.zeros(batch_size, device=device, dtype=torch.long)
            offsets[1:] = torch.cumsum(counts, dim=0)[:-1]
            g2l = torch.arange(lane_batch.shape[0], device=device) - offsets[lane_batch]

            e_mask = succ_conn  # (E_lane,) succ edges (a -> b means a succ b)
            if e_mask.any():
                src_g = l2l_edge_index[0][e_mask]
                dst_g = l2l_edge_index[1][e_mask]
                scene = lane_batch[src_g]
                a_local = g2l[src_g]
                b_local = g2l[dst_g]
                edge_a = matched_edge_full[scene, a_local]
                edge_b = matched_edge_full[scene, b_local]
                valid = (edge_a >= 0) & (edge_b >= 0)
                if valid.any():
                    scene = scene[valid]
                    a_end_node = col_idx[edge_a[valid]]    # end node of lane a
                    b_start_node = row_idx[edge_b[valid]]  # start node of lane b
                    continuity_loss = (node_xy[scene, a_end_node]
                                       - node_xy[scene, b_start_node]).abs().mean()

        return {
            'lane_reg_loss': reg_loss,
            'endpoint_loss': endpoint_loss,
            'edge_exist_loss': edge_exist_loss,
            'node_exist_loss': node_exist_loss,
            'continuity_loss': continuity_loss,
        }

    # ------------------------------------------------------------------ #
    # reconstruction (for visualization)                                 #
    # ------------------------------------------------------------------ #
    def reconstruct_lanes(self, dec, x_lane_states, lane_batch, l2l_edge_index=None, x_lane_conn=None):
        """Reconstruct per-lane polylines aligned 1:1 to the data's lane tokens.

        Each GT lane is shown as the predicted bezier edge it is assigned to,
        giving a ``(N_lanes, P, 2)`` tensor in the same ordering as
        ``x_lane_states`` so it can be fed to the existing ``visualize_batch``.
        Uses node-junction matching when ``use_gt_node_matching`` is set,
        otherwise the per-lane edge Hungarian.
        """
        device = x_lane_states.device
        lane_samples = x_lane_states.clone()  # fallback = GT for any unmatched lane

        if getattr(self.cfg, 'use_gt_node_matching', False) and l2l_edge_index is not None:
            mt = self._match_nodes(dec, x_lane_states, lane_batch, l2l_edge_index, x_lane_conn)
            if mt['lane_b'] is not None:
                basis = cubic_bezier_basis(x_lane_states.shape[1], device, x_lane_states.dtype)
                lb_, ls_, ld_ = mt['lane_b'], mt['lane_snode'], mt['lane_dnode']
                p0 = dec['node_xy'][lb_, ls_]
                p3 = dec['node_xy'][lb_, ld_]
                ctrl = dec['edge_ctrl'][lb_, ls_, ld_]
                p1 = p0 + ctrl[:, 0:2]
                p2 = p3 + ctrl[:, 2:4]
                ctrl4 = torch.stack([p0, p1, p2, p3], dim=1)
                poly = torch.einsum('pc,tcd->tpd', basis, ctrl4)
                lane_samples = lane_samples.index_copy(0, mt['lane_global'], poly)
            return lane_samples

        mt = self._match(dec, x_lane_states, lane_batch)
        if mt['matched_cand'] is not None:
            mb_idx, msel, mgt = mt['mb_idx'], mt['msel'], mt['mgt']
            offsets = torch.zeros(mt['batch_size'], device=device, dtype=torch.long)
            offsets[1:] = torch.cumsum(mt['counts'], dim=0)[:-1]
            global_idx = offsets[mb_idx] + mgt  # GT lane index in flat ordering
            lane_samples = lane_samples.index_copy(0, global_idx, mt['poly_sel'][mb_idx, msel])
        return lane_samples

    def reconstruct(self, data):
        """Encode -> decode -> match; returns normalized agent states and the
        per-lane reconstructed polylines aligned to ``data`` for visualization.
        """
        _, lane_batch, _ = get_batches(data)
        feats = get_features(data)
        x_lane_states, x_lane_conn = feats[4], feats[6]
        _, l2l_edge_index, _ = get_edge_indices(data)
        dec = self.forward(data)
        lane_samples = self.reconstruct_lanes(dec, x_lane_states, lane_batch, l2l_edge_index, x_lane_conn)
        return dec['agent_states_pred'], lane_samples, dec['agent_types_pred'], dec['lane_cond_dis_prob']

    def loss(self, data):
        """Compute the autoencoder loss for a batch of data."""
        agent_batch, lane_batch, lane_conn_batch = get_batches(data)
        x_agent, x_agent_states, x_agent_types, x_lane, x_lane_states, x_lane_types, x_lane_conn = get_features(data)
        a2a_edge_index, l2l_edge_index, l2a_edge_index = get_edge_indices(data)
        (a2a_edge_index_encoder, l2l_edge_index_encoder, l2a_edge_index_encoder,
         l2q_edge_index_encoder, x_lane_conn_encoder) = get_encoder_edge_indices(data)

        agent_mu, lane_mu, agent_log_var, lane_log_var, lane_cond_dis_logits, lane_cond_dis_prob = self.encoder(
            x_agent,
            x_lane,
            x_lane_conn_encoder,
            a2a_edge_index_encoder,
            l2l_edge_index_encoder,
            l2a_edge_index_encoder,
            l2q_edge_index_encoder,
            agent_batch)

        agent_latents = reparameterize(agent_mu, agent_log_var)
        lane_latents = reparameterize(lane_mu, lane_log_var)

        dec = self.decoder(
            agent_latents,
            lane_latents,
            lane_batch,
            a2a_edge_index,
            l2l_edge_index,
            l2a_edge_index)

        lg_type = data['lg_type']
        lane_cond_dis = data['num_lanes_after_origin']

        # ---- conditional lane distribution loss (unchanged) ------------- #
        ce_loss = nn.CrossEntropyLoss(reduction='none')
        lane_cond_dis_loss = ce_loss(lane_cond_dis_logits, lane_cond_dis)
        partition_mask = lg_type == NON_PARTITIONED
        assert torch.all(lane_cond_dis[partition_mask] == 0)
        lane_cond_dis_loss[partition_mask] = 0

        # ---- agent losses (unchanged) ----------------------------------- #
        agent_loss = self.agent_loss_fn(dec['agent_states_pred'], x_agent_states, agent_batch)
        agent_type_loss = self.agent_type_loss_fn(dec['agent_types_logits'], x_agent_types, agent_batch)

        # ---- kl losses (unchanged) -------------------------------------- #
        agent_kl_loss = self.kl_loss_fn(agent_mu, agent_log_var, agent_batch)
        lane_kl_loss = self.kl_loss_fn(lane_mu, lane_log_var, lane_batch)
        kl_loss = agent_kl_loss + lane_kl_loss

        # ---- bezier lane-graph losses ----------------------------------- #
        if getattr(self.cfg, 'use_gt_node_matching', False):
            # method B: GT-junction node matching -> structural node sharing
            graph_losses = self._graph_loss_gt_nodes(dec, x_lane_states, lane_batch, l2l_edge_index, x_lane_conn)
        else:
            # coordinate-continuity only (per-lane edge Hungarian)
            graph_losses = self._graph_loss(dec, x_lane_states, lane_batch, l2l_edge_index, x_lane_conn)

        loss = (agent_loss
                + agent_type_loss
                + self.cfg.kl_weight * kl_loss
                + self.cfg.lane_weight * graph_losses['lane_reg_loss']
                + self.cfg.endpoint_weight * graph_losses['endpoint_loss']
                + self.cfg.edge_exist_weight * graph_losses['edge_exist_loss']
                + self.cfg.node_exist_weight * graph_losses['node_exist_loss']
                + self.cfg.continuity_weight * graph_losses['continuity_loss'])

        loss = loss + self.cfg.cond_dis_weight * lane_cond_dis_loss
        lane_cond_dis_loss_log = lane_cond_dis_loss[~partition_mask].mean().detach()

        lane_cond_dis_pred_filtered = torch.argmax(lane_cond_dis_prob[~partition_mask], dim=-1)
        lane_cond_dis_acc = (torch.abs(lane_cond_dis_pred_filtered - lane_cond_dis[~partition_mask]) <= 3).float().mean()

        loss_dict = {
            'loss': loss.mean(),
            'agent_loss': agent_loss.mean().detach(),
            'agent_type_loss': agent_type_loss.mean().detach(),
            'lane_reg_loss': graph_losses['lane_reg_loss'].detach(),
            'endpoint_loss': graph_losses['endpoint_loss'].detach(),
            'edge_exist_loss': graph_losses['edge_exist_loss'].detach(),
            'node_exist_loss': graph_losses['node_exist_loss'].detach(),
            'continuity_loss': graph_losses['continuity_loss'].detach(),
            'kl_loss': kl_loss.mean().detach(),
            'lane_cond_dis_loss': lane_cond_dis_loss_log,
            'lane_cond_dis_acc': lane_cond_dis_acc,
        }
        return loss_dict

    # ------------------------------------------------------------------ #
    # inference helpers                                                  #
    # ------------------------------------------------------------------ #
    def forward_encoder(self, data, return_stats=False, return_lane_embeddings=False):
        """Forward pass through the (unchanged) encoder."""
        agent_batch, lane_batch, lane_conn_batch = get_batches(data)
        x_agent, x_agent_states, x_agent_types, x_lane, x_lane_states, x_lane_types, x_lane_conn = get_features(data)
        (a2a_edge_index_encoder, l2l_edge_index_encoder, l2a_edge_index_encoder,
         l2q_edge_index_encoder, x_lane_conn_encoder) = get_encoder_edge_indices(data)

        encoder_output = self.encoder(
            x_agent,
            x_lane,
            x_lane_conn_encoder,
            a2a_edge_index_encoder,
            l2l_edge_index_encoder,
            l2a_edge_index_encoder,
            l2q_edge_index_encoder,
            agent_batch,
            return_lane_embeddings)

        if return_lane_embeddings:
            return encoder_output
        agent_mu, lane_mu, agent_log_var, lane_log_var, lane_cond_dis_logits, lane_cond_dis_prob = encoder_output

        if return_stats:
            return agent_mu, lane_mu, agent_log_var, lane_log_var

        agent_latents = reparameterize(agent_mu, agent_log_var)
        lane_latents = reparameterize(lane_mu, lane_log_var)
        return agent_latents, lane_latents, lane_cond_dis_prob

    def forward_decoder(self, agent_latents, lane_latents, data):
        """Forward pass through the graph decoder; returns the raw decoder dict."""
        agent_batch, lane_batch, lane_conn_batch = get_batches(data)
        a2a_edge_index, l2l_edge_index, l2a_edge_index = get_edge_indices(data)
        return self.decoder(
            agent_latents,
            lane_latents,
            lane_batch,
            a2a_edge_index,
            l2l_edge_index,
            l2a_edge_index)

    def decode_graph_to_polylines(self, dec, edge_exist_threshold=0.5):
        """Resolve the predicted graph into per-scene lane polylines.

        Returns a list (length B) of dicts with ``node_xy``, ``polylines``
        (``(K, P, 2)``) and the ``(src, dst)`` node indices of each kept edge.
        Intended for visualization / downstream use.
        """
        node_xy = dec['node_xy']
        edge_ctrl = dec['edge_ctrl']
        edge_exist_logits = dec['edge_exist_logits']
        node_exist_logits = dec['node_exist_logits']
        batch_size, n = node_exist_logits.shape
        device = node_xy.device
        num_points = self.cfg.num_points_per_lane
        basis = cubic_bezier_basis(num_points, device, node_xy.dtype)
        offdiag = ~torch.eye(n, dtype=torch.bool, device=device)

        out = []
        for b in range(batch_size):
            poly_b = resolve_bezier_edges(node_xy[b], edge_ctrl[b], basis)  # (N, N, P, 2)
            keep = (torch.sigmoid(edge_exist_logits[b]) > edge_exist_threshold) & offdiag
            src, dst = keep.nonzero(as_tuple=True)
            out.append({
                'node_xy': node_xy[b],
                'node_exist_prob': torch.sigmoid(node_exist_logits[b]),
                'polylines': poly_b[src, dst],
                'edge_src': src,
                'edge_dst': dst,
            })
        return out

    def forward(self, data, return_latents=False, return_lane_embeddings=False):
        encoder_output = self.forward_encoder(
            data, return_stats=return_latents, return_lane_embeddings=return_lane_embeddings)

        if return_latents or return_lane_embeddings:
            return encoder_output
        agent_latents, lane_latents, lane_cond_dis_prob = encoder_output

        dec = self.forward_decoder(agent_latents, lane_latents, data)
        dec['lane_cond_dis_prob'] = lane_cond_dis_prob
        return dec
