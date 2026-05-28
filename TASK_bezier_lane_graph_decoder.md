# Task Checkpoint: Bezier Lane Graph Decoder

## Context

Current file under discussion: `nn_modules/autoencoder_bezier.py`.

As of this checkpoint, `autoencoder_bezier.py` is effectively the same as `nn_modules/autoencoder.py`; it does not yet implement Bezier output or shared-node graph decoding.

The current autoencoder is lane-token based:

```text
input N_lanes lane tokens
encoder -> N_lanes lane latents
decoder -> N_lanes reconstructed lane polylines
```

Each lane is reconstructed independently at the final output head:

```python
lane_states_pred = self.pred_lane_states(lane_embeddings).reshape(
    x_lane.shape[0],
    cfg.num_points_per_lane,
    cfg.lane_attr,
)
```

This explains the observed issue: even though lane embeddings exchange information through lane-to-lane attention, the final lane geometry is still predicted per lane token, so connected lane endpoints can be misaligned.

## Current Tensor Shapes

Dataset feature extraction currently produces:

```text
x_agent_states: (N_agents, state_dim)
x_agent_types:  (N_agents, num_agent_types)
x_agent:        (N_agents, state_dim + num_agent_types)

x_lane_states:  (N_lanes, num_points_per_lane, lane_attr)
x_lane:         (N_lanes, num_points_per_lane * lane_attr + num_lane_types)
                or (N_lanes, num_points_per_lane * lane_attr) if no lane type

x_lane_conn:    (E_lane, lane_conn_attr)
```

Default relevant config values:

```text
hidden_dim = 512
agent_hidden_dim = 256
lane_conn_hidden_dim = 64
lane_latent_dim = 24
agent_latent_dim = 8
lane_attr = 2
state_dim = 7
num_agent_types = 3
```

Encoder output:

```text
agent_mu:             (N_agents, agent_latent_dim)
lane_mu:              (N_lanes, lane_latent_dim)
agent_log_var:        (N_agents, agent_latent_dim)
lane_log_var:         (N_lanes, lane_latent_dim)
lane_cond_dis_logits: (B, max_num_lanes + 1)
lane_cond_dis_prob:   (B, max_num_lanes + 1)
```

Decoder input:

```text
agent_latents: (N_agents, agent_latent_dim)
lane_latents:  (N_lanes, lane_latent_dim)
a2a_edge_index: (2, E_agent)
l2l_edge_index: (2, E_lane)
l2a_edge_index: (2, E_cross)
```

Current decoder output:

```text
agent_states_pred: (N_agents, state_dim)
agent_types_logits: (N_agents, num_agent_types)
agent_types_pred: (N_agents,)
lane_states_pred: (N_lanes, num_points_per_lane, lane_attr)
lane_types_logits: (N_lanes, num_lane_types) or None
lane_types_pred: (N_lanes,) or None
lane_conn_logits: (E_lane, lane_conn_attr)
lane_conn_pred: (E_lane, lane_conn_attr)
```

## Existing Lane Connection Information

The dataset already contains lane connection information.

`l2l_edge_index` is built as a complete graph over lanes, and `data['lane', 'to', 'lane'].type` stores the connection class.

Waymo connection types:

```text
none / pred / succ / left / right / self
```

NuPlan connection types:

```text
none / pred / succ / self
```

Important distinction:

The existing connection labels are lane-level relationships. They are not shared-node or junction-level labels.

## Main Design Conclusion

The immediate endpoint mismatch can be addressed without replacing the whole autoencoder:

```text
lane token -> Bezier control points -> resolve_bezier -> vectorized lane points
```

Then add a successor/predecessor endpoint continuity loss:

```text
|| end(src_lane) - start(dst_lane) ||
```

Only apply this loss on successor/predecessor edges, not on left/right/self/none.

This is the least invasive path because it keeps:

```text
encoder input unchanged
lane_latent shape unchanged
decoder input unchanged
AutoEncoder.forward output mostly unchanged
existing lane reconstruction loss usable
existing visualization/downstream code mostly compatible
```

## Minimal Bezier Decoder Plan

Modify only the lane geometry head first.

Replace:

```python
self.pred_lane_states = ResidualMLP(
    input_dim=cfg.hidden_dim,
    hidden_dim=cfg.hidden_dim,
    n_hidden=3,
    output_dim=cfg.num_points_per_lane * cfg.lane_attr,
)
```

with a Bezier control-point head:

```python
self.pred_lane_bezier_ctrl = ResidualMLP(
    input_dim=cfg.hidden_dim,
    hidden_dim=cfg.hidden_dim,
    n_hidden=3,
    output_dim=cfg.num_bezier_control_points * cfg.lane_attr,
)
```

For single cubic Bezier:

```text
num_bezier_control_points = 4
lane_bezier_ctrl: (N_lanes, 4, 2)
```

For piecewise cubic Bezier with `S` segments and shared endpoints:

```text
num_bezier_control_points = 3 * S + 1
lane_bezier_ctrl: (N_lanes, 3 * S + 1, 2)
```

Add differentiable torch-based `resolve_bezier`:

```text
input:  lane_bezier_ctrl (N_lanes, K, 2)
output: lane_states_pred (N_lanes, num_points_per_lane, 2)
```

For a single Bezier segment, use Bernstein basis:

```text
basis: (P, K)
lane_points = einsum("pk,nkd->npd", basis, ctrl)
```

Then keep returning `lane_states_pred` in the existing decoder tuple so existing loss and visualization still work.

## Endpoint Continuity Loss Plan

Add an auxiliary loss in `AutoEncoder.loss`.

Inputs available there:

```text
lane_states_pred: (N_lanes, P, 2)
x_lane_conn:      (E_lane, lane_conn_attr)
l2l_edge_index:   (2, E_lane)
lane_batch:       (N_lanes,)
lane_conn_batch:  (E_lane,)
```

For each `l2l` edge that represents successor/predecessor:

```text
src = l2l_edge_index[0]
dst = l2l_edge_index[1]
loss = L1(lane_states_pred[src, -1], lane_states_pred[dst, 0])
```

Need to confirm exact class index mapping for `pred` and `succ` from helper functions before coding. The existing code uses `get_lane_connection_type_onehot_waymo` and `get_lane_connection_type_onehot_nuplan` during preprocessing.

Potential config:

```yaml
bezier_num_segments: 1
endpoint_continuity_weight: <tune>
```

## Shared Node Graph Idea

A stronger long-term representation would be:

```text
shared_nodes: (N_shared_node, 2)
edge_params:  (N_shared_node, N_shared_node, 5)
```

where:

```text
edge_params[..., 0] = edge indicator / valid logit
edge_params[..., 1:5] = two Bezier control points
```

For edge `i -> j`, the cubic curve would be:

```text
P0 = shared_nodes[i]
P1 = ctrl_1(i, j)
P2 = ctrl_2(i, j)
P3 = shared_nodes[j]
```

This would enforce endpoint continuity structurally.

However, this is not a small modification to the current autoencoder because current data/model are lane-token based, not junction-node based.

Main blockers:

```text
No explicit GT shared-node/junction ids currently exist.
Need to infer shared nodes from lane endpoints and successor/predecessor connections.
Need lane-to-start-node and lane-to-end-node mapping.
Need matching if output nodes are unordered query slots.
Dense N_node x N_node output is sparse and O(N_node^2).
Left/right lane relationships are not directly derivable from endpoint node graph.
```

## Node Probability Idea

A fixed-slot node decoder could output:

```text
node_pred: (B, max_N_node, 3)
```

where:

```text
node_pred[..., :2] = xy
node_pred[..., 2]  = valid/existence logit
```

This handles variable node count by using fixed slots plus `valid_logit`.

But it introduces DETR-style set prediction:

```text
predicted node slots <-> GT shared nodes
```

This requires Hungarian matching or a deterministic GT node ordering. Hungarian matching is the more standard approach.

## Query-Based Node Decoder Idea

A cleaner graph-decoder architecture:

```text
lane latents / lane embeddings as memory
learnable node queries as query slots
cross attention -> node tokens
node_head -> (B, max_N_node, 3)
edge_pair_head(node_i, node_j) -> edge indicator + Bezier ctrl
```

This is similar to DETR.

It can be implemented either with dense padded tensors:

```text
memory: (B, max_N_lanes, hidden_dim)
query:  (B, max_N_node, hidden_dim)
```

or with PyG-style bipartite attention:

```text
lane nodes + node query tokens
edge_index: lane -> node_query
```

The PyG route fits the current code better because `AttentionLayer(..., bipartite=True)` already supports a lane-to-query pattern in the encoder.

Needed for PyG query branch:

```text
decoder must receive lane_batch
construct per-scene node query tokens
construct l2node_edge_index connecting lanes to node queries in same scene
run bipartite AttentionLayer
predict node xy + valid logit
```

## Does Query-Based Decoding Break The Autoencoder?

It changes the design assumption.

Current autoencoder:

```text
lane token i -> latent i -> reconstructed lane i
```

Query-based graph decoder:

```text
set of lane latents -> set of node query slots -> graph
```

So the decoder output is no longer one-to-one with lane latents.

This may affect:

```text
latent interpretability
latent cache assumptions
latent diffusion model design
downstream code expecting N_lanes reconstructed lane polylines
visualization code
loss design
```

It is not necessarily invalid, but it is a larger architecture change. It turns the model from indexed lane-token reconstruction into scene-graph reconstruction from lane-token memory.

## Recommended Next Steps

Short-term path:

1. Implement Bezier lane head in `autoencoder_bezier.py`.
2. Add differentiable `resolve_bezier`.
3. Keep decoder returning `lane_states_pred` with shape `(N_lanes, P, 2)`.
4. Add endpoint continuity loss for successor/predecessor lane edges.
5. Keep existing lane_conn prediction head unchanged.

Medium-term path:

1. Add an auxiliary shared-node branch without replacing lane reconstruction.
2. Build GT shared nodes from lane endpoints and successor/predecessor edges.
3. Add node valid/head prediction and Hungarian matching.
4. Use it only as auxiliary supervision initially.

Long-term path:

1. Replace lane-token output with query-based graph decoder.
2. Predict fixed-slot nodes plus pairwise directed edge Bezier parameters.
3. Resolve graph edges into vectorized lane polylines.
4. Redesign lane connection prediction, especially left/right.
5. Revisit latent diffusion compatibility.

## Unresolved Questions

1. What is the exact one-hot index mapping for lane connection types in current configs?
2. Should Bezier be single cubic or piecewise cubic?
3. Should endpoint continuity be soft loss only, or should connected endpoints be explicitly averaged/shared during resolve?
4. How should GT shared nodes be built robustly?
5. If using node slots, should node GT use Hungarian matching or deterministic ordering?
6. How should left/right lane relationships be represented in a node-edge Bezier graph?
7. If decoder becomes query-based, should the LDM still diffuse lane latents, or should the latent space also become graph/node based?
8. Should `autoencoder_bezier.py` be wired into `models/scenario_dreamer_autoencoder.py`, which currently imports `nn_modules.autoencoder.AutoEncoder`?

