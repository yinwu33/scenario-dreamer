"""ctrl_sim planner: the frozen CtRL-Sim decision transformer as a rollout role.

CtRL-Sim (Rowe et al.) is a return-conditioned transformer over (state, action,
return-to-go) tokens. Tilting the predicted RTG downward makes its agents drive
badly on purpose, which is what makes it the paper's BEHAVIOR-driven adversarial
baseline: put it in the ``adv`` role and the inserted agent misbehaves, against
which our claim is that a well-PLACED ordinary agent is more critical.

Two properties make it fit this simulator where a history-conditioned model would
not. It cold-starts: the context is zeroed at t=0 and accumulated during the
rollout, so a generated initial scene with no past is a valid starting point. And
it needs no privileged map: ``get_normalized_lanes_in_fov`` wants plain
centreline geometry, which is all ``SimScene`` has.

DYNAMICS EXCEPTION. Every other planner emits an index into the shared 7x13
accel/steer table and is integrated by ``SimScene.step_dynamics``. CtRL-Sim emits
a k-disks token -- a discrete rigid motion of the agent's box -- and is
integrated by ``forward_k_disks``, because that is how the published method moves
agents. Projecting its output onto the accel/steer table would no longer be
CtRL-Sim. The paper must state this exception wherever it claims every role
shares one integrator.

Agents this planner does NOT drive still need action tokens in the context, since
the model was trained on sequences where every modelled agent has one. Those are
recovered from the motion each agent actually made, with the same tokenizer,
which keeps the context faithful no matter which planner moved it.

Single-process only: ``sim.parallel`` shuttles a planner's gather through a
shared-memory buffer shaped for one flat observation matrix, and this planner's
input is a set of per-agent-centric buffers plus lanes. Score it with
``--workers 0``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..world import SimScene
from .base import Planner, PlanItem

# CtRL-Sim agent state layout:
# [pos_x, pos_y, vel_x, vel_y, heading, length, width, existence]
POS_X, POS_Y, VEL_X, VEL_Y, HEAD, LEN, WID, EXIST = range(8)
NUM_AGENT_STATES = 8
# [is_unset, is_vehicle, is_pedestrian, is_cyclist, is_other]
NUM_AGENT_TYPES = 5
# PufferDrive type id -> CtRL-Sim one-hot column.
PTYPE_TO_COLUMN = {1: 1, 2: 2, 3: 3}
MAX_RTG_VAL = 349


class CtRLSimPlanner(Planner):
    # Neural: one forward per agent-centric buffer, so the batch composition is
    # part of the computation. sim.parallel cannot carry this planner's gather,
    # and refuses to shard a rollout that uses it.
    batched_across_scenes = True

    def __init__(self, planner_cfg, *, role: str, device: str | None = None):
        super().__init__(planner_cfg, role=role, device=device)
        from hydra import compose, initialize_config_dir
        from models.ctrl_sim import CtRLSim
        from datasets.waymo.dataset_ctrl_sim import CtRLSimDataset
        from cfgs.config import CONFIG_PATH

        self.device = device or "cuda"
        self.tilt = float(self._require("tilt"))
        self.action_temperature = float(self._require("action_temperature"))
        self.predict_rtgs = bool(self._require("predict_rtgs"))
        ckpt = str(self._require("checkpoint"))

        self.model = CtRLSim.load_from_checkpoint(ckpt, map_location=self.device)
        self.model.to(self.device).eval()
        self.cfg_model = self.model.cfg.model
        self.cfg_dataset = self.model.cfg.dataset
        self.context = int(self.cfg_dataset.train_context_length)

        # The dataset object is used ONLY for its pure helpers and the k-disks
        # vocabulary; nothing is read from disk. split_name='val' matters: the
        # 'train' branch of select_closest_max_num_agents shuffles agent order.
        with initialize_config_dir(config_dir=str(CONFIG_PATH), version_base=None):
            root = compose(config_name="config", overrides=["model_name=ctrl_sim"])
        self.dset = CtRLSimDataset(root.ctrl_sim.dataset, split_name="val")
        self.vocab = np.asarray(self.dset.V)

    # ------------------------------------------------------------------ state
    def _ctx_for(self, sim: SimScene) -> dict:
        """This role's rolling CtRL-Sim context for ``sim`` (planner-internal)."""
        store = getattr(sim, "_ctrl_sim_ctx", None)
        if store is None:
            store = sim._ctrl_sim_ctx = {}
        ctx = store.get(self.role)
        if ctx is None or ctx["states"].shape[0] != sim.n:
            types = np.zeros((sim.n, NUM_AGENT_TYPES), dtype=np.float32)
            for i, p in enumerate(sim.ptype):
                types[i, PTYPE_TO_COLUMN[int(p)]] = 1.0
            ctx = {
                "states": np.zeros((sim.n, self.context, NUM_AGENT_STATES), dtype=np.float32),
                "actions": np.zeros((sim.n, self.context), dtype=np.float32),
                "rtgs": np.full((sim.n, self.context), float(MAX_RTG_VAL), dtype=np.float32),
                "types": types,
                "filled": 0,
                "prev_pose": None,
            }
            store[self.role] = ctx
        return ctx

    def _advance(self, ctx: dict) -> int:
        """Make room for this step and return the column to write it into.

        The buffers ARE the model's context window, so once it is full the whole
        window slides by one, which is the same slice
        ``CtRLSimBehaviourModel.get_motion_data`` takes out of its full-episode
        buffer. The vacated column is reset to the values a fresh step starts
        from, not to whatever scrolled off.
        """
        if ctx["filled"] < self.context:
            pos = ctx["filled"]
            ctx["filled"] += 1
            return pos
        for key in ("states", "actions", "rtgs"):
            ctx[key][:, :-1] = ctx[key][:, 1:]
        ctx["actions"][:, -1] = 0.0
        ctx["rtgs"][:, -1] = float(MAX_RTG_VAL)
        return self.context - 1

    def _current_states(self, sim: SimScene) -> np.ndarray:
        out = np.zeros((sim.n, NUM_AGENT_STATES), dtype=np.float32)
        out[:, POS_X], out[:, POS_Y] = sim.x, sim.y
        out[:, VEL_X], out[:, VEL_Y] = sim.vx, sim.vy
        out[:, HEAD] = sim.heading
        out[:, LEN], out[:, WID] = sim.length, sim.width
        out[:, EXIST] = (~sim.removed).astype(np.float32)
        return out

    def _tokenize_motion(self, prev: np.ndarray, cur: np.ndarray,
                         length: np.ndarray, width: np.ndarray) -> np.ndarray:
        """The k-disks token that best explains prev -> cur, for every agent.

        The encoding counterpart of ``forward_k_disks``: it lets agents driven by
        some OTHER planner still occupy the context with a real action token.
        Deterministic (argmin), because this describes an observed motion rather
        than sampling a new one.
        """
        from utils.k_disks_helpers import (
            get_local_state_transition,
            transform_box_corners_from_local_state,
            transform_box_corners_from_vocab,
        )

        half_l, half_w = length / 2.0, width / 2.0
        box_corners = np.stack(
            [
                np.stack([-half_l, -half_w], axis=-1),
                np.stack([-half_l, half_w], axis=-1),
                np.stack([half_l, half_w], axis=-1),
                np.stack([half_l, -half_w], axis=-1),
            ],
            axis=1,
        )  # [A, 4, 2]
        local = get_local_state_transition(current_state=prev, next_state=cur)
        target = transform_box_corners_from_local_state(box_corners, local)
        vocab_corners = transform_box_corners_from_vocab(box_corners, self.vocab)
        err = np.linalg.norm(vocab_corners - target[:, None, :, :], axis=-1).mean(2)
        return np.argmin(err, axis=1)

    # ------------------------------------------------------------------- plan
    def _motion_batches(self, sim: SimScene, ctx: dict, ids: np.ndarray, norm_t: int):
        """Agent-centric model inputs covering ``ids``, plus their index maps.

        Ported from ``CtRLSimBehaviourModel.get_motion_data``: the context is
        normalized to the ego, agents outside the field of view are masked out,
        and the driven agents are packed into buffers of at most
        ``max_num_agents`` until all of them are accounted for.
        """
        import copy

        from utils.data_container import CtRLSimData
        from utils.data_helpers import add_batch_dim
        from utils.geometry import normalize_agents
        from utils.lane_graph_helpers import resample_lanes_with_mask
        from utils.torch_helpers import from_numpy

        states = ctx["states"].copy()
        actions = ctx["actions"].copy()
        rtgs = ctx["rtgs"].copy()
        rtg_mask = states[:, :, EXIST].copy()
        timesteps = np.repeat(
            np.arange(self.context)[None, :, None], self.cfg_dataset.max_num_agents, 0
        )

        normalize_dict = {
            "center": states[0, norm_t, :2].copy(),
            "yaw": states[0, norm_t, HEAD].copy(),
        }
        agent_mask = self.dset.get_agent_mask(
            copy.deepcopy(states[:, :, : HEAD + 1]), normalize_dict
        )
        moving = np.ones(states.shape[0], dtype=bool)

        lanes_full = resample_lanes_with_mask(
            sim.lane_polylines,
            np.ones(sim.lane_polylines.shape[:2], dtype=bool),
            int(self.cfg_dataset.num_points_per_lane),
        )

        out, remaining = [], np.asarray(ids, dtype=np.int64)
        while len(remaining):
            (
                state_buf, type_buf, mask_buf, action_buf, rtg_buf, rtg_mask_buf,
                _, origin_idx, correspondence,
            ) = self.dset.select_closest_max_num_agents(
                states, ctx["types"], agent_mask, actions, rtgs, rtg_mask, moving,
                origin_agent_idx=0, timestep=norm_t, active_agents=remaining,
            )
            lanes, lanes_mask = self.dset.get_normalized_lanes_in_fov(
                lanes_full, normalize_dict
            )
            state_buf = normalize_agents(state_buf, normalize_dict)

            is_ego = np.zeros(len(state_buf), dtype=np.int64)
            is_ego[origin_idx] = 1
            is_ego = np.tile(is_ego[:, None, None], (1, self.context, 1))
            state_buf = np.concatenate(
                [state_buf[:, :, :-1], is_ego, state_buf[:, :, -1:]], axis=-1
            )
            state_buf[~mask_buf.astype(bool)] = 0
            rtg_mask_buf[~mask_buf.astype(bool)] = 0
            lanes = np.concatenate([lanes, lanes_mask[:, :, None]], axis=-1)

            data = CtRLSimData({
                "agent": from_numpy({
                    "agent_states": add_batch_dim(state_buf),
                    "agent_types": add_batch_dim(type_buf),
                    "actions": add_batch_dim(action_buf),
                    "rtgs": add_batch_dim(rtg_buf[:, :, None]),
                    "rtg_mask": add_batch_dim(rtg_mask_buf[:, :, None]),
                    "timesteps": add_batch_dim(timesteps),
                    "moving_agent_mask": add_batch_dim(moving[: self.cfg_dataset.max_num_agents]),
                }),
                "map": from_numpy({"road_points": add_batch_dim(lanes)}),
            })
            out.append((data, correspondence, origin_idx))
            remaining = np.setdiff1d(remaining, correspondence)
        return out

    def _tilt_logits(self) -> torch.Tensor:
        bins = self.tilt * np.linspace(0, 1, int(self.cfg_dataset.rtg_discretization))
        return torch.from_numpy(bins).to(self.device)

    @torch.no_grad()
    def plan(self, items: Sequence[PlanItem]) -> list:
        plans = []
        for sim, ids in items:
            ctx = self._ctx_for(sim)
            cur = self._current_states(sim)
            pos = self._advance(ctx)

            # Record what every agent actually did last step, whoever drove it.
            if ctx["prev_pose"] is not None:
                ctx["actions"][:, pos - 1] = self._tokenize_motion(
                    ctx["prev_pose"], cur[:, [POS_X, POS_Y, HEAD]], sim.length, sim.width
                )
            ctx["states"][:, pos] = cur
            ctx["prev_pose"] = cur[:, [POS_X, POS_Y, HEAD]].copy()

            ids = np.asarray(ids, dtype=np.int64)
            if len(ids) == 0:
                plans.append(np.zeros(0, dtype=np.int64))
                continue

            batches = self._motion_batches(sim, ctx, ids, pos)
            tokens = np.zeros(sim.n, dtype=np.int64)
            token_index = pos
            for data, correspondence, origin_idx in batches:
                data = data.to(self.device)
                if self.predict_rtgs:
                    rtg_logits = self.model(data, eval=True)["rtg_preds"]
                    tilt = self._tilt_logits()
                    for slot, agent in enumerate(correspondence):
                        if slot == origin_idx:
                            continue
                        logits = rtg_logits[0, slot, token_index].reshape(
                            int(self.cfg_dataset.rtg_discretization),
                            int(self.cfg_model.num_reward_components),
                        )
                        probs = F.softmax(logits[:, 0] + tilt, dim=0)
                        drawn = torch.multinomial(probs, 1)
                        data["agent"].rtgs[0, slot, token_index, 0] = drawn.item()

                action_logits = self.model(data, eval=True)["action_preds"]
                for slot, agent in enumerate(correspondence):
                    if slot == origin_idx:
                        continue
                    probs = F.softmax(
                        action_logits[0, slot, token_index] / self.action_temperature, dim=0
                    )
                    tokens[int(agent)] = int(torch.multinomial(probs, 1).item())
            plans.append(tokens[ids])
        return plans

    # ------------------------------------------------------------------ apply
    def apply(self, items: Sequence[PlanItem], plans: list) -> None:
        from utils.k_disks_helpers import forward_k_disks

        for (sim, ids), tokens in zip(items, plans):
            ids = np.asarray(ids, dtype=np.int64)
            if len(ids) == 0:
                continue
            # Same freeze semantics as SimScene.step_dynamics: agents that have
            # stopped at their goal or crashed do not move.
            moving = ~(sim.stopped | sim.crashed)[ids]
            sim.vx[ids[~moving]] = 0.0
            sim.vy[ids[~moving]] = 0.0
            idx = ids[moving]
            if len(idx) == 0:
                continue
            states = self._current_states(sim)[idx]
            nxt = forward_k_disks(
                states, np.asarray(tokens)[moving], self.vocab, sim.dt, states[:, EXIST]
            )
            sim.x[idx] = nxt[:, POS_X]
            sim.y[idx] = nxt[:, POS_Y]
            sim.vx[idx] = nxt[:, VEL_X]
            sim.vy[idx] = nxt[:, VEL_Y]
            sim.heading[idx] = nxt[:, HEAD]
            sim.heading_x[idx] = np.cos(sim.heading[idx])
            sim.heading_y[idx] = np.sin(sim.heading[idx])
