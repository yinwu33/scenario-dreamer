import torch
import torch.nn as nn

import numpy as np
from utils.dit_ex_layers import FactorizedDiTBlock, FinalLayer, LabelEmbedder, TimestepEmbedder, get_1d_sincos_pos_embed_from_grid, TwoLayerResMLP
from utils.pyg_helpers import get_indices_within_scene


class DiT(nn.Module):

    def __init__(self, cfg):
        super(DiT, self).__init__()
        self.cfg = cfg
        self.cfg_model = self.cfg.model
        self.cfg_dataset = self.cfg.dataset


        self.emb_drop = nn.Dropout(self.cfg_model.dropout)

        # Condition on number of agents and lanes
        self.num_agents_embedder = LabelEmbedder(self.cfg_dataset.max_num_agents + 1, self.cfg_model.hidden_dim, 0)
        self.num_lanes_embedder = LabelEmbedder(self.cfg_dataset.max_num_lanes + 1, self.cfg_model.hidden_dim, 0)
        
        # Diffusion timestep embedding
        self.t_embedder = TimestepEmbedder(self.cfg_model.hidden_dim)
        # Used because agent embedding is smaller than lane embedding
        self.downsample_c = nn.Linear(self.cfg_model.hidden_dim, self.cfg_model.agent_hidden_dim)

        self.adv_latent_dim = self.cfg_model.agent_latent_dim

        # Embed agent, lane, and adversarial-agent latents
        self.lane_embedder = TwoLayerResMLP(self.cfg_model.lane_latent_dim, self.cfg_model.hidden_dim)
        self.agent_embedder = TwoLayerResMLP(self.cfg_model.agent_latent_dim, self.cfg_model.agent_hidden_dim)
        self.adv_embedder = TwoLayerResMLP(self.adv_latent_dim, self.cfg_model.agent_hidden_dim)

        # Optional conditioning: discretized labels embedded and added onto the
        # agent / adv streams' conditioning vectors. ``cond_dropout_prob`` (> 0)
        # gives every embedder a null index, used for the per-agent unconditional
        # dropout -- so uncontrolled agents (whose conditions we don't care about
        # at inference) can be left unconditioned. No classifier-free guidance
        # scaling; conditioning is either the real label or the null token.
        #   normal agent: [type, motion, goal_dist]
        #   adversary:    [type, motion, goal_dist, ego_dist]
        self.use_adv_conditioning = bool(self.cfg_model.get("use_adv_conditioning", False))
        self.use_agent_conditioning = bool(self.cfg_model.get("use_agent_conditioning", False))
        self.cond_dropout_prob = float(self.cfg_model.get("cond_dropout_prob", 0.0))
        # Per-SCENE joint dropout on top of the per-token/per-field dropout above.
        # The iid dropout alone never produces the fully-unconditional scene that
        # prior-mode generation feeds in (every agent null AND all four adv fields
        # null at once): that state has probability p^n_agents * p^4, i.e. never.
        # This draws whole scenes to be fully null, so unconditional generation is
        # in-distribution -- the classifier-free-guidance convention of dropping
        # the entire conditioning jointly, kept alongside the iid dropout that
        # makes DDPO's partial-null targets in-distribution.
        self.uncond_scene_prob = float(self.cfg_model.get("cond_uncond_scene_prob", 0.0))
        if (self.use_adv_conditioning or self.use_agent_conditioning) and self.cond_dropout_prob <= 0:
            raise ValueError(
                "cond_dropout_prob must be > 0 when conditioning is enabled "
                "(needed for the null/unconditional token)."
            )

        if self.use_agent_conditioning:
            self.agent_type_embedder = LabelEmbedder(self.cfg_model.cond_num_types, self.cfg_model.hidden_dim, self.cond_dropout_prob)
            self.agent_motion_embedder = LabelEmbedder(self.cfg_model.cond_num_motion, self.cfg_model.hidden_dim, self.cond_dropout_prob)
            self.agent_goaldist_embedder = LabelEmbedder(self.cfg_model.cond_num_goaldist, self.cfg_model.hidden_dim, self.cond_dropout_prob)

        if self.use_adv_conditioning:
            self.adv_type_embedder = LabelEmbedder(self.cfg_model.cond_num_types, self.cfg_model.hidden_dim, self.cond_dropout_prob)
            self.adv_motion_embedder = LabelEmbedder(self.cfg_model.cond_num_motion, self.cfg_model.hidden_dim, self.cond_dropout_prob)
            self.adv_goaldist_embedder = LabelEmbedder(self.cfg_model.cond_num_goaldist, self.cfg_model.hidden_dim, self.cond_dropout_prob)
            self.adv_egodist_embedder = LabelEmbedder(self.cfg_model.cond_num_egodist, self.cfg_model.hidden_dim, self.cond_dropout_prob)
        
        # These will be overwritten by sin/cos positional encodings
        self.pos_emb_lane = nn.Parameter(torch.zeros(self.cfg_dataset.max_num_lanes, self.cfg_model.hidden_dim), requires_grad=False)
        self.pos_emb_agent = nn.Parameter(torch.zeros(self.cfg_dataset.max_num_agents, self.cfg_model.agent_hidden_dim), requires_grad=False)
        
        # factorized dit blocks
        self.blocks = nn.ModuleList([
            FactorizedDiTBlock(
                self.cfg_model.hidden_dim, 
                self.cfg_model.agent_hidden_dim, 
                self.cfg_model.num_heads, 
                self.cfg_model.agent_num_heads, 
                self.cfg_model.dropout, 
                mlp_ratio=4, 
                num_l2l_blocks=self.cfg_model.num_l2l_blocks 
                ) for _ in range(self.cfg_model.num_factorized_dit_blocks)
        ])

        # noise prediction heads
        self.pred_agent_noise = FinalLayer(self.cfg_model.agent_hidden_dim, self.cfg_model.agent_latent_dim)
        self.pred_lane_noise = FinalLayer(self.cfg_model.hidden_dim, self.cfg_model.lane_latent_dim)
        self.pred_adv_noise = FinalLayer(self.cfg_model.agent_hidden_dim, self.adv_latent_dim)
        self.initialize_weights()


    def initialize_weights(self):
        """ Custom initialization for DiT model"""
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) lane and agent pos_embed by sin-cos embedding:
        pos_emb_lane = get_1d_sincos_pos_embed_from_grid(self.pos_emb_lane.shape[-1], np.arange(self.pos_emb_lane.shape[0]))
        self.pos_emb_lane.data.copy_(torch.from_numpy(pos_emb_lane).float())
        pos_emb_agent = get_1d_sincos_pos_embed_from_grid(self.pos_emb_agent.shape[-1], self.cfg_dataset.max_num_lanes + np.arange(self.pos_emb_agent.shape[0]))
        self.pos_emb_agent.data.copy_(torch.from_numpy(pos_emb_agent).float())

        # Initialize num lane and num agent embedding tables:
        nn.init.normal_(self.num_agents_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.num_lanes_embedder.embedding_table.weight, std=0.02)

        # Initialize conditioning embedding tables (null row included):
        if self.use_agent_conditioning:
            nn.init.normal_(self.agent_type_embedder.embedding_table.weight, std=0.02)
            nn.init.normal_(self.agent_motion_embedder.embedding_table.weight, std=0.02)
            nn.init.normal_(self.agent_goaldist_embedder.embedding_table.weight, std=0.02)
        if self.use_adv_conditioning:
            nn.init.normal_(self.adv_type_embedder.embedding_table.weight, std=0.02)
            nn.init.normal_(self.adv_motion_embedder.embedding_table.weight, std=0.02)
            nn.init.normal_(self.adv_goaldist_embedder.embedding_table.weight, std=0.02)
            nn.init.normal_(self.adv_egodist_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            for l2l_block in block.l2l_blocks:
                nn.init.constant_(l2l_block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(l2l_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.a2a_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.a2a_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.l2a_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.l2a_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.la2adv_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.la2adv_block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.pred_agent_noise.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.pred_agent_noise.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.pred_agent_noise.linear.weight, 0)
        nn.init.constant_(self.pred_agent_noise.linear.bias, 0)

        nn.init.constant_(self.pred_lane_noise.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.pred_lane_noise.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.pred_lane_noise.linear.weight, 0)
        nn.init.constant_(self.pred_lane_noise.linear.bias, 0)

        nn.init.constant_(self.pred_adv_noise.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.pred_adv_noise.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.pred_adv_noise.linear.weight, 0)
        nn.init.constant_(self.pred_adv_noise.linear.bias, 0)


    def freeze_non_adv_parameters(self, freeze_cond_embedders=False):
        """Freeze everything except the adversary branch. When
        ``freeze_cond_embedders`` is True the adv conditioning embedders are ALSO
        kept frozen (used by DDPO, which fixes the condition to a constant target
        and only reshapes the adv latent denoiser, preserving the supervised
        conditional prior). The default keeps them trainable for adv_only base
        training, where the conditioning is still being learned."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)

        adv_modules = [self.adv_embedder, self.pred_adv_noise]
        if self.use_adv_conditioning and not freeze_cond_embedders:
            adv_modules.extend([
                self.adv_type_embedder,
                self.adv_motion_embedder,
                self.adv_goaldist_embedder,
                self.adv_egodist_embedder,
            ])
        for block in self.blocks:
            adv_modules.extend([
                block.downsample_x_lane_to_adv,
                block.la2adv_block,
            ])

        for module in adv_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)


    def _uncond_scene_mask(self, batch_size, device):
        """Per-scene ``[batch_size]`` bool mask of scenes whose conditioning is
        dropped *entirely* (every agent, every adv field). Drawn once per forward
        so the agent and adv streams are dropped together -- that joint state is
        exactly what prior-mode generation feeds in. ``None`` outside training or
        when ``cond_uncond_scene_prob`` is 0."""
        if not self.training or self.uncond_scene_prob <= 0:
            return None
        return torch.rand(batch_size, device=device) < self.uncond_scene_prob

    def _cond_drop_mask(self, num_tokens, store, device, num_fields=1,
                        uncond_scene=None, token_scene=None):
        """Conditioning dropout mask for a token stream.

        ``num_fields == 1`` (default, normal-agent stream): a single ``[num_tokens]``
        mask shared across all of a token's labels, so each token is conditioned
        all-or-nothing (never on a partial label subset).

        ``num_fields > 1`` (adv stream): an independent ``[num_tokens, num_fields]``
        mask, so each field is dropped iid -- the model then sees partial-null
        combinations (e.g. type/motion pinned while goal_dist/ego_dist are null),
        which makes the DDPO per-field null-token target in-distribution.

        ``uncond_scene`` (training only) is the per-scene joint-drop mask from
        :meth:`_uncond_scene_mask`; every token of a selected scene is forced to
        null on top of the iid draw. ``token_scene`` maps each token to its scene
        (omit when the stream already has exactly one token per scene, like adv).

        Random while training; otherwise an explicit ``cond_drop`` mask if the
        caller provides one (inference: drop the agents whose conditions are
        irrelevant; broadcast across fields); else ``None`` (use the labels as
        given)."""
        if self.training and self.cond_dropout_prob > 0:
            shape = (num_tokens,) if num_fields == 1 else (num_tokens, num_fields)
            drop = (torch.rand(*shape, device=device) < self.cond_dropout_prob).long()
            if uncond_scene is not None:
                per_token = uncond_scene if token_scene is None else uncond_scene[token_scene]
                per_token = per_token.long()
                if drop.dim() == 2:
                    per_token = per_token.unsqueeze(1)
                drop = torch.maximum(drop, per_token)
            return drop
        if "cond_drop" in store:
            m = store.cond_drop.long().to(device)
            return m if num_fields == 1 else m.unsqueeze(1).expand(-1, num_fields)
        return None

    def forward(self,
                x_lane,
                x_agent,
                x_adv,
                data,
                agent_timestep,
                lane_timestep,
                adv_timestep):
        """ Forward pass of the DiT model."""
        if x_adv.shape[0] != data.batch_size:
            raise ValueError(
                f"x_adv should contain one adversarial-agent token per scene, "
                f"got {x_adv.shape[0]} tokens for batch_size={data.batch_size}"
            )
        
        lane_idx_batch = get_indices_within_scene(data['lane'].batch)
        agent_idx_batch = get_indices_within_scene(data['agent'].batch)
        
        # add positional embeddings
        pos_emb_lane = self.pos_emb_lane[lane_idx_batch]
        pos_emb_agent = self.pos_emb_agent[agent_idx_batch]
        x_lane = self.lane_embedder(x_lane[:, 0]) + pos_emb_lane
        x_agent = self.agent_embedder(x_agent[:, 0]) + pos_emb_agent
        x_adv = self.adv_embedder(x_adv[:, 0])
                
        agent_batch = data['agent'].batch 
        lane_batch = data['lane'].batch
        
        num_agents = data['num_agents'].long()
        num_lanes = data['num_lanes'].long()
        num_agents_emb = self.num_agents_embedder(num_agents, train=self.training)
        num_lanes_emb = self.num_lanes_embedder(num_lanes, train=self.training)
        num_agents_emb_per_agent = num_agents_emb[agent_batch]
        num_lanes_emb_per_lane = num_lanes_emb[lane_batch]
        num_context_emb_per_adv = num_agents_emb + num_lanes_emb

        # Per-agent conditioning: add the embedded [type, motion, goal_dist] labels
        # onto each normal-agent token's context embedding. One drop mask per agent
        # is shared across its three labels (all-or-nothing). When labels are absent
        # (e.g. init_adv / a DDPO rollout where the agents are given) every agent is
        # treated as the trained null state -- the same distribution training saw
        # for a fully-dropped agent -- rather than dropping the term entirely.
        # Scenes drawn to be fully unconditional this step: one draw shared by the
        # agent and adv branches below, so a selected scene goes all-null on both.
        uncond_scene = self._uncond_scene_mask(data.batch_size, x_adv.device)

        if self.use_agent_conditioning:
            device = num_agents_emb_per_agent.device
            n_agent = num_agents_emb_per_agent.shape[0]
            if "cond" in data["agent"]:
                agent_cond = data["agent"].cond.long()
                drop = self._cond_drop_mask(
                    n_agent, data["agent"], device,
                    uncond_scene=uncond_scene, token_scene=agent_batch)
            else:
                agent_cond = torch.zeros((n_agent, 3), dtype=torch.long, device=device)
                drop = torch.ones(n_agent, dtype=torch.long, device=device)
            num_agents_emb_per_agent = (
                num_agents_emb_per_agent
                + self.agent_type_embedder(agent_cond[:, 0], train=self.training, force_drop_ids=drop)
                + self.agent_motion_embedder(agent_cond[:, 1], train=self.training, force_drop_ids=drop)
                + self.agent_goaldist_embedder(agent_cond[:, 2], train=self.training, force_drop_ids=drop)
            )

        # Adv conditioning: add the embedded [type, motion, goal_dist, ego_dist]
        # labels onto the adv stream's context embedding (one adv token per scene,
        # so this is per-scene). Only feeds num_context_emb_per_adv -> c_adv, so the
        # lane and normal-agent streams are untouched by the adv labels. Cond-absent
        # is treated as the trained null state (see the agent branch above).
        if self.use_adv_conditioning:
            device = num_context_emb_per_adv.device
            n_adv = num_context_emb_per_adv.shape[0]
            if "cond" in data["adv"]:
                adv_cond = data["adv"].cond.long()
                # Independent per-field dropout (adv stream only): each of the four
                # labels is dropped iid, so partial-null combinations are trained.
                # One adv token per scene, so uncond_scene indexes it directly.
                drop = self._cond_drop_mask(n_adv, data["adv"], device, num_fields=4,
                                            uncond_scene=uncond_scene)
            else:
                adv_cond = torch.zeros((n_adv, 4), dtype=torch.long, device=device)
                drop = torch.ones((n_adv, 4), dtype=torch.long, device=device)
            drop_col = (lambda _k: None) if drop is None else (lambda k: drop[:, k])
            num_context_emb_per_adv = (
                num_context_emb_per_adv
                + self.adv_type_embedder(adv_cond[:, 0], train=self.training, force_drop_ids=drop_col(0))
                + self.adv_motion_embedder(adv_cond[:, 1], train=self.training, force_drop_ids=drop_col(1))
                + self.adv_goaldist_embedder(adv_cond[:, 2], train=self.training, force_drop_ids=drop_col(2))
                + self.adv_egodist_embedder(adv_cond[:, 3], train=self.training, force_drop_ids=drop_col(3))
            )

        # embedding of timestep
        t = self.t_embedder(torch.cat([lane_timestep, agent_timestep, adv_timestep], dim=-1))
        # embedding of number of agents and lanes
        n = torch.cat([num_lanes_emb_per_lane, num_agents_emb_per_agent, num_context_emb_per_adv], dim=0)

        l2l_edge_index = data['lane', 'to', 'lane'].edge_index
        a2a_edge_index = data['agent', 'to', 'agent'].edge_index
        l2a_edge_index = data['lane', 'to', 'agent'].edge_index.clone()
        l2a_edge_index[1] = l2a_edge_index[1] + x_lane.shape[0]
        # Bipartite lane+agent -> adv edges for the la2adv cross-attention block.
        # Sources index into the concatenated [lane; agent] key/value tensor;
        # destinations are the per-scene adv tokens (one per scene), so dst is
        # simply the scene index (lane_batch / agent_batch).
        lane_to_adv_edge_index = torch.stack([
            torch.arange(x_lane.shape[0], device=x_lane.device),
            lane_batch,
        ], dim=0)
        agent_to_adv_edge_index = torch.stack([
            torch.arange(x_agent.shape[0], device=x_agent.device) + x_lane.shape[0],
            agent_batch,
        ], dim=0)
        la2adv_edge_index = torch.cat([lane_to_adv_edge_index, agent_to_adv_edge_index], dim=1)
        
        # conditioning vector for DiT block
        c = t  + n 
        # necessary for A2A and L2A attention
        c_small = self.downsample_c(c)
        
        # apply dropout
        x_lane = self.emb_drop(x_lane)
        x_agent = self.emb_drop(x_agent)
        x_adv = self.emb_drop(x_adv)
        
        # factorized dit block processing
        for block in self.blocks:
            x_lane, x_agent, x_adv = block(
                x_lane, 
                x_agent, 
                x_adv,
                c, 
                c_small, 
                l2l_edge_index, 
                a2a_edge_index, 
                l2a_edge_index,
                la2adv_edge_index)

        # decode the noise as in the original DiT paper
        c_lane = c[:x_lane.shape[0]]
        c_agent = c_small[x_lane.shape[0]:x_lane.shape[0] + x_agent.shape[0]]
        c_adv = c_small[x_lane.shape[0] + x_agent.shape[0]:]
        x_lane = self.pred_lane_noise(x_lane, c_lane).unsqueeze(1)
        x_agent = self.pred_agent_noise(x_agent, c_agent).unsqueeze(1)
        x_adv = self.pred_adv_noise(x_adv, c_adv).unsqueeze(1)

        return x_agent, x_lane, x_adv
