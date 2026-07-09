# Latent Adversary Diffusion: Controllable Safety-Critical Scenario Generation via Reinforcement-Learned Latent Diffusion

<!--
Working title. Alternatives:
  - "Dreaming Up Trouble: RL-Finetuned Latent Diffusion for Safety-Critical Driving Scenarios"
  - "One Bad Agent: Inserting Reinforcement-Learned Adversaries into Real Driving Scenes"
Method name used throughout: LDM-Adv (architecture) + DDPO fine-tuning stage.
Consider a catchier system name before submission (check for collisions on arXiv).
-->

**Anonymous Authors**
*(Draft v0.1 — 2026-07-05. Numbers marked ⟨TBD⟩ are placeholders pending final runs; see the "Gap Analysis" appendix at the end of this file, which is NOT part of the paper.)*

---

## Abstract

Validating autonomous-driving planners requires exposure to rare, safety-critical interactions that are underrepresented in driving logs. We present a two-stage generative framework that *inserts a single controllable adversarial agent into real driving scenes* and then *optimizes it, closed-loop, to induce criticality against a reactive planner*. First, we extend a vectorized latent-diffusion scene generator with a dedicated adversary stream (**LDM-Adv**): the adversary is one extra agent latent diffused jointly with — or conditioned on — the frozen latents of a pre-trained scene autoencoder, requiring no autoencoder retraining and inheriting the base model's realism. Discretized semantic labels (agent type, motion, travel distance, distance-to-ego) with per-field null-token dropout make the adversary controllable without classifier-free guidance. Second, we fine-tune only the adversary branch with denoising-diffusion policy optimization (DDPO) using group-relative (GRPO-style) advantages, against a frozen reactive planner in a lightweight closed-loop simulator, under a criticality reward combining an analytic time-to-collision risk term, a gated approach bonus with annealing, and time-ramped collision and ego-at-fault bonuses, while validity is enforced through graded condition-violation and overlap penalties. We identify and analyze two characteristic failure modes of group-whitened RL on constrained diffusion models — *reward-cliff collapse* and *dense-term substitution with rare-event blindness* — and introduce a set of stabilizers (graded penalties, global whitening of degenerate groups, adaptive KL-to-base control, log-ratio dropping, decoupled gradient clipping, and asymmetric agent conditioning) that address them. On the Waymo Open Motion Dataset, our fine-tuned generator increases the planner's collision rate from ⟨TBD⟩% to ⟨TBD⟩% and the ego-at-fault collision rate from ~0% to ⟨TBD⟩%, while maintaining ⟨TBD⟩% condition adherence and staying within a KL budget of the pretrained scene prior.

---

## 1. Introduction

Autonomous vehicles fail rarely, and that is precisely the problem: the situations that matter most for safety validation — cut-ins, occluded crossings, sudden encroachments — occupy a vanishing fraction of collected driving logs [30, 34]. Replaying logs cannot cover them; hand-authored test cases do not scale and inherit their designers' blind spots. This has motivated *learned scenario generation*: train a generative model on real scenes so it captures realistic road layouts and agent configurations, then steer it toward the safety-critical tail.

Recent scene-generation models based on (latent) diffusion produce realistic initial scenes — road geometry, agent placements, and goals — directly from noise [1, 27, 28]. Scenario Dreamer [1] in particular introduced a vectorized latent diffusion model over autoencoded lane and agent tokens that supports scene inpainting and closed-loop simulator construction. But such models are trained with a likelihood objective on *nominal* traffic: sampling from them reproduces the safe interactions that dominate the data. Criticality is a property of the *closed-loop interaction* between a scene and the planner under test — it cannot be expressed as a denoising loss, and it is planner-dependent.

We bridge this gap by treating scenario generation as a *reinforcement-learning problem on top of a frozen generative prior*. Our key design choice is to make the action space as small as safety-critical testing allows: **one adversarial agent inserted into an otherwise real scene**. The map, the ego vehicle, and all normal agents are taken from a real log; the generator only decides the adversary's initial state and goal. This isolates credit assignment, keeps the scene distribution anchored to reality, and matches the standard practice of perturbation-based scenario attacks [14, 15, 16] — but with the perturbation *generated from a learned, semantically controllable prior* rather than optimized per-scene.

Concretely, we make four contributions:

1. **LDM-Adv, an adversary-aware latent diffusion architecture (§3.2).** We add a third *adversary stream* to a factorized lane/agent diffusion transformer operating in the latent space of a frozen, goal-aware scene autoencoder. Because the adversary is just one extra agent latent, the pretrained autoencoder and its exported latent cache are reused *unchanged*. A *mixed-mode* training scheme places all three generation modes — full-scene, agents-given-lanes, adversary-given-scene — in the training distribution, so conditional adversary insertion at inference is not out-of-distribution. A permutation-equivariance argument lets us decode the inserted adversary jointly with the real agents in a single autoencoder pass, avoiding a decode artifact that otherwise collapses ~9% of adversaries onto the ego (§3.4).

2. **Semantic, droppable conditioning (§3.3).** Every agent is conditioned on discretized labels (type, motion, goal distance); the adversary additionally on its distance-to-ego bucket. Per-field null-token dropout — independent across the adversary's four fields — makes any subset of the specification optional at inference without classifier-free guidance, and gives the downstream RL stage a well-defined *condition-violation* gate.

3. **Closed-loop RL fine-tuning with a criticality reward (§3.5–3.6).** We fine-tune only the adversary branch (6% of parameters) with DDPO [7] under GRPO-style per-context group advantages [10], against a frozen reactive planner rolled out in a lightweight numpy re-implementation of a GPU driving simulator. The reward combines a dense analytic time-to-collision risk term, a *gated* approach bonus that pays only for genuine closing-in (not for spawning near the ego) and is annealed over training, time-ramped collision and ego-at-fault bonuses sized to exceed the within-group reward spread, and graded validity penalties.

4. **A failure-mode analysis of group-whitened RL on constrained diffusion (§5).** From instrumented training runs we characterize two failure modes we believe are general: (i) *reward-cliff collapse*, where a flat rejection penalty adjacent to the reward optimum turns the collapsed mode into a stable attractor under shift-invariant group whitening; and (ii) *dense-term substitution*, where a directly controllable dense shaping term absorbs the entire gradient while the rare event of interest stays invisible because its bonus is smaller than the within-group reward std. We introduce matching stabilizers — graded penalties, global whitening of degenerate groups, adaptive KL-to-base control [12], per-step log-ratio dropping, decoupled clipping of policy and KL gradients, and an *asymmetric conditioning* trick that makes only the adversary reckless — and ablate each.

> **[Figure 1 — PLACEHOLDER, teaser, single column]**
> *Left:* a real Waymo scene (grey lanes, blue normal agents, red ego with its goal). *Middle:* the same scene with an adversary (green) sampled from the pretrained LDM-Adv prior — plausible but benign; the planner rollout (faded trajectory traces) passes without conflict. *Right:* the same scene and conditioning after DDPO fine-tuning — the adversary spawns on a merging course, the rollout shows the ego braking late and colliding (collision marker ⚡, "ego-at-fault" tag). Caption states: same map, same ego, same semantic condition (vehicle / moving / goal:far); only the adversary latent distribution changed.
> *Source material: viz GIF frames from `eval_visualize_before_rl` vs. the fine-tuned checkpoint on the same val scene (wandb media, runs after 8rlw8ay8-fixes).*

---

## 2. Related Work

**Initial-scene and traffic-scene generation.** SceneGen [23] autoregressively inserts agents into HD maps; TrafficGen [24] combines learned placement with rule-based rollouts. Diffusion-based generators now produce full scenes from scratch: DriveSceneGen [28] and SLEDGE [27] generate rasterized/vectorized scenes and lane graphs, and Scenario Dreamer [1] — whose architecture and codebase we build on — trains a vectorized latent diffusion model over autoencoded lane and agent tokens with goal-augmented agent states. These models target *realism and coverage*, not criticality; sampling them reproduces the nominal-traffic distribution. Our method converts such a prior into an adversarial generator while retaining its realism through a KL anchor.

**Safety-critical scenario generation.** A first family perturbs logged trajectories against a planner: AdvSim [16] optimizes acceleration perturbations with black-box search, Learning-to-Collide [17] parameterizes attacks with learned samplers, and KING [15] uses kinematics gradients through a differentiable simulator. STRIVE [14] regularizes the attack with a learned traffic prior by optimizing in its latent space — closest in spirit to us, but it performs per-scenario test-time optimization, whereas we *amortize* the attack into a fine-tuned generator that samples critical scenes in one pass. CAT [18] resamples adversarial trajectories from a motion-forecasting prior for closed-loop adversarial training. A second family steers diffusion models at sampling time: CTG [19] and CTG++ [20] apply guidance (including language-derived costs) to trajectory diffusion; DiffScene [22] and SAFE-SIM [21] use safety-objective guidance to push a trajectory diffusion model toward critical behaviors while keeping plausibility. All of these attack the *behavior* of agents over time in a fixed scene; our action space is instead the *initial configuration* (placement, heading, speed, goal) of one inserted agent, with behavior delegated to the same reactive policy that drives all traffic — guaranteeing the adversary is *reactive and physically driveable* rather than an open-loop trajectory. Ding et al. [34] survey this design space.

**RL fine-tuning of diffusion models.** DDPO [7] casts the denoising chain as an MDP and applies policy gradients with per-step importance ratios; DPOK [8] adds KL regularization to the pretrained model; Diffusion-DPO [9] transfers preference optimization to diffusion. Group-relative advantage normalization (GRPO) was introduced for LLM fine-tuning in DeepSeekMath [10] and removes the learned value baseline by whitening rewards within groups sharing a prompt/context. We combine DDPO's per-step ratios with GRPO's per-context groups (each real scene context replicated K times), a closed-form Gaussian KL-to-base penalty, and an adaptive KL controller in the style of Ziegler et al. [12]. To our knowledge we provide the first analysis of GRPO-style whitening interacting pathologically with hard validity gates in constrained generation (§5), and the first application of this recipe to driving-scenario generation.

**Closed-loop driving simulation.** Evaluating criticality requires reactive rollouts. Waymax [31] and GPUDrive [32] provide hardware-accelerated data-driven simulators; nuPlan [33] established closed-loop planning evaluation; CtRL-Sim [29] and BITS [25] learn controllable/reactive agent policies; MixSim [36] studies mixed-reality traffic resimulation. We use a frozen self-play RL policy (trained in a PufferLib-based [35] GPUDrive-style simulator) to drive *all* agents including the ego, and score criticality from oriented-bounding-box geometry along the rollout.

---

## 3. Method

### 3.1 Problem statement and overview

Let a *scene context* $c = (L, A_{1:N}, e)$ consist of vectorized lanes $L$, normal agents $A_{1:N}$, and an ego vehicle $e$ with a real on-road goal, all taken from a driving log. We seek a generator $\pi_\theta(a \mid c, s)$ over a single adversarial agent $a$ — its position, heading, speed, extent, and goal — such that (i) $a$ satisfies a user-specified semantic condition $s$ (e.g. *vehicle, moving, far goal*), (ii) the composed scene $(c, a)$ remains physically valid and on-distribution, and (iii) rolling out a fixed reactive planner $\rho$ for all agents from $(c, a)$ yields high *criticality* for the ego: low time-to-collision, genuine closing interactions, and ideally ego-at-fault collisions.

Our pipeline (Fig. 2) has two stages. **Stage 1** trains LDM-Adv, a latent diffusion model with an explicit adversary stream, by maximum likelihood on real scenes where one logged agent per scene is relabeled as the "adversary" (§3.2–3.4). **Stage 2** freezes everything except the adversary branch and fine-tunes it with DDPO against rollout rewards (§3.5–3.6).

> **[Figure 2 — PLACEHOLDER, architecture + pipeline, two columns]**
> Top row (Stage 1): frozen goal-autoencoder encodes a real scene into per-lane latents (24-d) and per-agent latents (8-d); one non-ego agent's latent is split off as the adversary token. The three token sets enter the factorized diffusion transformer; one block is expanded showing the attention order *lane→lane, lane→agent, agent→agent, (lane+agent)→adv cross-attention*. Conditioning embedders (type/motion/goal-dist per agent; +ego-dist for the adv) with a null-token slot feed the per-token conditioning vector. A "mixed-mode" panel shows the three training configurations as masks over (lane, agent, adv) noise levels: init_scene (all noised), init_agent (lanes clean), init_adv (lanes+agents clean, only adv noised).
> Bottom row (Stage 2): conditioning pool of real scenes → init_adv denoising of the adversary latent only (base frozen, snowflake icons) → reinsert-and-decode through the frozen AE decoder → closed-loop rollout with the frozen planner (all agents reactive) → reward assembly (reject / init-invalid / valid branches) → GRPO group advantages → DDPO update of the adversary branch, with a KL-to-base tether drawn back to the pretrained weights.

### 3.2 LDM-Adv: an adversary stream on a frozen scene latent space

**Base model.** Following Scenario Dreamer [1], a scene autoencoder maps each lane segment to a 24-dim latent and each agent to an 8-dim latent; the decoder reconstructs continuous states — position, heading (as $\cos\theta,\sin\theta$), speed, extent, and *goal* — together with discrete attributes (agent type, lane connectivity) from the latent set using inter-token attention. A diffusion transformer (DiT [5]) with factorized attention blocks then models the joint distribution of the latent sets, trained with the standard $\epsilon$-prediction objective [2] under a cosine noise schedule [37].

**Adversary stream.** We designate one non-ego agent per training scene as the adversary and split its latent out of the agent set *at data-loading time*. Crucially, the autoencoder is unaware of this split — an adversary is just one more 8-dim agent latent — so the pretrained autoencoder and its exported latent cache (487k train / 44k val scenes from the Waymo Open Motion Dataset [30]) are reused without any re-export or retraining. The DiT gains a third token stream holding this single adversary latent; each factorized block appends a cross-attention step in which the adversary token attends over the processed lane and agent tokens (order per block: $l{\to}l$, $l{\to}a$, $a{\to}a$, $(l{+}a){\to}\mathrm{adv}$). The adversary thus sees the full scene at every block, while lanes and agents are unaffected by it — matching the causal structure of "insert one agent into a fixed scene" and allowing the base streams to be frozen later without breaking their computation graph.

**Mixed-mode training.** The model supports three generation modes: *init_scene* (generate everything), *init_agent* (lanes given), and *init_adv* (lanes and agents given, generate the adversary only). Naive joint training shares one diffusion timestep $t$ across all streams in a scene, so the conditional modes — where conditioning streams sit at $t=0$ with exactly-clean latents — are never visited and are out-of-distribution at sampling time. We therefore train in a *mixed* regime: per scene we draw a mode $m \sim p_\text{mode}$ (0.7 init_scene / 0.3 init_adv in our runs), feed conditioning streams their clean latents at $t=0$ exactly as the sampler will, and mask their $\epsilon$-losses:

$$\mathcal{L} = \mathbb{E}_{m,\,t,\,\epsilon}\Big[ \sum_{k \in \{l,a,\mathrm{adv}\}} w_k \, \mathbb{1}[k \in \mathcal{G}(m)] \, \big\| \epsilon_k - \hat\epsilon_k(x_t, t, c) \big\|^2 \Big],$$

where $\mathcal{G}(m)$ is the set of generated streams under mode $m$ and $w_k$ are stream weights (lane 10, agent 1, adv 1). The scheme adds no parameters, so checkpoints from joint training warm-start it exactly.

### 3.3 Semantic conditioning with per-field null tokens

Controllable stress testing needs a *specification interface*: "a moving vehicle, goal 25+ m away, near the ego." We condition every agent token on discretized labels, embedded by per-field lookup tables and added to the token's conditioning vector:

- **normal agent:** [type ∈ {vehicle, pedestrian, cyclist}, motion ∈ {parked, moving}, goal-dist ∈ {near, middle, far}];
- **adversary:** the same three, plus **ego-dist** ∈ {near, middle, far}, the bucketed distance from the adversary to the ego at spawn.

Buckets use fixed metric thresholds (near < 10 m, far > 25 m; parked if the goal is within 2 m of spawn), computed from each agent's own decoded state, so the labels are *verifiable post-hoc* — a property Stage 2 exploits. Each embedder is trained with null-token dropout ($p=0.2$): a dropped field is replaced by a dedicated null embedding, so at inference any field can be left unspecified by feeding its null token, with no classifier-free guidance [6] and hence no guidance-scale tuning. For the adversary the dropout mask is drawn *independently per field*, so partial specifications (e.g. type and motion pinned, distances free) are in-distribution; for normal agents dropout is all-or-nothing per token, so a fully unconditioned agent — the DDPO setting, where agents are given, not asked for — is exactly the trained null state.

### 3.4 Decoding: reinsert, don't isolate

At sampling time the generated adversary latent must be decoded back to a physical agent. The autoencoder's agent decoder is a *permutation-equivariant set model* (complete agent-to-agent attention, lane-to-agent attention, no positional embedding, no ego token), trained only on full agent sets. Decoding the adversary latent in isolation is out-of-distribution and biases the lone agent toward the origin — the ego's canonical position: in a 400-scene diagnostic, isolated decoding placed 9.2% of adversaries within 3 m of the ego when the true agent was tens of meters away. We instead *append the adversary latent to the scene's full agent-latent set and decode everything in a single pass*, then read the adversary row back out. Permutation equivariance makes the insertion position irrelevant; the reinserted decode matches an independent full-set decode to numerical precision (max deviation 0.0000 m) and reduces the false ego-overlap rate from 9.2% to 0%, with median reconstruction error 0.16 m against ground truth.

> **[Figure 3 — PLACEHOLDER, decode artifact, single column, 3 panels]**
> Panel (a): isolated adversary decode — green adversary box collapsed onto the red ego at the origin, ground-truth adversary position shown as a dashed outline ~29 m away. Panel (b): reinsert-and-decode — green box matches the dashed outline. Panel (c): scatter/CDF over 400 val scenes of adversary-to-ego distance under both decoding schemes vs. ground truth, showing the isolated decoder's spurious mass near 0 m.
> *Source: outputs of `verify_adv_overlap.py`.*

### 3.5 Closed-loop RL fine-tuning

**MDP and policy.** Following DDPO [7], the reverse diffusion chain is an MDP whose per-step action is the sampled next latent, with Gaussian policy $\pi_\theta(x_{t-1} \mid x_t, c) = \mathcal{N}(\mu_\theta(x_t, t, c), \sigma_t^2 I)$. In our *init_adv* setting the state is the single 8-dim adversary latent; lanes and agent latents are clean conditioning, and only the adversary branch of the DiT — its stream-specific blocks and conditioning embedders, 48 of 178 parameter tensors (≈6%) — is trainable. Log-probabilities and KL are accumulated over the adversary token only.

**Conditioning pool.** Contexts are drawn from a pool of 40k real scenes, filtered to egos that actually drive (ground-truth goal ≥ 10 m from spawn; a stationary ego gives the criticality reward no signal). The generated adversary is *appended as an extra agent* to the full real scene — every real neighbor is kept, so criticality must be achieved amid genuine traffic. Per scene, a semantic target $s$ is drawn deterministically from a user distribution (our default: vehicle, moving, goal-dist ∈ {middle, far}, ego-dist free/null) and held fixed across that scene's group replicas and across epochs.

**Rollout and scoring.** The composed scene is decoded (§3.4) and rolled out for 91 steps at $\Delta t = 0.1$ s in a lightweight numpy re-implementation of a GPUDrive-style simulator [32]; a frozen recurrent self-play policy drives *all* agents, including the ego and the adversary. The rollout is scored by geometric hooks: an *analytic* minimum time-to-collision (closed-form first-contact time of oriented boxes under constant relative velocity, replacing a 101-step separating-axis sweep at identical output; 22× faster), oriented-box collision detection with an at-fault attribution, adversary–ego clearance over time, spawn-overlap fraction, and lane-distance diagnostics.

**Objective.** With per-context groups of size $K{=}8$ (batch 128 → 16 contexts), rewards are whitened within each group (GRPO [10]):

$$A_i = \mathrm{clip}\!\Big(\frac{r_i - \mu_{g(i)}}{\sigma_{g(i)} + \varepsilon},\ \pm 5\Big),$$

and the update maximizes the PPO-clipped importance-sampling objective over $k{=}16$ randomly subsampled denoising steps, with a closed-form Gaussian KL-to-base penalty (DPOK-style [8]) tethering the policy to the *pretrained* weights $\theta_0$:

$$\mathcal{L}(\theta) = -\mathbb{E}_{i,t}\Big[\min\big(\rho_{i,t} A_i,\ \mathrm{clip}(\rho_{i,t}, 1\pm\epsilon) A_i\big)\Big] + \beta \, \mathbb{E}_{i,t}\Big[\frac{\|\mu_\theta - \mu_{\theta_0}\|^2}{2\sigma_t^2}\Big],$$

where $\rho_{i,t} = \pi_\theta / \pi_{\text{old}}$. The KL is differentiated through $\mu_\theta$ (the sampled estimator $\log\pi_\theta - \log\pi_{\theta_0}$ at fixed samples has no opposing term and diverges). $\beta$ is set by an adaptive controller (§3.7).

### 3.6 Criticality reward

The reward for a rolled-out scene is assembled in three ordered branches:

**(a) Reject** — the decoded adversary violates its semantic condition $s$ on any checked field (its realized type/motion/goal-dist/ego-dist bucket, recomputed from the decoded scene with the *training* thresholds, differs from the target; null fields are skipped). The penalty is *graded* by the metric gap $g$ (meters) past the violated bucket boundary:

$$r = -\Big(b + (1-b)\,\min(g / s_g,\ 1)\Big), \qquad b = 0.5,\ s_g = 10\ \text{m},$$

so a near-miss of the bucket boundary scores −0.5 and only gross violations reach −1. §5.1 shows the flat −1 alternative is a collapse engine.

**(b) Init-invalid** — the adversary interpenetrates a neighbor at spawn (overlap fraction $f = \max_j \text{area}(a \cap A_j)/\text{area}(a) > 0$). Criticality is hard-gated to zero and the scene scores $-R_\text{constraint} \le 0$; the graded overlap term keeps a smooth "separate the boxes" gradient.

**(c) Valid** — the scene scores

$$r = \mathrm{clip}\big(R_\text{crit} - R_\text{constraint},\ -1,\ 1\big) + R_\text{bonus},$$

with

- $R_\text{crit} = \mathrm{noisyOR}\big(R_\text{ttc},\ w_\text{app} R_\text{app}\big) = 1 - (1 - R_\text{ttc})(1 - w_\text{app} R_\text{app})$, where $R_\text{ttc} = \mathrm{clip}(1 - \min\mathrm{TTC}/\tau,\ 0,\ 1)$ with horizon $\tau = 3$ s gives a dense near-miss gradient, and the *gated approach* term $R_\text{app} = \sigma\!\big(\tfrac{d_\text{safe} - d_\text{min}}{s_d}\big)\,\sigma\!\big(\tfrac{d_0 - d_\text{min} - \delta}{s_c}\big)$ pays only when the adversary both gets close ($d_\text{min}$ below $d_\text{safe}{=}6$ m) *and* actually closed in over the rollout ($d_0 - d_\text{min} > \delta{=}2$ m, measured after a 0.5 s warmup) — spawning next to the ego scores nothing;
- $R_\text{constraint}$: continuous smoothstep lane-distance penalties (spawn and goal, ramping over 1.75–2.75 m from the nearest centerline, weight 0.25 each) plus the graded overlap penalty ($0.5 f$) — *soft* constraints that reduce reward without destroying the criticality gradient;
- $R_\text{bonus} = r_\text{col}\,\big(\,0.5 + 1.0 \cdot \mathbb{1}[\text{ego at fault}]\,\big)$, where $r_\text{col}$ ramps from 0 to 1 over collision times $t \in [0.75, 1.25]$ s so trivial spawn-adjacent contacts earn nothing. The bonus is paid *outside* the clip and sized against the GRPO group statistics: it must exceed the within-group reward std (~0.2 in our runs) by a multiple, or group whitening cannot see the rare event (§5.2). An *ego-at-fault* collision — the actual test objective, e.g. a cut-in the ego crashes into — earns 1.5 total.

**Annealed bootstrap.** The approach weight $w_\text{app}$ anneals 1.0 → 0.25 over iterations 2k–6k: early on, TTC and collision have near-zero support and the dense approach term is the only usable gradient; left at full weight, it substitutes for criticality indefinitely (§5.2).

**Asymmetric conditioning (breaking the avoidance deadlock).** The frozen planner conditions each agent on a per-agent "collision-aversion" observation. When both ego and adversary are driven by the same defensive policy, *no initial configuration* yields more than ~2% collisions — two mutually avoidant agents negotiate around any geometry, capping what initial-condition attacks can achieve. We set the adversary's aversion input to 0 (reckless) while all other agents keep the nominal 0.5. This changes no weights and stays within the policy's training distribution of that observation; it removes the adversary's veto over contact while leaving the *ego's* avoidance — the behavior under test — intact.

> **[Figure 4 — PLACEHOLDER, reward anatomy, two columns]**
> (a) Curves of the reward components as functions of their driving variable: $R_\text{ttc}$ vs. min-TTC; the two sigmoid factors of $R_\text{app}$ vs. $d_\text{min}$ and vs. closed-in distance $d_0{-}d_\text{min}$; the lane smoothstep vs. centerline distance; the collision-bonus time ramp vs. collision time; the graded reject penalty vs. bucket gap $g$ (contrasted with the flat −1 cliff as a dashed line). (b) A schematic decision tree of the three branches (reject / init-invalid / valid) with the formulas at the leaves. (c) The $w_\text{app}$ annealing schedule vs. iteration.

### 3.7 Stabilizers for group-whitened diffusion RL

Motivated by the failure analysis in §5, the fine-tuning stage adds five guards (all ablated in Table 3):

1. **Graded penalties everywhere** (§3.6a): no flat cliffs adjacent to the reward optimum.
2. **Global whitening of degenerate groups:** a group whose reward std is ~0 (e.g. all samples rejected) is whitened against the *whole batch's* statistics instead of being skipped — a uniformly bad group keeps a restoring gradient instead of becoming a fixed point.
3. **Adaptive KL-to-base [12]:** $\beta \mathrel{*}= 1.5$ while measured KL exceeds $1.5\times$ the target (0.05), $\beta \mathrel{/}= 1.5$ below it, clamped to $[0.01, 100]$. The reference is always the *pretrained* checkpoint, never the current weights, so the tether cannot drift.
4. **Log-ratio dropping:** with one inner epoch the update is on-policy and true per-step log-ratios are ≈0; steps with $|\log\rho| > 1$ are numerical junk from low-noise steps (tiny posterior variances) and are dropped. The PPO $\min$ keeps exploded ratios exactly when the advantage is negative, turning noise into unbounded repulsion.
5. **Decoupled gradient clipping:** the policy-gradient and KL terms are clipped separately; a single global clip lets an exploding policy term scale the KL pullback to nothing at the exact moment it is needed.

---

## 4. Experiments

### 4.1 Setup

**Data & models.** Waymo Open Motion Dataset [30]; scene autoencoder and base LDM per Scenario Dreamer [1] (agent latent 8-d, lane latent 24-d, goal-augmented decoder). LDM-Adv trained for 200k steps (batch ⟨TBD⟩, EMA 0.9999, mixed mode 0.7/0.3). DDPO: DDPM sampler with 100 steps (a 30-step DDIM variant is ablated), batch 128, group size 8, $k{=}16$ subsampled steps, lr $10^{-5}$, AdamW, up to 40k iterations on a single H100; conditioning pool 40k train scenes, evaluation on 64 held-out validation scenes every 100 iterations (rates over 8 scenes have 0.125 resolution — too coarse; media is rendered for 8 scenes only).

**Planner under test.** A frozen recurrent self-play policy ("bad_driver": 256-d GRU over a 64-d observation encoder) trained in a GPUDrive-style simulator [32, 35]; it drives all agents including the ego. Rollouts are 9.1 s (91 × 0.1 s) in a 64×64 m scene frame.

**Metrics.** *Criticality:* ego collision rate, ego-at-fault collision rate, min-TTC distribution, near-miss rate (min-TTC < 1.5 s), post-warmup ego–adversary minimum clearance. *Validity/controllability:* condition-adherence rate (1 − reject), spawn-overlap rate, off-lane rates, per-field bucket accuracy. *Realism/proximity to the prior:* KL-to-base, latent-space distance, and scene-statistic JSDs (agent speed, nearest-neighbor distance, lane offsets) between generated adversaries and the matched real-adversary distribution [1]. *Diversity:* std of adversary spawn position/heading per context across seeds.

**Baselines.**
(i) *Prior sampling:* LDM-Adv without RL, unconditional and condition-targeted;
(ii) *Best-of-N prior:* sample N=8/64 adversaries from the prior per scene, report the most critical (matches our group budget; the natural inference-time-compute baseline);
(iii) *Heuristic placement:* adversary spawned on the ego's route at a range of gaps with matched speed (an AdvSim-style [16] initial-condition attack without a learned prior);
(iv) *Real adversary replay:* the logged agent that the dataloader would have labeled adversary;
(v) *(stretch)* STRIVE-style per-scene latent optimization [14] with our prior and reward, matching compute.

### 4.2 Main results

**Table 1 — Criticality, validity, and realism on 64 held-out validation scenes (frozen bad_driver planner; mean over 3 seeds ± std). ⟨TBD⟩ = pending final runs.**

| Method | Collision % ↑ | Ego-fault % ↑ | minTTC (s) ↓ | Near-miss % ↑ | Adherence % ↑ | Overlap % ↓ | Off-lane % ↓ | JSD(speed) ↓ | KL-to-base ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Real adversary replay | ~2 | ~0 | ⟨TBD⟩ | ⟨TBD⟩ | — | 0 | ⟨TBD⟩ | 0 | — |
| Prior sampling (uncond.) | ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ | — | ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ | 0 |
| Prior sampling (conditioned) | ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ | 0 |
| Best-of-8 prior | ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ | 0 |
| Heuristic on-route placement | ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ | — | ⟨TBD⟩ | ⟨TBD⟩ | high | — |
| **LDM-Adv + DDPO (ours)** | **⟨TBD⟩** | **⟨TBD⟩** | **⟨TBD⟩** | **⟨TBD⟩** | **⟨TBD⟩** | **⟨TBD⟩** | **⟨TBD⟩** | **⟨TBD⟩** | **≤0.05** |

*Anchor points already measured in development runs (to be superseded): with both agents defensive, collision rate is pinned at ~2% regardless of initial conditions; the guarded reward rose 0 → 0.31 over 3.8k iterations before the collision-bonus/anneal/asymmetry interventions; healthy KL-to-base sits at 0.005–0.02.*

> **[Figure 5 — PLACEHOLDER, training dynamics, two columns, 2×3 panel grid]**
> Wandb curves for the final guarded run vs. the two diagnostic runs (3zjnupnh "collapse", 8rlw8ay8 "stall"): (a) mean train reward; (b) ego collision rate and ego-at-fault rate; (c) KL-to-base (log scale) with the healthy band 0.005–0.02 and the adaptive-controller target 0.05 shaded; (d) reject / init-invalid rates; (e) reward decomposition over training (stacked/overlaid: r_ttc, w·r_approach, r_bonus) showing the approach anneal handing gradient over to TTC/collision; (f) ego–adversary min clearance, showing the ~7 m planner-separation plateau in the stall run vs. the guarded run breaking below it.

> **[Figure 6 — PLACEHOLDER, qualitative, full width]**
> A 4×4 grid of rollout filmstrips (4 frames each: t = 0, 3, 6, 9 s) on validation scenes. Rows: (1) prior sample, (2) best-of-8 prior, (3) ours, (4) ours with a different semantic condition on the same scene (e.g. ego-dist:near vs. far). Annotate collision frames and ego-at-fault tags; adversary in green, ego in red, condition string printed under each strip (e.g. "adv: veh | moving | goal:far | ego:—").

### 4.3 Controllability

**Table 2 — Per-field condition adherence of the fine-tuned model (fraction of decoded adversaries whose realized bucket matches the request; null = unconstrained). ⟨TBD⟩.**

| Requested field | Prior (cond.) | + DDPO | Notes |
|---|---|---|---|
| type = vehicle | ⟨TBD⟩ | ⟨TBD⟩ | |
| motion = moving | ⟨TBD⟩ | ⟨TBD⟩ | parked adversary is a condition violation |
| goal-dist ∈ {middle, far} | ⟨TBD⟩ | ⟨TBD⟩ | the collapse-prone field (§5.1) |
| ego-dist = near / middle / far | ⟨TBD⟩ | ⟨TBD⟩ | swept one bucket at a time |

> **[Figure 7 — PLACEHOLDER, controllability + distribution shift, two columns]**
> (a) Histograms of realized goal-distance and ego-distance for the prior vs. the fine-tuned model under each conditioning target, with bucket boundaries (10/25 m) as vertical lines — shows the fine-tuned mass shifting toward the boundary but respecting it (the graded penalty's design intent). (b) Ego-frame heatmap of adversary spawn positions before vs. after fine-tuning — shows concentration into forward/merging conflict geometries rather than a single collapsed point. (c) Diversity: per-context spread of adversary placements across 8 group samples, prior vs. fine-tuned.

### 4.4 Ablations

**Table 3 — Ablating the stabilizers and reward terms (each row = final guarded config minus one component; 1 seed unless noted). Report: peak collision %, iterations-to-collapse (∞ = stable), final KL-to-base, adherence %. ⟨TBD⟩ except where a diagnostic run already established the outcome.**

| Variant | Collision % | Collapse iter | KL-to-base | Adherence % | Observed outcome |
|---|---|---|---|---|---|
| Full method | ⟨TBD⟩ | ∞ | ~0.05 | ⟨TBD⟩ | |
| flat −1 reject (no grading) | ⟨TBD⟩ | ~10.5k | 5.7e4 (diverged) | →0 | reward-cliff collapse; run 3zjnupnh (§5.1) |
| degenerate groups skipped (not global) | ⟨TBD⟩ | co-factor of collapse | ⟨TBD⟩ | ⟨TBD⟩ | collapsed mode becomes a fixed point |
| fixed KL coef (no adaptation) | ⟨TBD⟩ | ⟨TBD⟩ | unbounded | ⟨TBD⟩ | |
| no ratio-drop / coupled clipping | ⟨TBD⟩ | numerical blow-up follows semantic collapse | ⟨TBD⟩ | ⟨TBD⟩ | IS ratios → 1e6 |
| no collision/ego-fault bonus | ~2 (stall) | ∞ | healthy | ⟨TBD⟩ | run 8rlw8ay8: approach farmed, collisions flat (§5.2) |
| no approach anneal | ⟨TBD⟩ | ∞ | ⟨TBD⟩ | ⟨TBD⟩ | dense-term substitution persists |
| no approach term at all | ⟨TBD⟩ | — | — | — | expected cold-start failure: no early gradient |
| symmetric conditioning (adv defensive) | ≤~2 | ∞ | healthy | ⟨TBD⟩ | avoidance deadlock caps collisions |
| DDIM-30 sampler | ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ | ~2× faster sampling phase |
| prune scene to ego+adv | ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ | ⟨TBD⟩ | cleaner credit vs. losing real traffic |

### 4.5 Throughput

**Table 4 — Wall-clock per DDPO iteration (batch 64 profile, H100; sample → rollout+reward → update run serially). Real measured numbers.**

| Configuration | sample | reward | update | total | speedup |
|---|---|---|---|---|---|
| Baseline (SAT-sweep TTC, per-scene grids) | 3.2 s (30%) | 6.4 s (60%) | 1.1 s (10%) | 10.6 s | 1.0× |
| + analytic first-contact TTC (bit-exact, 22× on the TTC hook) | | | | | |
| + lane-grid cache across group replicas (~8× on grid build) | 3.2 s (52%) | 1.9 s (31%) | 1.0 s (17%) | 6.1 s | **1.73×** |

Both reward optimizations are bit-exact (all metrics max |Δ| = 0). After them the bottleneck flips to the 100-step DDPM sampling chain (Amdahl ceiling of further reward work: 1.44×), motivating the DDIM ablation.

---

## 5. Analysis: how group-whitened RL on a constrained diffusion model fails

We report two instrumented failure cases from development. We believe both are general to GRPO-style fine-tuning with validity gates, and they motivated §3.6–3.7.

### 5.1 Reward-cliff collapse

In an early run the reject penalty was a flat −1 and degenerate groups were skipped. Training was healthy for 7.7k iterations, then collapsed in two phases: a *semantic* collapse (condition-violation rate 13% → 80%, KL-to-base 0.02 → 130 over ~3k iterations), followed by a *numerical* one (on-policy IS ratios reaching 10^6, policy loss ~10^6). The mechanism: criticality is maximized at goal distances of 10–20 m, *directly adjacent* to the 100%-invalid "near" bucket (< 10 m); the gradient walked the goal-distance distribution over the cliff. Once a group was mostly rejected, three properties made the cliff absorbing: (i) the flat penalty has zero within-group contrast, so whitening — which is shift-invariant and blind to absolute reward — produced no restoring gradient; (ii) fully-rejected groups were skipped entirely; (iii) any garbage sample scoring anything other than −1 received a large positive advantage. Finally, global gradient clipping let the exploding policy term crush the KL pullback exactly when it was needed, and the PPO $\min$ retains exploded ratios precisely for negative advantages — a pure repulsion engine. The graded penalty, global degenerate-group whitening, adaptive KL, ratio drop, and decoupled clipping (§3.7) each break one link of this chain.

> **[Figure 8 — PLACEHOLDER, collapse anatomy, two columns]**
> (a) Timeline of run 3zjnupnh: reject rate, KL-to-base (log), max |log-ratio|, policy loss on a shared x-axis with the three phases shaded (healthy / semantic collapse / numerical collapse). (b) Scatter of reward vs. realized goal distance among valid samples (corr ≈ −0.42) with the 10 m bucket boundary drawn — the optimum sits against the cliff. (c) Schematic: within-group advantage under a flat −1 (all zeros once the group is over the cliff) vs. under the graded penalty (restoring gradient), and vs. global whitening for the all-rejected group.

### 5.2 Dense-term substitution and rare-event blindness

With the collapse guards active but before the collision bonus and asymmetry interventions, reward rose 0 → 0.31 over 3.8k iterations — yet the collision rate never left ~2% and ego-at-fault stayed at ~0. Decomposition showed *all* gain came from the dense approach term (0.07 → 0.36) while the TTC term was static; the adversary–ego clearance plateaued at ~7 m, the frozen planner's preferred separation — the policy had learned to farm the shaping term to its planner-limited ceiling. Two blind spots sustained the stall: a collision out-scored a deep near-miss by only ~0.1 ≈ 0.5σ of the within-group reward std, invisible after whitening; and mutual avoidance capped achievable collisions at ~2% for *any* initial condition. The fixes are dimensional-analysis simple: pay rare events a bonus that is a multiple of the within-group σ (ours: 0.5 and 1.5 vs. σ ≈ 0.2), anneal the dense bootstrap once its plateau is reached, and remove the adversary's (but not the ego's) avoidance via its conditioning input. A corollary for evaluation: with 8-scene evals, rates have 0.125 resolution and a true 2% reads as constant 0 — eval sample sizes must resolve the base rate being claimed.

---

## 6. Limitations

**Planner specificity.** Criticality is optimized against one frozen planner; the scenarios are adversarial *for it*. Cross-planner transfer (e.g. to an IDM controller or an independently trained RL policy [32]) is measured only as a secondary experiment ⟨TBD⟩; adversarial-training-loop closure (retrain the planner on generated scenes, à la CAT [18]) is future work.

**Initial-condition attacks only.** The adversary's *behavior* during rollout is the shared reactive policy (made reckless via conditioning); we do not optimize trajectories. This guarantees reactivity and physical plausibility but bounds achievable criticality by what the behavior policy can express — our asymmetric-conditioning intervention is exactly a (coarse, discrete) knob on that bound.

**Single adversary.** The architecture holds one adversary token; multi-agent conspiracies (e.g. one agent occluding another) are out of scope, though the stream design extends naturally.

**Simulator fidelity.** The rollout uses simplified dynamics and no road-edge/off-road signal for the generated maps; lane adherence is enforced only through reward penalties.

**Reward hand-tuning.** Despite the stabilizers, the reward carries ~15 shaped constants (Table A1). The failure analysis (§5) is partly a story about how sensitive this class of pipeline is to them.

---

## 7. Conclusion

We turned a realism-oriented latent diffusion scene generator into a controllable adversary generator: one extra latent stream, semantic null-token conditioning, and a GRPO-style DDPO stage against a frozen planner, held together by a criticality reward and a set of stabilizers derived from an explicit failure-mode analysis. The result is an amortized generator that samples reactive, condition-adherent, safety-critical variants of real scenes in a single pass. We hope the failure analysis — reward cliffs under shift-invariant whitening, dense-term substitution, rare-event bonuses sized against group σ — transfers to the growing body of work applying group-relative RL to constrained generative models beyond driving.

---

## References

[1] L. Rowe, R. Girgis, A. Gosselin, L. Paull, C. Pal, F. Heide. *Scenario Dreamer: Vectorized Latent Diffusion for Generating Driving Simulation Environments.* CVPR 2025. arXiv:2503.22496.

[2] J. Ho, A. Jain, P. Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020. arXiv:2006.11239.

[3] J. Song, C. Meng, S. Ermon. *Denoising Diffusion Implicit Models.* ICLR 2021. arXiv:2010.02502.

[4] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, B. Ommer. *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR 2022. arXiv:2112.10752.

[5] W. Peebles, S. Xie. *Scalable Diffusion Models with Transformers.* ICCV 2023. arXiv:2212.09748.

[6] J. Ho, T. Salimans. *Classifier-Free Diffusion Guidance.* NeurIPS 2021 Workshop / arXiv:2207.12598.

[7] K. Black, M. Janner, Y. Du, I. Kostrikov, S. Levine. *Training Diffusion Models with Reinforcement Learning.* ICLR 2024. arXiv:2305.13301.

[8] Y. Fan, O. Watkins, Y. Du, H. Liu, M. Ryu, C. Boutilier, P. Abbeel, M. Ghavamzadeh, K. Lee, K. Lee. *DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models.* NeurIPS 2023. arXiv:2305.16381.

[9] B. Wallace, M. Dang, R. Rafailov, L. Zhou, A. Lou, S. Purushwalkam, S. Ermon, C. Xiong, S. Joty, N. Naik. *Diffusion Model Alignment Using Direct Preference Optimization.* CVPR 2024. arXiv:2311.12908.

[10] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, D. Guo. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv:2402.03300, 2024. (GRPO.)

[11] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347, 2017.

[12] D. Ziegler, N. Stiennon, J. Wu, T. Brown, A. Radford, D. Amodei, P. Christiano, G. Irving. *Fine-Tuning Language Models from Human Preferences.* arXiv:1909.08593, 2019. (Adaptive KL controller.)

[13] L. Ouyang et al. *Training Language Models to Follow Instructions with Human Feedback.* NeurIPS 2022. arXiv:2203.02155.

[14] D. Rempe, J. Philion, L. Guibas, S. Fidler, O. Litany. *Generating Useful Accident-Prone Driving Scenarios via a Learned Traffic Prior (STRIVE).* CVPR 2022. arXiv:2112.05077.

[15] N. Hanselmann, K. Renz, K. Chitta, A. Bhattacharyya, A. Geiger. *KING: Generating Safety-Critical Driving Scenarios for Robust Imitation via Kinematics Gradients.* ECCV 2022. arXiv:2204.13683.

[16] J. Wang, A. Pun, J. Tu, S. Manivasagam, A. Sadat, S. Casas, M. Ren, R. Urtasun. *AdvSim: Generating Safety-Critical Scenarios for Self-Driving Vehicles.* CVPR 2021. arXiv:2101.06549.

[17] W. Ding, B. Chen, M. Xu, D. Zhao. *Learning to Collide: An Adaptive Safety-Critical Scenarios Generating Method.* IROS 2020. arXiv:1911.06531.

[18] L. Zhang, Z. Peng, Q. Li, B. Zhou. *CAT: Closed-loop Adversarial Training for Safe End-to-End Driving.* CoRL 2023. arXiv:2310.12432.

[19] Z. Zhong, D. Rempe, D. Xu, Y. Chen, S. Veer, T. Che, B. Ray, M. Pavone. *Guided Conditional Diffusion for Controllable Traffic Simulation (CTG).* ICRA 2023. arXiv:2210.17366.

[20] Z. Zhong, D. Rempe, Y. Chen, B. Ivanovic, Y. Cao, D. Xu, M. Pavone, B. Ray. *Language-Guided Traffic Simulation via Scene-Level Diffusion (CTG++).* CoRL 2023. arXiv:2306.06344.

[21] W.-J. Chang, F. Pittaluga, M. Tomizuka, W. Zhan, M. Chandraker. *SAFE-SIM: Safety-Critical Closed-Loop Traffic Simulation with Diffusion-Controllable Adversaries.* ECCV 2024. arXiv:2401.00391.

[22] C. Xu, D. Zhao, A. Sangiovanni-Vincentelli, B. Li. *DiffScene: Diffusion-Based Safety-Critical Scenario Generation for Autonomous Vehicles.* ICML 2023 Workshop on New Frontiers in Adversarial Machine Learning. ⚠︎ verify venue.

[23] S. Tan, K. Wong, S. Wang, S. Manivasagam, M. Ren, R. Urtasun. *SceneGen: Learning to Generate Realistic Traffic Scenes.* CVPR 2021. arXiv:2101.06541.

[24] L. Feng, Q. Li, Z. Peng, S. Tan, B. Zhou. *TrafficGen: Learning to Generate Diverse and Realistic Traffic Scenarios.* ICRA 2023. arXiv:2210.06609.

[25] D. Xu, Y. Chen, B. Ivanovic, M. Pavone. *BITS: Bi-level Imitation for Traffic Simulation.* ICRA 2023. arXiv:2208.12403.

[26] S. Tan, B. Ivanovic, X. Weng, M. Pavone, P. Kraehenbuehl. *Language Conditioned Traffic Generation (LCTGen).* CoRL 2023. arXiv:2307.07947.

[27] K. Chitta, D. Dauner, A. Geiger. *SLEDGE: Synthesizing Driving Environments with Generative Models and Rule-Based Traffic.* ECCV 2024. arXiv:2403.17933.

[28] S. Sun et al. *DriveSceneGen: Generating Diverse and Realistic Driving Scenarios from Scratch.* IEEE RA-L 2024. arXiv:2309.14685.

[29] L. Rowe, R. Girgis, A. Gosselin, B. Carrez, F. Golemo, F. Heide, L. Paull, C. Pal. *CtRL-Sim: Reactive and Controllable Driving Agents with Offline Reinforcement Learning.* CoRL 2024. arXiv:2403.19918.

[30] S. Ettinger et al. *Large Scale Interactive Motion Forecasting for Autonomous Driving: The Waymo Open Motion Dataset.* ICCV 2021. arXiv:2104.10133.

[31] C. Gulino et al. *Waymax: An Accelerated, Data-Driven Simulator for Large-Scale Autonomous Driving Research.* NeurIPS 2023 Datasets & Benchmarks. arXiv:2310.08710.

[32] S. Kazemkhani, A. Pandya, D. Cornelisse, B. Shacklett, E. Vinitsky. *GPUDrive: Data-driven, Multi-Agent Driving Simulation at 1 Million FPS.* ICLR 2025. arXiv:2408.01584.

[33] H. Caesar et al. *nuPlan: A Closed-Loop ML-Based Planning Benchmark for Autonomous Vehicles.* CVPR ADP3 Workshop, 2021. arXiv:2106.11810.

[34] W. Ding, C. Xu, M. Arief, H. Lin, B. Li, D. Zhao. *A Survey on Safety-Critical Driving Scenario Generation — A Methodological Perspective.* IEEE T-ITS 2023. arXiv:2202.02215.

[35] J. Suarez. *PufferLib: Making Reinforcement Learning Libraries and Environments Play Nice.* 2024. ⚠︎ verify exact reference/arXiv id.

[36] S. Suo, K. Wong, J. Xu, J. Tu, A. Cui, S. Casas, R. Urtasun. *MixSim: A Hierarchical Framework for Mixed Reality Traffic Simulation.* CVPR 2023. ⚠︎ verify author list.

[37] A. Nichol, P. Dhariwal. *Improved Denoising Diffusion Probabilistic Models.* ICML 2021. arXiv:2102.09672. (Cosine noise schedule.)

---

## Appendix A — Hyperparameters (real values from the repo configs)

**Table A1 — Stage-2 (DDPO) configuration (`cfgs/ddpo/ldm_adv.yaml`).**

| Group | Key | Value |
|---|---|---|
| Sampling | sampler / steps | DDPM / 100 (DDIM-30 variant) |
| | batch / group size | 128 / 8 (16 contexts) |
| | k_steps (subsampled update steps) | 16 |
| Optim | lr / weight decay / grad clip | 1e-5 / 1e-4 / 1.0 |
| | estimator / clip range | IS (PPO-clip) / 1e-4 |
| | advantage | z-score, per-context, clip ±5 |
| Guards | degenerate group | global whitening (std < 1e-4) |
| | ratio_drop | \|log ρ\| > 1 dropped |
| | KL | closed-form to base; adaptive: target 0.05, band 1.5, rate 1.5, coef ∈ [0.01, 100], init 0.2 |
| | decoupled pg/KL grad clipping | on |
| Pool | size / ego filter | 40 000 scenes / min ego drive 10 m |
| Condition | target | type=vehicle, motion=moving, goal-dist ∈ {middle, far}, ego-dist=null |
| Rollout | steps × dt | 91 × 0.1 s |
| | adversary collision-aversion obs | 0.0 (others 0.5) |
| Reward | ttc_tau / risk_coef | 3.0 s / 1.0 |
| | approach d_safe / d_scale / close_delta / close_scale | 6 / 2 / 2 / 1 m |
| | approach anneal | 1.0 → 0.25 over iters 2k–6k |
| | lane soft/hard/weight | 1.75 m / 2.75 m / 0.25 |
| | overlap penalty weight | 0.5 |
| | reject grading base / scale | 0.5 / 10 m |
| | collision warmup / window | 0.75 s / 0.5 s |
| | collision / ego-fault bonus | 0.5 / 1.0 |
| Eval | scenes (metrics / media) | 64 / 8, every 100 iters |

**Table A2 — Stage-1 (LDM-Adv) configuration.**

| Key | Value |
|---|---|
| latent dims (agent / lane / adv) | 8 / 24 / 8 |
| conditioning vocab | type 3, motion 2, goal-dist 3, ego-dist 3 (+1 null each) |
| cond dropout | 0.2 (adv: per-field independent; agents: per-token) |
| bucket thresholds | goal-dist & ego-dist: near < 10 m, far > 25 m; parked < 2 m |
| train mode / mode probs | mixed / 0.7 init_scene, 0.0 init_agent, 0.3 init_adv |
| loss weights (lane / agent / adv) | 10 / 1 / 1 |
| steps / EMA | 200k / 0.9999 |
| data | WOMD goal-AE latent cache: 487k train / 44k val scenes |

## Appendix B — Reproducibility

```bash
# Stage 1: adversary-aware latent diffusion (reuses the frozen goal AE + latent cache)
python train.py --config-name config_ldm_adv_train

# Stage 2: DDPO fine-tuning of the adversary branch (DDPM; _ddim variant available)
python train.py --config-name config_critical_scene_ldm_adv_ddpo
```

---
---

# ⚠️ Gap Analysis — NOT part of the paper (для submission checklist)

**这一节是给作者的差距清单，投稿前删除。按优先级排序。**

### A. 实验（最大缺口 — 论文目前没有一个成功的主结果）
1. **主结果 run 还没有跑通**:collapse (3zjnupnh) 和 stall (8rlw8ay8) 之后的全 guard + collision bonus + anneal + 非对称 conditioning 的 run 需要真正跑出 "ego collision rate 显著上升且不 collapse" 的曲线。这是整篇文章成立的前提。
2. **Baselines 一个都没有实现/评测**:prior sampling、best-of-N、heuristic 放置、real-adv replay,以及(加分项)STRIVE 式 per-scene 优化。best-of-N 尤其重要——审稿人一定会问"RL 比 sample 8 次取最差情形好在哪"。
3. **多 seed**:至少 3 seeds 的主结果 + 误差棒;RL 论文没有 seed 方差几乎必被拒。
4. **Ablation 矩阵**(Table 3)大部分格子是空的;其中 flat-reject 和 no-bonus 两行已有诊断 run 支撑,其余(fixed KL、no ratio-drop、DDIM、prune-to-ego、对称 conditioning)需要补跑。
5. **跨 planner 迁移**:生成的场景对没见过的 planner(IDM、GPUDrive RL policy、CtRL-Sim agent)是否仍然 critical?这是"生成的是真困难场景"而非"过拟合单个 planner 弱点"的关键证据。
6. **Realism/diversity 量化**:JSD 指标、latent 距离、placement 多样性——代码里 metrics.py 有基础,但没对 adv 生成分布跑过。
7. **eval 协议**:64 val scenes 偏小,建议扩到 ≥512 做终评;固定 scene 集合 + 固定 condition 抽样,报告置信区间。

### B. 方法/写作上的决定
8. **故事定位**:目前有三条可讲的主线——(a) 架构(adv stream + mixed mode + reinsert decode)、(b) reward 设计、(c) §5 的失败模式分析。建议以 (c)+(b) 为主卖点、(a) 为 enabling contribution,因为 (a) 单独看增量偏薄(在 Scenario Dreamer 上加一个 stream)。
9. **方法命名**:LDM-Adv 是代码名;需要一个不撞车的系统名(查 arXiv:AdvDreamer 已被占用)。
10. **§5 的普适性主张**需要至少一个 driving 之外的 toy 验证(比如一个 bandit/toy diffusion 上复现 reward-cliff collapse),否则"we believe both are general"是裸断言。可选但强烈加分。
11. **非对称 conditioning 的伦理/合理性讨论**:adv 变 reckless 是否让"criticality"变得 trivial?需要一段论证(ego 的 avoidance 未变,被测对象不变)+ 一个对照(对称 conditioning 行)。

### C. 图表
12. 所有 8 张图都是 placeholder;图 1(teaser)、图 5(训练曲线)、图 6(qualitative)优先。图 5/8 的原始数据在 wandb runs 3zjnupnh、8rlw8ay8 里已经存在,可以先做。
13. 表 1/2/3 的数字全部待填。

### D. 引用核对(我标了 ⚠︎ 的)
14. [22] DiffScene 的 venue、[35] PufferLib 的正式引用、[36] MixSim 作者表需要核对;其余引用的 arXiv 号我有较高把握,但投稿前请全部过一遍 Google Scholar/官方 bib。

### E. 工程/基础设施
15. `research/root.tex` 是空的:需要选定 venue(CVPR/ICLR/CoRL?格式差异大)并搭 LaTeX 模板,把本 markdown 迁移过去。
16. 复现包:checkpoint 与 latent cache 的发布计划、去除 wandb/scratch 路径硬编码、smoke_test_ddpo.py 目前已损坏(引用被删的 PufferDriveReward)。
17. **数据许可**:WOMD 许可证禁止再分发衍生数据,latent cache 只能给"生成脚本"不能直接发布——写 repo release 计划时注意。
