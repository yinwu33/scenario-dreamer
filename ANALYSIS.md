我的判断是：**当前不收敛的第一原因更像 reward 语义和 DDPO advantage 设计，而不是“同时生成 position、heading、goal”本身。**后者会放大问题，但不是最先该改的地方。

当前 reward 最可疑的地方
1. collision 判断在 init_invalid 前面

你现在是：

if ego_collision:
    base = 1.0
elif init_invalid:
    base = -1.0

如果 ego_collision 包括首帧或接近 t=0 的碰撞，那么“初始 overlap → collision”可能先命中 ego_collision，反而拿到正奖励。

应该首先处理 hard invalid：

if init_invalid:
    return -1.0
elif ego_collision:
    ...

并且不只是 initial overlap，过近出生后在 0.1 秒碰撞也属于 trivial collision。建议单独设置：

trivial_collision = collision and collision_time < 0.75

这种碰撞不应得到正奖励。

2. 你的 penalty 是为 multi-agent 写的，one-agent 下退化成了离散跳变

如果 goal_offlane_frac 的分母是 controlled agents 数量，那么 one-agent 下它通常就是：

goal_offlane_frac∈{0,1}

parking_mismatch_frac 和 controlled_parking_frac 也类似。

于是一个样本可能因为 goal 稍微越过 threshold，reward 瞬间下降 1.0。相比之下：

TTC = 1.5 秒只有 +0.5；
距离 5 米只有 +0.25；
parking mismatch 直接 -0.5。

这使得训练首先在学习“不要触发离散 penalty”，而不是学习 criticality。等 valid rate 稍微提高后，大部分 reward 又集中到 0 附近，advantage 方差突然变小，PPO 更新开始被噪声支配。

特别是 parking_mismatch 如果指相对 GT 的 mismatch，它会直接阻止模型探索和 GT 不同的 critical configuration。对 adversarial generation 来说，这是目标冲突。

建议：

one-agent 阶段直接删除 parking_mismatch_penalty；
controlled adversary 固定为 dynamic/non-parked，不让 diffusion 生成 parking 状态；
删除 controlled_parking_penalty；
将 binary goal_offlane_frac 改成连续 lane-distance penalty。
3. distance_bonus 很可能在鼓励你看到的“固定几个 ego 附近位置”

当前：

R
dist
	​

=0.5clip(1−
10
d
min
	​

	​

,0,1)

它没有区分：

agent 一开始就靠近 ego；
agent 在交互过程中逐渐接近 ego；
agent 和 ego 并行但很近；
agent 真正处于 closing/conflict 状态。

所以最简单的刷分方式就是：生成在 ego 附近几个高概率、lane-valid 的位置。

这既会降低空间 diversity，也会和 diffusion pretrained prior 的高频位置叠加，形成 mode collapse。

应该把它替换为 approach bonus：

R
approach
	​

=σ(
s
d
	​

d
safe
	​

−d
min
	​

	​

)⋅σ(
s
a
	​

d
0
	​

−d
min
	​

−Δd
min
	​

	​

)

其中：

d
0
	​

：初始 oriented-box clearance；
d
min
	​

：忽略前 0.5 秒后的最小距离；
d
0
	​

−d
min
	​

：agent 是否真的在 rollout 中接近了 ego。

这样 agent 只是在初始位置靠近，但没有形成 closing interaction，就拿不到高分。

4. collision reward 没有区分碰撞质量

目前任何合法 collision 都是 +1：

0.1 秒后碰撞；
5 秒后发生合理 crossing collision；
低速贴蹭；
高相对速度侧碰；
ego 主动撞到静止车辆；

都一样。

这会导致 reward landscape 很尖，也很容易被 hack。

建议 collision 只是额外 bonus，而不是覆盖 TTC 的分支：

R
collision
	​

=1
collision
	​

⋅σ(
s
t
	​

t
collision
	​

−t
warmup
	​

	​

)

例如：

t
warmup
	​

=0.75s；
t
collision
	​

<0.5s：负奖励；
t
collision
	​

>1s：才逐渐获得 collision bonus。

并且 collision bonus 不需要太大，因为 collision 样本本身已经会有高 proximity、低 TTC、强 ego response。

我建议的第一版 reward

先不要一次加入过多复杂指标。可以改成：

R={
−1,
clip(0.25R
conflict
	​

+0.40R
risk
	​

+0.20R
ego
	​

+0.15R
collision
	​

−0.40C
lane
	​

−0.20C
heading
	​

−0.20C
route
	​

−0.50C
trivial
	​

,−1,1),
	​

hard invalid
otherwise
	​


这些初始权重只是合理起点，真正训练时建议做 ablation。

R
conflict
	​

：interaction potential

你提出“agent 的 init-goal 和 ego init-goal 是否有交叉”，方向是对的，但不要直接判断两条起终点直线是否相交。直线会穿过建筑物、逆向车道和不可行区域。

应该先在 lane graph 上得到：

ego 从 init 到 goal 的可行 route；
adversary 从 init 到 goal 的可行 route。

然后判断：

两条 route centerline 是否相交；
是否共享同一 lane segment；
是否存在 merge；
route corridor 是否重叠；
是否存在 oncoming conflict。

再计算到 conflict point 的 time-to-arrival：

ΔTTA=∣TTA
ego
	​

−TTA
adv
	​

∣

定义：

R
conflict
	​

=1
route conflict
	​

exp(−
2τ
TTA
2
	​

ΔTTA
2
	​

)

这个项非常适合作为早期 dense shaping：即使 planner 最终避让成功、没有产生有效 TTC，只要两者在路线和时间上形成冲突，就会有梯度信号。

Safe-Sim 采用了类似的结构化思路：先搜索 ego 和 adversarial agent 的 lane centerline intersection，再利用相对 conflict-point 距离、加速度和 lateral offset 构造危险交互，而不是只依赖最终碰撞距离；它还把 TTC、route guidance 和 collision 分开建模。

R
risk
	​

：动态风险

先不要只用一个 hard minTTC。建议：

R
TTC
	​

=σ(
s
TTC
	​

T
safe
	​

−TTC
soft
	​

	​

)

其中：

TTC
soft
	​

=−τlog(
T
1
	​

t
∑
	​

e
−TTC
t
	​

/τ
)

只对以下 timestep 计算 TTC：

relative closing speed 大于阈值；
不是并排行驶；
predicted distance at closest approach 足够小。

再定义前面的 approach reward：

R
approach
	​

=σ(
s
d
	​

d
safe
	​

−d
min
	​

	​

)σ(
s
a
	​

d
0
	​

−d
min
	​

−Δd
min
	​

	​

)

组合时避免重复叠加：

R
risk
	​

=1−(1−R
TTC
	​

)(1−R
approach
	​

)

这样：

有 TTC：有信号；
TTC 不成立但明显快速接近：仍然有信号；
只是出生很近：approach gate 会抑制；
并行近距离：closing gate 会抑制。
R
ego
	​

：真正对 planner 造成了什么影响

criticality 最好不仅是几何接近，还包括 ego planner 的实际反应：

R
ego
	​

=0.5R
brake
	​

+0.5R
progress-loss
	​


例如：

R
brake
	​

=σ(
s
a
	​

max
t
	​

(−a
ego,t
	​

)−a
0
	​

	​

)

更推荐做 counterfactual：

rollout A：有 generated adversary；
rollout B：没有 generated adversary，其余条件相同。

定义：

R
cf-progress
	​

=clip(
L
scale
	​

progress
without
	​

−progress
with
	​

	​

,0,1)

以及：

R
cf-collision
	​

=max(collision
with
	​

−collision
without
	​

,0)

你的 ego、map、ego goal 都固定，所以 without-adversary rollout 可以对每个 context 只计算一次并缓存，不必每个 generated sample 都重新计算。

这会大幅改善 reward 的因果语义：

奖励的是 generated agent 导致的 planner degradation，而不是 planner 在这个地图上本来就会失败。

这也比单纯 minTTC 更像一个 paper-level 的 criticality definition。

lane、heading 和 goal 应该怎么处理
Goal 是否应该离 init 足够远？

应该，但不要使用：

+α⋅init-goal distance

否则模型会把 goal 一直推到最远处。

应当把它作为区间约束：

L
min
	​

≤L
route
	​

≤L
max
	​


这里必须是 lane-graph route length，不是 Euclidean distance。

可以定义：

C
route
	​

=clip(
L
min
	​

L
min
	​

−L
	​

,0,1)+clip(
L
max
	​

L−L
max
	​

	​

,0,1)

其中：

goal route 不可达：hard invalid；
route 太短：agent 很快停车或几乎不运动；
route 太长：goal 对当前 simulation horizon 没意义；
goal 在 init 后方且要求车辆 U-turn：通常 hard invalid。

还可以检查 rollout 中 adversary 的 route progress。controlled agent 完全不动时，应当受到 progress penalty，而不是 parking penalty。

是否惩罚 heading 与 lane 不一致？

应该，而且这个比 parking penalty 更重要。

连续 penalty 可以写成：

C
heading
	​

=
2
1−cos(Δθ)
	​


或者使用 margin：

0∼10
∘
：不罚；
10
∘
∼45
∘
：平滑增加；
>45
∘
：hard invalid。

车辆的 heading 最好直接来自 lane tangent：

θ
agent
	​

=θ
lane
	​

+Δθ

diffusion 只生成一个较小的 heading residual，而不是完整的绝对 heading。

同时注意不要直接 diffusion 一个 [−π,π] 的 raw angle，因为在 −π/π 边界上不连续。使用：

(sinθ,cosθ)

或者 lane-relative angle residual。

Goal offlane 应该怎么改？

不要再使用 binary goal_offlane_frac。可以使用到合法 lane/route corridor 的连续距离：

C
goal-lane
	​

=smoothstep(d
goal-lane
	​

;d
soft
	​

,d
hard
	​

)

例如概念上：

小于 0.5 米：0；
0.5–2 米：平滑增加；
超过 2 米或找不到 reachable lane：hard invalid。

start lane 和 goal lane 应该分别记录，不要合成一个 fraction。

position、heading 和 goal 联合生成是否高度耦合？

**联合建模本身是正确的。**一个自然场景本来就要求：

p(position,heading,goal∣map,ego)

而不是三个独立分布。

问题在于你当前可能是在 raw Euclidean space 里同时 diffusion：

(x,y,θ,g
x
	​

,g
y
	​

)

这里混合了三类不同结构：

x,y：连续空间；
heading：圆周变量；
goal：lane graph 上的拓扑变量。

再加上 DDPO 只有一个 terminal reward，模型很难知道 reward 变化应归因于 position、heading 还是 goal。

我建议的 one-agent curriculum

第一阶段只生成：

(lane anchor,s,d,v)

其中：

heading 由 lane tangent 决定；
goal 从当前 lane graph 上向前采样 reachable route；
controlled agent 固定 dynamic；
type 固定为 vehicle。

第二阶段再生成：

(goal lane/route branch,L
route
	​

)

第三阶段再加入：

Δθ,d
goal
	​


这仍然属于 data-space diffusion，只是换成了 structured map-relative data space。

最有说服力的 ablation 是：

position only，heading/goal 派生；
position + goal，heading 派生；
position + heading + goal 全部生成。

如果 1 稳定、2 尚可、3 collapse，才有充分证据说明 joint action space 的 credit assignment 是问题。

更重要的 DDPO 训练问题：必须按 context 归一化 advantage

不同 map/ego condition 的 criticality 难度差别很大：

有些 intersection 很容易形成冲突；
有些 straight road 几乎没有可行 conflict route；
有些 ego planner 本身速度很低；
有些场景 horizon 内根本不会交互。

如果你全 batch 做 reward normalization，那么模型学到的可能主要是“哪些 context 天然 reward 高”，而不是“同一个 context 下哪个生成结果更 critical”。

原始 DDPO 明确采用了 per-context reward normalization，并指出 PPO-style update 需要很小的 clip；论文实验中甚至使用了 10
−4
 的 clip range。

对你来说，一个 training group 应该是：

固定同一个 map；
固定 ego init；
固定 ego goal；
从 diffusion 采样 K=8∼16 个 adversary。

然后：

A
i
	​

=
std
k
	​

R
k
	​

+ϵ
R
i
	​

−mean
k
	​

R
k
	​

	​


更稳的是 rank advantage：

A
i
	​

=2
K−1
rank(R
i
	​

)−1
	​

−1

如果组内所有 reward 相同或标准差过小，这组直接跳过 policy update。

最好进一步分开归一化：

A=A
criticality
	​

−λA
constraint
	​


不要先把所有 penalty 和 criticality 加成 total reward，再统一 z-score。否则一个二值 offlane penalty 很容易支配全部 advantage。

把 validity 当成 constraint，而不是和 criticality 混成一个手调总分

从 paper 角度，我更推荐把方法写成 constrained DDPO：

θ
max
	​

E
x∼p
θ
	​

	​

[R
critical
	​

(x)]

subject to：

E[C
init-invalid
	​

]≤ϵ
1
	​

E[C
route-invalid
	​

]≤ϵ
2
	​

E[KL(p
θ
	​

∥p
ref
	​

)]≤κ

然后使用 Lagrangian：

R
effective
	​

=R
critical
	​

−λ
1
	​

C
invalid
	​

−λ
2
	​

C
route
	​


并动态更新 λ，而不是永久使用 1.0, 0.5, 0.5 这种固定系数。

这也更符合 safety-critical generation 的问题结构：criticality 是 objective，lane validity、functionality 和 realism 是 constraints。DiffScene 同样明确区分 safety、functionality 和 constraint objectives，而不是只使用单一碰撞分数。

一定要加 reference regularization

建议训练目标至少是：

L=L
DDPO
	​

+βKL(π
θ
	​

(⋅∣x
t
	​

,c)∥π
ref
	​

(⋅∣x
t
	​

,c))+μL
denoise-data
	​


其中：

pi_ref 是 frozen pretrained diffusion；
每若干次 DDPO update，混入一批 Waymo data 做原始 denoising loss；
只更新 LoRA、adapter 或后几层；
map encoder 和 fixed ego conditioning 先冻结。

DPOK 的核心做法之一就是在 diffusion RL fine-tuning 中加入 KL regularization，以限制模型偏离 pretrained distribution。

还要检查几个可能的实现问题
inpainting log-prob mask
DDPO 的 log probability 只能对 generated dimensions 求和。固定的 ego init、ego goal 或被 overwrite 的维度不能参与 policy ratio。
sampler 必须有非零 stochastic variance
如果使用 deterministic DDIM、eta=0，或者 reverse variance 极小，DDPO ratio 很容易不稳定。
所有生成变量是否真的被加噪
检查 goal、heading 是否误用了 inpainting mask，导致它们事实上被固定。
特征 scale
x,y,g
x
	​

,g
y
	​

 是几十米，heading 是弧度。所有字段必须单独标准化到近似 unit variance。
reward 是否可复现
对完全相同的生成场景重复 rollout 10 次。如果 reward 标准差明显，先固定 planner seed 或使用 common random numbers。
更新是否过多
一批 on-policy samples 不要反复 PPO 很多 epoch。否则在 terminal noisy reward 下很容易过拟合这一批样本。
你现在应该记录的指标

不要再把 reward > 0 叫做 critical_rate。至少分开记录：

valid_init_rate
reachable_goal_rate
heading_valid_rate
route_conflict_rate
tta_conflict_rate
near_miss_rate
meaningful_collision_rate
trivial_collision_rate
ego_brake_rate
ego_progress_loss
start_lane_distance
goal_lane_distance
position_mode_entropy
KL_to_reference

并且每个 reward component 都记录：

mean；
std；
非零比例；
和 total reward 的 correlation；
top-10% reward 样本中该项的平均值。

你很可能会发现，当前训练早期 advantage 主要由 init_invalid 和 goal_offlane_frac 决定；后期则由 distance_bonus 决定，而不是 minTTC。