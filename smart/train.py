#!/usr/bin/env python
"""Train the SMART-style traffic model.

Plain torch on purpose: the Lightning stack in this repo exists for the
diffusion pipeline, and this package is meant to stay independent of it. The
only thing shared with the main repo is the observation spec and the action
table, both of which are shared by construction rather than by convention.

Two losses, and they do different jobs:

  * **cross-entropy** on the cached closed-loop action labels
    (``smart/preprocess.py``). Per step, and it keeps the action distribution
    sharp -- it is a classification loss, so it does not blur distinct futures
    together.
  * **trajectory** (``smart.trajectory``): the model's own actions are integrated
    K steps forward and the resulting poses are scored, in metres, against the
    logged ones. This is the term that sees ACCUMULATED drift, which per-step
    cross-entropy is blind to.

Cross-entropy stays primary and the trajectory term is a correction on top.
Measured dynamic range of the trajectory term at K=10 is 0.086 m (label actions)
to 0.259 m (uniform policy), so it is a refinement signal, not a driver -- but it
is the only one aimed at what the model is actually asked to produce. Longer
chains widen that range (1.33 m at K=40) at the cost of averaging genuinely
distinct futures together, which is why the default is 1 s rather than 4.

The visible history length is randomly masked by the dataset, so the model is
trained to drive from an empty past as well as a full one.

Top-1 accuracy over 91 actions understates quality -- neighbouring accel/steer
cells are nearly equivalent motions -- so treat it as a training signal, not as
the evaluation. Watch the val trajectory distance instead, and remember that the
evaluation that counts is a closed-loop rollout scored the way every other
planner is (``scripts/score_paired_sources.py``).

Usage:
    python smart/train.py --steps 20000 --out checkpoints/planners/smart/v1.pt
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smart.dataset import SMARTScenes, collate
from smart.net import NUM_ACTIONS, OBS_DIM, NetConfig, SMARTTrafficNet
from smart.trajectory import ActionTable, rollout_loss

DT = 0.1  # cfgs/rollout/base.yaml


def _parse():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-scenes", type=int, default=32,
                    help="scenes per step; rows per step is this x steps-per-scene x agents")
    ap.add_argument("--chain-steps", type=int, default=10,
                    help="length of the integrated chain, in 0.1 s sim steps")
    ap.add_argument("--starts-per-scene", type=int, default=2)
    ap.add_argument("--perturb-prob", type=float, default=0.0,
                    help="fraction of chains synthesised off the log so the model is "
                         "taught to recover from its own drift (0 = plain behaviour cloning)")
    ap.add_argument("--perturb-lat", type=float, default=0.5, help="metres, std")
    ap.add_argument("--perturb-yaw-deg", type=float, default=3.0)
    ap.add_argument("--traj-weight", type=float, default=1.0,
                    help="weight of the trajectory term (metres) against cross-entropy (nats)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--hidden-size", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wandb", action="store_true", help="log to wandb")
    ap.add_argument("--wandb-project", default="smart-traffic",
                    help="kept separate from the diffusion repo's 'scenario-dreamer'")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--viz-scenes", type=int, default=6,
                    help="scenes rolled out and drawn into wandb at each validation")
    return ap.parse_args()


def _loader(split, args, shuffle):
    ds = SMARTScenes(split, chain_steps=args.chain_steps,
                     starts_per_scene=args.starts_per_scene, seed=args.seed,
                     # validation stays on the true logged chains, or the metric
                     # would move whenever the perturbation settings move
                     perturb_prob=args.perturb_prob if shuffle else 0.0,
                     perturb_lat=args.perturb_lat, perturb_yaw_deg=args.perturb_yaw_deg)
    return ds, DataLoader(
        ds, batch_size=args.batch_scenes, shuffle=shuffle, collate_fn=collate,
        num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0,
    )


def _endless(loader):
    while True:
        yield from loader


def forward_batch(net, batch, table, device):
    """(cross-entropy, trajectory distance in m, logits [N, K, 91], flat targets)."""
    obs, act, s0, ref, ln, wd = (t.to(device, non_blocking=True) for t in batch)
    n, k = act.shape
    logits = net(obs.reshape(n * k, OBS_DIM)).reshape(n, k, NUM_ACTIONS)
    ce = torch.nn.functional.cross_entropy(logits.reshape(n * k, NUM_ACTIONS),
                                           act.reshape(n * k))
    traj, _ = rollout_loss(logits.float(), s0, ref, ln, wd, table, DT)
    return ce, traj, logits, act.reshape(n * k)


@torch.no_grad()
def evaluate(net, loader, table, device, batches):
    net.eval()
    n = correct = top5 = 0
    ce_sum = traj_sum = 0.0
    for i, batch in enumerate(loader):
        if i >= batches or not batch[0].numel():
            break
        ce, traj, logits, target = forward_batch(net, batch, table, device)
        flat = logits.reshape(len(target), -1)
        ce_sum += float(ce) * len(target)
        traj_sum += float(traj) * len(target)
        correct += int((flat.argmax(-1) == target).sum())
        top5 += int((flat.topk(5, dim=-1).indices == target[:, None]).any(-1).sum())
        n += len(target)
    net.train()
    return ce_sum / max(n, 1), traj_sum / max(n, 1), correct / max(n, 1), top5 / max(n, 1)


def main() -> int:
    args = _parse()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    _, train_loader = _loader("train", args, shuffle=True)
    _, val_loader = _loader("val", args, shuffle=False)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.run_name, config=vars(args))

    table = ActionTable(device)
    net = SMARTTrafficNet(NetConfig(
        weights="random", device=device, hidden_size=args.hidden_size,
        num_layers=args.num_layers, num_heads=args.num_heads, seed=args.seed,
    )).to(device)
    print(f"params: {sum(p.numel() for p in net.parameters()):,}  device: {device}")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    def lr_at(step):
        if step < args.warmup:
            return step / max(args.warmup, 1)
        p = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    t0 = time.perf_counter()

    net.train()
    for step, batch in enumerate(_endless(train_loader), start=1):
        if step > args.steps:
            break
        if not batch[0].numel():
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            ce, traj, _, target = forward_batch(net, batch, table, device)
        loss = ce + args.traj_weight * traj
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 100 == 0:
            s_per_step = (time.perf_counter() - t0) / step
            print(f"step {step:>6}  ce {float(ce):.4f}  traj {float(traj):.4f} m  "
                  f"rows {len(target):>5}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{s_per_step:.3f} s/step")
            if run is not None:
                run.log({"train/ce": float(ce), "train/traj_m": float(traj),
                         "train/rows": len(target), "train/lr": sched.get_last_lr()[0],
                         "perf/s_per_step": s_per_step}, step=step)
        if step % args.val_every == 0 or step == args.steps:
            vce, vtraj, acc, acc5 = evaluate(net, val_loader, table, device, args.val_batches)
            print(f"  [val] ce {vce:.4f}  traj {vtraj:.4f} m  "
                  f"top1 {100 * acc:.2f}%  top5 {100 * acc5:.2f}%")
            if run is not None:
                payload = {"val/ce": vce, "val/traj_m": vtraj,
                           "val/top1": acc, "val/top5": acc5}
                # A rollout picture says things the scalars cannot: the model can
                # have a fine cross-entropy and still drive off a curve.
                try:
                    import wandb
                    from smart.viz import render
                    net.eval()
                    fig = render(None, "val", args.viz_scenes, None, net=net)
                    payload["val/rollout"] = wandb.Image(fig)
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                    net.train()
                except Exception as exc:                       # viz must never kill a run
                    print(f"  [viz] skipped: {exc}")
                run.log(payload, step=step)
            vl = vce + args.traj_weight * vtraj
            if vl < best:
                best = vl
                torch.save({
                    "state_dict": net.state_dict(),
                    "arch": {"hidden_size": args.hidden_size, "num_layers": args.num_layers,
                             "num_heads": args.num_heads},
                    "step": step, "val_loss": vl, "val_ce": vce, "val_traj_m": vtraj,
                }, out)
                print(f"  [val] new best -> {out}")
    if run is not None:
        run.summary["best_val_loss"] = best
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
