#!/usr/bin/env python
"""Train the joint (SMART-structured) traffic net.

Cross-entropy only, over the agents that have a label at the sampled timestep.
No trajectory term: the joint model needs one full-scene forward per chain step,
and adding it would change two things at once against the per-agent baseline.

Watch ``val/neutral`` as closely as the loss. The per-agent model reached a
respectable cross-entropy while emitting the do-nothing action for 78.9% of
rollout decisions -- constant velocity in a straight line, which is why its
agents drove through each other and off curves. A model whose argmax is neutral
four times out of five has not learned to drive no matter what its loss says.

Usage:
    python smart/joint_train.py --steps 20000 --out checkpoints/planners/smart/joint_v1.pt
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

from smart.joint_dataset import JointScenes, collate
from smart.joint_net import JointNetConfig, JointTrafficNet

NUM_STEER = 13
NEUTRAL_ACTION = 3 * NUM_STEER + 6      # accel -0.0, steer 0.0


def _parse():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-scenes", type=int, default=8)
    ap.add_argument("--steps-per-scene", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--hidden-size", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="smart-traffic")
    ap.add_argument("--run-name", default=None)
    return ap.parse_args()


def _loader(split, args, shuffle):
    ds = JointScenes(split, steps_per_scene=args.steps_per_scene, seed=args.seed)
    return DataLoader(ds, batch_size=args.batch_scenes, shuffle=shuffle,
                      collate_fn=collate, num_workers=args.workers, drop_last=True,
                      persistent_workers=args.workers > 0)


def _endless(loader):
    while True:
        yield from loader


def _loss(net, batch, device):
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    logits = net(batch)
    sel = batch["target_valid"]
    if not sel.any():
        return None, None, None
    return (torch.nn.functional.cross_entropy(logits[sel], batch["action"][sel]),
            logits[sel], batch["action"][sel])


@torch.no_grad()
def evaluate(net, loader, device, batches):
    net.eval()
    n = correct = top5 = neutral = 0
    loss_sum = 0.0
    for i, batch in enumerate(loader):
        if i >= batches:
            break
        if batch is None:
            continue
        ce, logits, target = _loss(net, batch, device)
        if ce is None:
            continue
        pred = logits.argmax(-1)
        loss_sum += float(ce) * len(target)
        correct += int((pred == target).sum())
        top5 += int((logits.topk(5, -1).indices == target[:, None]).any(-1).sum())
        neutral += int((pred == NEUTRAL_ACTION).sum())
        n += len(target)
    net.train()
    m = max(n, 1)
    return loss_sum / m, correct / m, top5 / m, neutral / m


def main() -> int:
    args = _parse()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    train_loader, val_loader = _loader("train", args, True), _loader("val", args, False)

    net = JointTrafficNet(JointNetConfig(
        "random", device, args.hidden_size, args.num_layers, args.num_heads, args.seed
    )).to(device)
    for p in net.parameters():
        p.requires_grad_(True)
    print(f"params: {sum(p.numel() for p in net.parameters()):,}  device: {device}")

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    def lr_at(step):
        if step < args.warmup:
            return step / max(args.warmup, 1)
        p = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    best, t0 = float("inf"), time.perf_counter()
    net.train()
    for step, batch in enumerate(_endless(train_loader), start=1):
        if step > args.steps:
            break
        if batch is None:
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            ce, _, target = _loss(net, batch, device)
        if ce is None:
            continue
        opt.zero_grad(set_to_none=True)
        ce.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 100 == 0:
            sps = (time.perf_counter() - t0) / step
            print(f"step {step:>6}  ce {float(ce):.4f}  agents {len(target):>5}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {sps:.3f} s/step")
            if run:
                run.log({"train/ce": float(ce), "train/agents": len(target),
                         "train/lr": sched.get_last_lr()[0], "perf/s_per_step": sps}, step=step)
        if step % args.val_every == 0 or step == args.steps:
            vce, acc, acc5, neu = evaluate(net, val_loader, device, args.val_batches)
            print(f"  [val] ce {vce:.4f}  top1 {100*acc:.2f}%  top5 {100*acc5:.2f}%  "
                  f"neutral {100*neu:.1f}%")
            if run:
                run.log({"val/ce": vce, "val/top1": acc, "val/top5": acc5,
                         "val/neutral": neu}, step=step)
            if vce < best:
                best = vce
                torch.save({"state_dict": net.state_dict(),
                            "arch": {"hidden_size": args.hidden_size,
                                     "num_layers": args.num_layers,
                                     "num_heads": args.num_heads},
                            "step": step, "val_ce": vce, "val_neutral": neu}, out)
                print(f"  [val] new best -> {out}")
    if run:
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
