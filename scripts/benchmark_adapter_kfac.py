#!/usr/bin/env python3
"""
Compare per-step wall time (overhead) and short-run loss stability between
adapter_diag and ASDL K-FAC adapter optimizers on the same synthetic batches.

Usage (from repo root):
  PYTHONPATH=. python scripts/benchmark_adapter_kfac.py --config configs/sst2.yaml --steps 30 --warmup 3
"""

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import statistics
import time

import torch
import yaml

from src.model import build_model
from src.optimizers import get_optimizers
from src.trainer import set_seed, uses_asdl_kfac
from src.kfac import build_kfac_gradient_maker, kfac_forward_backward


def fake_batch(batch_size, seq_len, device):
    return {
        "input_ids": torch.randint(0, 30522, (batch_size, seq_len), device=device),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
        "labels": torch.randint(0, 2, (batch_size,), device=device),
    }


def run_steps(cfg, device, optimizer_name, steps, warmup):
    cfg = dict(cfg)
    cfg["optimizer"] = optimizer_name
    set_seed(cfg["seed"])
    model = build_model(cfg).to(device)
    optimizers = get_optimizers(model, cfg)
    total_steps = steps * 2  # dummy schedule length; only used by K-FAC maker
    gm = None
    if uses_asdl_kfac(cfg):
        gm = build_kfac_gradient_maker(model, cfg, total_steps)

    times = []
    losses = []

    model.train()
    for i in range(warmup + steps):
        batch = fake_batch(cfg["batch_size"], cfg["max_length"], device)
        t0 = time.perf_counter()
        if gm is not None:
            _, loss = kfac_forward_backward(gm, model, batch)
        else:
            out = model(**batch)
            loss = out.loss
            loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            cfg["grad_clip"],
        )
        for opt in optimizers.values():
            opt.step()
        for opt in optimizers.values():
            opt.zero_grad()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            times.append(elapsed_ms)
            losses.append(loss.detach().float().cpu().item())

    return {
        "mean_ms": statistics.mean(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "loss_mean": statistics.mean(losses),
        "loss_stdev": statistics.stdev(losses) if len(losses) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps", "auto"),
        default="cpu",
        help="`auto` picks CUDA, then MPS, then CPU.",
    )
    parser.add_argument(
        "--methods",
        default="adapter_diag,adapter_kfac",
        help="Comma-separated optimizers to compare: adapter_diag, adapter_kfac.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    else:
        device = torch.device(args.device)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    allowed = {"adapter_diag", "adapter_kfac"}
    unknown = set(methods) - allowed
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}")
    results = {}
    for name in methods:
        results[name] = run_steps(base_cfg, device, name, args.steps, args.warmup)

    ref = "adapter_diag" if "adapter_diag" in results else methods[0]
    baseline = results[ref]["mean_ms"]
    print(f"Device: {device}")
    print(f"Reference for ratio: {ref} ({baseline:.2f} ms/step)")
    print(f"Timed steps (after {args.warmup} warmup): {args.steps}")
    print()
    for name in methods:
        r = results[name]
        overhead = r["mean_ms"] / baseline if baseline > 0 else float("nan")
        print(
            f"{name:16s}  step={r['mean_ms']:7.2f}±{r['stdev_ms']:5.2f} ms  "
            f"ratio_vs_{ref}={overhead:5.2f}x  "
            f"batch_loss_mean={r['loss_mean']:.4f} σ={r['loss_stdev']:.4f}"
        )


if __name__ == "__main__":
    main()
