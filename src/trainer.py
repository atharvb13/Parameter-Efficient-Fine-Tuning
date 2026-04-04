import json
import os
import platform
import random
import resource
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.data import build_dataloaders
from src.metrics import compute_metrics, selection_metric
from src.model import build_model
from src.optimizers import get_optimizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_name):
    if device_name != "auto":
        return torch.device(device_name)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch_to_device(batch, device):
    # DistilBERT does not need token_type_ids
    return {
        k: v.to(device)
        for k, v in batch.items()
        if k != "token_type_ids"
    }


def get_process_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_scale = 1.0 / (1024 * 1024) if platform.system() == "Darwin" else 1.0 / 1024
    return rss * rss_scale


def reset_accelerator_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_accelerator_peak_memory_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return None


def build_stability_tracker(cfg):
    return {
        "spike_threshold": cfg.get("loss_spike_ratio", 1.5),
        "unstable_threshold": cfg.get("unstable_loss_ratio", 2.0),
        "non_finite_steps": 0,
        "loss_spike_steps": 0,
        "unstable_steps": 0,
        "largest_loss_increase_ratio": 1.0,
        "peak_grad_norm": 0.0,
        "diverged": False,
        "divergence_reason": None,
        "previous_loss": None,
    }


def update_stability_tracker(tracker, loss_value, grad_norm):
    if not np.isfinite(loss_value):
        tracker["non_finite_steps"] += 1
        tracker["diverged"] = True
        tracker["divergence_reason"] = "non_finite_loss"
        return

    if not np.isfinite(grad_norm):
        tracker["diverged"] = True
        tracker["divergence_reason"] = "non_finite_grad_norm"
        return

    tracker["peak_grad_norm"] = max(tracker["peak_grad_norm"], float(grad_norm))

    prev_loss = tracker["previous_loss"]
    if prev_loss is not None and prev_loss > 0:
        increase_ratio = loss_value / prev_loss
        tracker["largest_loss_increase_ratio"] = max(
            tracker["largest_loss_increase_ratio"],
            float(increase_ratio),
        )
        if increase_ratio >= tracker["spike_threshold"]:
            tracker["loss_spike_steps"] += 1
        if increase_ratio >= tracker["unstable_threshold"]:
            tracker["unstable_steps"] += 1

    tracker["previous_loss"] = float(loss_value)


@torch.no_grad()
def evaluate(model, dataloader, device, task):
    model.eval()

    total_loss = 0.0
    total_examples = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        outputs = model(**batch)

        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

        bs = batch["labels"].size(0)
        total_loss += loss.item() * bs
        total_examples += bs

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch["labels"].cpu().tolist())

    metrics = compute_metrics(task, all_preds, all_labels)
    metrics["loss"] = total_loss / total_examples
    return metrics


def train(cfg):
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)

    device = get_device(cfg["device"])
    run_start_time = time.time()

    train_loader, val_loader, tokenizer = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    optimizer = get_optimizer(model, cfg)

    total_steps = len(train_loader) * cfg["epochs"]
    warmup_steps = int(cfg["warmup_ratio"] * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history = []
    best_score = -1.0
    run_stability = build_stability_tracker(cfg)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    run_accelerator_peak_memory_mb = 0.0

    print("\nTrainable parameters:")
    model.print_trainable_parameters()
    reset_accelerator_peak_memory(device)

    for epoch in range(cfg["epochs"]):
        model.train()
        start_time = time.time()
        running_loss = 0.0
        epoch_stability = build_stability_tracker(cfg)
        epoch_step_time_total = 0.0
        epoch_step_time_peak = 0.0
        epoch_process_rss_peak_mb = get_process_rss_mb()
        steps_completed = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")

        for step, batch in enumerate(pbar, start=1):
            step_start_time = time.time()
            batch = move_batch_to_device(batch, device)

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params,
                cfg["grad_clip"],
            )
            loss_value = float(loss.item())
            grad_norm_value = float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)

            update_stability_tracker(epoch_stability, loss_value, grad_norm_value)
            update_stability_tracker(run_stability, loss_value, grad_norm_value)
            if epoch_stability["diverged"] or run_stability["diverged"]:
                print(
                    f"Stopping early due to divergence: "
                    f"{epoch_stability['divergence_reason'] or run_stability['divergence_reason']}"
                )
                break

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step_time = time.time() - step_start_time
            epoch_step_time_total += step_time
            epoch_step_time_peak = max(epoch_step_time_peak, step_time)
            epoch_process_rss_peak_mb = max(epoch_process_rss_peak_mb, get_process_rss_mb())
            steps_completed += 1

            running_loss += loss_value
            pbar.set_postfix(
                train_loss=f"{running_loss / step:.4f}",
                spikes=epoch_stability["loss_spike_steps"],
            )

        if epoch_stability["diverged"]:
            break

        val_metrics = evaluate(model, val_loader, device, cfg["task"])
        train_loss = running_loss / max(steps_completed, 1)
        epoch_time = time.time() - start_time
        accelerator_peak_memory_mb = get_accelerator_peak_memory_mb(device)
        if accelerator_peak_memory_mb is not None:
            run_accelerator_peak_memory_mb = max(
                run_accelerator_peak_memory_mb,
                accelerator_peak_memory_mb,
            )

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "epoch_time_sec": round(epoch_time, 2),
            "steps_completed": steps_completed,
            "avg_step_time_ms": round(1000 * epoch_step_time_total / max(steps_completed, 1), 2),
            "peak_step_time_ms": round(1000 * epoch_step_time_peak, 2),
            "process_peak_rss_mb": round(epoch_process_rss_peak_mb, 2),
            "accelerator_peak_memory_mb": (
                round(accelerator_peak_memory_mb, 2)
                if accelerator_peak_memory_mb is not None
                else None
            ),
            "loss_spike_steps": epoch_stability["loss_spike_steps"],
            "unstable_steps": epoch_stability["unstable_steps"],
            "non_finite_steps": epoch_stability["non_finite_steps"],
            "largest_loss_increase_ratio": round(
                epoch_stability["largest_loss_increase_ratio"],
                4,
            ),
            "peak_grad_norm": round(epoch_stability["peak_grad_norm"], 4),
        }

        if "f1" in val_metrics:
            row["val_f1"] = val_metrics["f1"]

        history.append(row)
        print(row)

        score = selection_metric(cfg["task"], val_metrics)
        if score > best_score:
            best_score = score
            model.save_pretrained(os.path.join(cfg["output_dir"], "best_adapter"))
            tokenizer.save_pretrained(os.path.join(cfg["output_dir"], "best_adapter"))

        reset_accelerator_peak_memory(device)

    with open(os.path.join(cfg["output_dir"], "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    summary = {
        "task": cfg["task"],
        "optimizer": cfg["optimizer"],
        "device": str(device),
        "epochs_completed": len(history),
        "run_time_sec": round(time.time() - run_start_time, 2),
        "process_peak_rss_mb": round(get_process_rss_mb(), 2),
        "accelerator_peak_memory_mb": (
            round(run_accelerator_peak_memory_mb, 2)
            if device.type == "cuda"
            else None
        ),
        "loss_spike_steps": run_stability["loss_spike_steps"],
        "unstable_steps": run_stability["unstable_steps"],
        "non_finite_steps": run_stability["non_finite_steps"],
        "largest_loss_increase_ratio": round(
            run_stability["largest_loss_increase_ratio"],
            4,
        ),
        "peak_grad_norm": round(run_stability["peak_grad_norm"], 4),
        "diverged": run_stability["diverged"],
        "divergence_reason": run_stability["divergence_reason"],
    }

    with open(os.path.join(cfg["output_dir"], "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return history
