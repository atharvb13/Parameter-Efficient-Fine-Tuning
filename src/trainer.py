import json
import os
import random
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

    print("\nTrainable parameters:")
    model.print_trainable_parameters()

    for epoch in range(cfg["epochs"]):
        model.train()
        start_time = time.time()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")

        for step, batch in enumerate(pbar, start=1):
            batch = move_batch_to_device(batch, device)

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                cfg["grad_clip"],
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            pbar.set_postfix(train_loss=f"{running_loss / step:.4f}")

        val_metrics = evaluate(model, val_loader, device, cfg["task"])
        train_loss = running_loss / len(train_loader)
        epoch_time = time.time() - start_time

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "epoch_time_sec": round(epoch_time, 2),
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

    with open(os.path.join(cfg["output_dir"], "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return history