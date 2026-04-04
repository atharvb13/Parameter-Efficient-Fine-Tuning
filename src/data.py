from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

TASK_TO_KEYS = {
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "rte": ("sentence1", "sentence2"),
}


def build_dataloaders(cfg):
    task = cfg["task"]
    sentence1_key, sentence2_key = TASK_TO_KEYS[task]

    raw = load_dataset("glue", task)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)

    def preprocess(examples):
        if sentence2_key is None:
            return tokenizer(
                examples[sentence1_key],
                truncation=True,
                max_length=cfg["max_length"],
            )
        return tokenizer(
            examples[sentence1_key],
            examples[sentence2_key],
            truncation=True,
            max_length=cfg["max_length"],
        )

    # Keep label, remove raw text/index columns
    remove_cols = [c for c in raw["train"].column_names if c != "label"]

    tokenized = raw.map(
        preprocess,
        batched=True,
        remove_columns=remove_cols,
    )

    tokenized = tokenized.rename_column("label", "labels")

    # Only expose tensor fields to the dataloader/collator
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt",
    )

    train_loader = DataLoader(
        tokenized["train"],
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        tokenized["validation"],
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=collator,
    )

    return train_loader, val_loader, tokenizer