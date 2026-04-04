from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model


def build_model(cfg):
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=2,
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_targets"],
        bias="none",
        modules_to_save=["pre_classifier", "classifier"],
    )

    model = get_peft_model(model, peft_config)
    return model