from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(task, preds, labels):
    metrics = {
        "accuracy": accuracy_score(labels, preds)
    }

    # MRPC is the one where F1 is especially useful
    if task == "mrpc":
        metrics["f1"] = f1_score(labels, preds)

    return metrics


def selection_metric(task, metrics):
    if task == "mrpc":
        return metrics["f1"]
    return metrics["accuracy"]