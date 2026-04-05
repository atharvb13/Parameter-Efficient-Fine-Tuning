import torch


class AdapterDiagOptimizer(torch.optim.Optimizer):
    """
    Diagonal preconditioner with bias correction.
    Use this only for LoRA adapter parameters.
    """

    def __init__(self, params, lr=1e-4, beta2=0.999, eps=1e-6, weight_decay=0.0):
        defaults = dict(lr=lr, beta2=beta2, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if not torch.isfinite(grad).all():
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction2 = 1 - (beta2 ** state["step"])
                denom = (exp_avg_sq / bias_correction2).sqrt().add_(eps)

                # decoupled weight decay
                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)

                p.addcdiv_(grad, denom, value=-lr)

        return loss


def get_optimizers(model, cfg):
    name = cfg["optimizer"].lower()

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    if name == "adamw":
        params = [p for _, p in trainable]
        return {
            "main": torch.optim.AdamW(
                params,
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"],
                eps=cfg["eps"],
            )
        }

    if name == "sgdm":
        params = [p for _, p in trainable]
        return {
            "main": torch.optim.SGD(
                params,
                lr=cfg["lr"],
                momentum=cfg["momentum"],
                weight_decay=cfg["weight_decay"],
            )
        }

    if name == "adapter_diag":
        lora_params = []
        head_params = []

        for n, p in trainable:
            if "lora_" in n:
                lora_params.append(p)
            else:
                head_params.append(p)

        return {
            "adapter": AdapterDiagOptimizer(
                lora_params,
                lr=cfg["lr"],
                beta2=cfg["beta2"],
                eps=cfg["eps"],
                weight_decay=cfg["weight_decay"],
            ),
            "head": torch.optim.AdamW(
                head_params,
                lr=cfg.get("head_lr", 2e-4),
                weight_decay=cfg.get("head_weight_decay", 0.01),
                eps=cfg["eps"],
            ),
        }

    raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")