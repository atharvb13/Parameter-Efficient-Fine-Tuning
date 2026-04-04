import torch


class AdapterDiagOptimizer(torch.optim.Optimizer):
    """
    Simple diagonal preconditioner using moving averages of squared gradients.
    This is the minimum working 'adapter-only diagonal preconditioning' baseline.
    """

    def __init__(self, params, lr=2e-4, beta2=0.999, eps=1e-8, weight_decay=0.0):
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
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg_sq = state["exp_avg_sq"]
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(eps)

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                p.addcdiv_(grad, denom, value=-lr)

        return loss


def get_optimizer(model, cfg):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    name = cfg["optimizer"].lower()

    if name == "adamw":
        return torch.optim.AdamW(
            trainable_params,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            eps=cfg["eps"],
        )

    if name == "sgdm":
        return torch.optim.SGD(
            trainable_params,
            lr=cfg["lr"],
            momentum=cfg["momentum"],
            weight_decay=cfg["weight_decay"],
        )

    if name == "adapter_diag":
        return AdapterDiagOptimizer(
            trainable_params,
            lr=cfg["lr"],
            beta2=cfg["beta2"],
            eps=cfg["eps"],
            weight_decay=cfg["weight_decay"],
        )

    raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")