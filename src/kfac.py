import torch

from src.asdl_compat import patch_nvtx_if_needed


def build_kfac_gradient_maker(model, cfg, total_steps: int):
    """
    Factory for ASDL K-FAC gradient maker applied only to modules that are not ignored
    (see kfac_ignore_substrings). Intended for LoRA adapter Linears; classifier heads
    should be listed in ignore patterns and trained with AdamW.
    """
    patch_nvtx_if_needed()

    from asdl import KfacGradientMaker, PreconditioningConfig
    from asdl.matrices import FISHER_MC

    ignore = cfg.get(
        "kfac_ignore_substrings",
        ["pre_classifier", "classifier"],
    )
    batch_size = cfg["batch_size"]
    damping = float(cfg.get("kfac_damping", 1e-3))
    ema = float(cfg.get("kfac_ema_decay", 0.95))
    name = cfg["optimizer"].lower()

    prec = PreconditioningConfig(
        num_total_steps=total_steps,
        damping=damping,
        ema_decay=ema,
        data_size=batch_size,
        ignore_modules=ignore,
        preconditioner_upd_ratio=float(cfg.get("kfac_preconditioner_upd_ratio", 1.0)),
        curvature_upd_ratio=float(cfg.get("kfac_curvature_upd_ratio", 1.0)),
    )

    if name != "adapter_kfac":
        raise ValueError(f"Expected optimizer adapter_kfac, got {name}")

    fisher_type = cfg.get("kfac_fisher_type", FISHER_MC)
    swift = bool(cfg.get("kfac_swift", False))
    n_mc = int(cfg.get("kfac_n_mc_samples", 1))
    return KfacGradientMaker(
        model,
        prec,
        fisher_type=fisher_type,
        loss_type="cross_entropy",
        n_mc_samples=n_mc,
        swift=swift,
    )


def kfac_forward_backward(grad_maker, model, batch):
    dummy = grad_maker.setup_model_call(model, **batch)
    grad_maker.setup_logits_repr(dummy.logits)
    grad_maker.setup_loss_repr(dummy.loss)
    return grad_maker.forward_and_backward()


def make_adapter_sgd_optimizer(lora_params, cfg):
    return torch.optim.SGD(
        lora_params,
        lr=cfg["lr"],
        momentum=float(cfg.get("adapter_momentum", 0.0)),
        weight_decay=cfg["weight_decay"],
    )
