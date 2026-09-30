import json
import math

import torch


def create_optimizer(model, learning_rate, weight_decay):
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )


def create_scheduler(
    optimizer,
    max_steps,
    warmup_steps,
    learning_rate,
):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_steps - warmup_steps,
        eta_min=learning_rate * 0.1,
    )

    return scheduler


def set_warmup_lr(optimizer, learning_rate, step, warmup_steps):
    lr = learning_rate * step / warmup_steps

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]["lr"]


def save_checkpoint(
    path,
    step,
    tokens_seen,
    model,
    optimizer,
    scheduler,
    scaler,
    config,
):
    checkpoint = {
        "step": step,
        "tokens_seen": tokens_seen,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config,
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
):
    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    scaler.load_state_dict(
        checkpoint["scaler_state_dict"]
    )

    return checkpoint


def append_log(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )


def calculate_perplexity(loss):
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")
