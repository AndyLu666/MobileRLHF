from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_alignment.dpo_utils import (
    binary_feedback_collate_batch,
    compute_kto_batch_metrics,
    evaluate_kto,
)


@dataclass(frozen=True)
class KtoTrainingConfig:
    max_steps: int = 10
    batch_size: int = 4
    beta: float = 0.1
    lr: float = 1e-3
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0
    eval_max_batches: int = 10
    seed: int = 42
    metrics_path: Path | None = None


def summarize_gradients(model) -> tuple[dict[str, float], dict[str, float], list[str]]:
    trainable = {}
    frozen_with_grad = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad_norm = 0.0 if param.grad is None else float(param.grad.norm().item())
            trainable[name] = grad_norm
        elif param.grad is not None:
            frozen_with_grad.append(name)
    nonzero_trainable = {key: value for key, value in trainable.items() if value > 0.0}
    return trainable, nonzero_trainable, frozen_with_grad


def tensor_metrics_to_float(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu().item()) for key, value in metrics.items()}


def _prepare_reference_model(reference_model) -> None:
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False


def _status_from_last_metrics(last_metrics: dict[str, Any] | None) -> str:
    if not last_metrics:
        return "failed"
    if last_metrics["policy_nonzero_grad_params"] <= 0:
        return "failed"
    if last_metrics["reference_grad_params"] != 0:
        return "failed"
    if not torch.isfinite(torch.tensor(last_metrics["loss"])):
        return "failed"
    return "ok"


def train_kto(
    policy_model,
    reference_model,
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
    config: KtoTrainingConfig,
    device: torch.device,
) -> dict[str, Any]:
    if len(train_dataset) == 0:
        raise ValueError("KTO training dataset is empty")
    if len(train_dataset) < 2:
        raise ValueError("canonical KTO requires at least two training rows")
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if config.batch_size <= 1:
        raise ValueError("canonical KTO batch_size must be greater than one")

    torch.manual_seed(config.seed)
    policy_model.to(device)
    reference_model.to(device)
    _prepare_reference_model(reference_model)

    train_batch_size = min(config.batch_size, len(train_dataset))
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=binary_feedback_collate_batch,
        drop_last=(len(train_dataset) % train_batch_size == 1),
    )
    eval_source = eval_dataset if eval_dataset is not None and len(eval_dataset) > 0 else train_dataset
    if len(eval_source) < 2:
        raise ValueError("canonical KTO evaluation requires at least two rows")
    eval_batch_size = min(config.batch_size, len(eval_source))
    eval_loader = DataLoader(
        eval_source,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=binary_feedback_collate_batch,
        drop_last=(len(eval_source) % eval_batch_size == 1),
    )

    optimizer = torch.optim.AdamW(
        [param for param in policy_model.parameters() if param.requires_grad],
        lr=config.lr,
    )

    metrics_file = None
    if config.metrics_path is not None:
        config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_file = config.metrics_path.open("w", encoding="utf-8")

    last_metrics = None
    try:
        for step, batch in zip(range(1, config.max_steps + 1), cycle(train_loader)):
            policy_model.train()
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            metrics = compute_kto_batch_metrics(
                policy_model,
                reference_model,
                batch,
                beta=config.beta,
                desirable_weight=config.desirable_weight,
                undesirable_weight=config.undesirable_weight,
            )
            metrics["loss"].backward()
            trainable, nonzero_trainable, frozen_with_grad = summarize_gradients(policy_model)
            ref_grad_params = [
                name for name, param in reference_model.named_parameters() if param.grad is not None
            ]
            optimizer.step()

            row = {
                "step": step,
                "source": "pytorch_kto",
                "batch_size": config.batch_size,
                "train_samples": len(train_dataset),
                "policy_trainable_params": len(trainable),
                "policy_nonzero_grad_params": len(nonzero_trainable),
                "policy_frozen_params_with_grad": len(frozen_with_grad),
                "reference_grad_params": len(ref_grad_params),
                **tensor_metrics_to_float(metrics),
            }
            if metrics_file is not None:
                metrics_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            last_metrics = row
    finally:
        if metrics_file is not None:
            metrics_file.close()

    eval_metrics = evaluate_kto(
        policy_model,
        reference_model,
        eval_loader,
        device,
        max_batches=config.eval_max_batches,
        beta=config.beta,
        desirable_weight=config.desirable_weight,
        undesirable_weight=config.undesirable_weight,
    )

    return {
        "status": _status_from_last_metrics(last_metrics),
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_source),
        "max_steps": config.max_steps,
        "batch_size": config.batch_size,
        "beta": config.beta,
        "desirable_weight": config.desirable_weight,
        "undesirable_weight": config.undesirable_weight,
        "last_train_metrics": last_metrics,
        "eval_metrics": eval_metrics,
        "metrics_path": str(config.metrics_path) if config.metrics_path is not None else None,
    }
