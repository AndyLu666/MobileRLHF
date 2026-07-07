import json
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset


class PreferenceDataset(Dataset):
    def __init__(self, path: str, tokenizer, seq_len: int):
        self.samples = []
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    prompt = rec["prompt"]
                    chosen = rec["chosen"]
                    rejected = rec["rejected"]
                    chosen_pair = self._build_pair(prompt, chosen)
                    rejected_pair = self._build_pair(prompt, rejected)
                    if chosen_pair["response_mask"].sum().item() == 0:
                        continue
                    if rejected_pair["response_mask"].sum().item() == 0:
                        continue
                    self.samples.append({
                        "chosen": chosen_pair,
                        "rejected": rejected_pair,
                    })
                except Exception:
                    continue

    def _build_pair(self, prompt: str, response: str):
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        ids = (prompt_ids + response_ids)[: self.seq_len]
        response_mask = ([0] * len(prompt_ids) + [1] * len(response_ids))[: self.seq_len]

        pad_len = self.seq_len - len(ids)
        if pad_len > 0:
            ids += [self.tokenizer.pad_token_id] * pad_len
            response_mask += [0] * pad_len

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                [1 if t != self.tokenizer.pad_token_id else 0 for t in ids],
                dtype=torch.long,
            ),
            "response_mask": torch.tensor(response_mask, dtype=torch.long),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class BinaryFeedbackDataset(Dataset):
    def __init__(self, path: str, tokenizer, seq_len: int):
        self.samples = []
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    prompt = rec["prompt"]
                    response = rec["response"]
                    label = rec.get("label", rec.get("desirable"))
                    pair = self._build_pair(prompt, response)
                    if pair["response_mask"].sum().item() == 0:
                        continue
                    self.samples.append({
                        **pair,
                        "label": torch.tensor(1.0 if is_desirable_label(label) else -1.0, dtype=torch.float32),
                    })
                except Exception:
                    continue

    def _build_pair(self, prompt: str, response: str):
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        ids = (prompt_ids + response_ids)[: self.seq_len]
        response_mask = ([0] * len(prompt_ids) + [1] * len(response_ids))[: self.seq_len]

        pad_len = self.seq_len - len(ids)
        if pad_len > 0:
            ids += [self.tokenizer.pad_token_id] * pad_len
            response_mask += [0] * pad_len

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                [1 if t != self.tokenizer.pad_token_id else 0 for t in ids],
                dtype=torch.long,
            ),
            "response_mask": torch.tensor(response_mask, dtype=torch.long),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def is_desirable_label(label) -> bool:
    if isinstance(label, str):
        return label.lower() in {
            "1",
            "true",
            "yes",
            "desirable",
            "positive",
            "up",
            "like",
            "chosen",
            "correct",
            "accepted",
            "good",
        }
    return bool(label)


def dpo_collate_batch(batch: List[dict]) -> Dict[str, torch.Tensor]:
    out = {}
    for branch in ["chosen", "rejected"]:
        for key in ["input_ids", "attention_mask", "response_mask"]:
            out[f"{branch}_{key}"] = torch.stack([x[branch][key] for x in batch], dim=0)
    return out


def binary_feedback_collate_batch(batch: List[dict]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch], dim=0),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch], dim=0),
        "response_mask": torch.stack([x["response_mask"] for x in batch], dim=0),
        "label": torch.stack([x["label"] for x in batch], dim=0),
    }


def sequence_logprob(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    shifted_mask = mask[:, 1:].to(dtype=logits.dtype)
    log_probs = torch.nn.functional.log_softmax(shifted_logits.float(), dim=-1)
    token_logps = log_probs.gather(-1, shifted_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * shifted_mask.float()
    return token_logps.sum(dim=-1)


def compute_kto_batch_metrics(
    model,
    ref_model,
    batch: Dict[str, torch.Tensor],
    beta: float,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
):
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    policy_logps = sequence_logprob(out.logits, batch["input_ids"], batch["response_mask"])
    with torch.no_grad():
        ref_out = ref_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        ref_logps = sequence_logprob(ref_out.logits, batch["input_ids"], batch["response_mask"])
    logratios = policy_logps - ref_logps
    kl_center = torch.clamp(logratios.mean().detach(), min=0.0)
    rewards = beta * (logratios - kl_center)
    labels = batch["label"].to(dtype=rewards.dtype)
    weights = torch.where(
        labels > 0,
        rewards.new_tensor(desirable_weight),
        rewards.new_tensor(undesirable_weight),
    )
    desirable_losses = 1.0 - torch.sigmoid(rewards)
    undesirable_losses = 1.0 - torch.sigmoid(-rewards)
    losses = torch.where(labels > 0, desirable_losses, undesirable_losses) * weights
    desirable_mask = labels > 0
    undesirable_mask = labels < 0
    desirable_reward = rewards[desirable_mask].mean() if desirable_mask.any() else rewards.new_tensor(0.0)
    undesirable_reward = rewards[undesirable_mask].mean() if undesirable_mask.any() else rewards.new_tensor(0.0)
    return {
        "loss": losses.mean(),
        "reward_accuracy": ((labels * rewards) > 0).float().mean(),
        "reward_margin": desirable_reward - undesirable_reward,
        "desirable_reward": desirable_reward,
        "undesirable_reward": undesirable_reward,
        "kl_center": kl_center,
    }


def evaluate_kto(
    model,
    ref_model,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int,
    beta: float,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
):
    model.eval()
    totals = {
        "loss": 0.0,
        "reward_accuracy": 0.0,
        "reward_margin": 0.0,
        "desirable_reward": 0.0,
        "undesirable_reward": 0.0,
        "kl_center": 0.0,
    }
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            metrics = compute_kto_batch_metrics(
                model,
                ref_model,
                batch,
                beta,
                desirable_weight=desirable_weight,
                undesirable_weight=undesirable_weight,
            )
            for key in totals:
                totals[key] += float(metrics[key].item())
            count += 1
    model.train()
    if count == 0:
        return {key: float("inf") for key in totals}
    return {key: value / float(count) for key, value in totals.items()}


def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
    loss_type: str = "sigmoid",
):
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    if loss_type == "sigmoid":
        losses = -torch.nn.functional.logsigmoid(logits)
    elif loss_type == "robust":
        losses = -torch.nn.functional.logsigmoid(logits) + 0.1 * torch.nn.functional.softplus(-logits)
    elif loss_type == "ipo":
        losses = 0.5 * (logits - 1.0).pow(2)
    elif loss_type == "sigmoid_norm":
        losses = -torch.nn.functional.logsigmoid(logits / 2.0)
    else:
        raise ValueError(f"unsupported DPO loss_type: {loss_type}")
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    reward_margin = chosen_rewards - rejected_rewards
    reward_accuracy = (reward_margin > 0).float().mean()
    return {
        "loss": losses.mean(),
        "chosen_reward": chosen_rewards.mean(),
        "rejected_reward": rejected_rewards.mean(),
        "reward_margin": reward_margin.mean(),
        "reward_accuracy": reward_accuracy,
    }


def compute_dpo_batch_metrics(
    model,
    ref_model,
    batch: Dict[str, torch.Tensor],
    beta: float,
    loss_type: str = "sigmoid",
):
    chosen_inputs = {
        "input_ids": batch["chosen_input_ids"],
        "attention_mask": batch["chosen_attention_mask"],
    }
    rejected_inputs = {
        "input_ids": batch["rejected_input_ids"],
        "attention_mask": batch["rejected_attention_mask"],
    }

    chosen_out = model(**chosen_inputs)
    rejected_out = model(**rejected_inputs)

    chosen_logps = sequence_logprob(
        chosen_out.logits,
        batch["chosen_input_ids"],
        batch["chosen_response_mask"],
    )
    rejected_logps = sequence_logprob(
        rejected_out.logits,
        batch["rejected_input_ids"],
        batch["rejected_response_mask"],
    )

    with torch.no_grad():
        ref_chosen_out = ref_model(**chosen_inputs)
        ref_rejected_out = ref_model(**rejected_inputs)
        ref_chosen_logps = sequence_logprob(
            ref_chosen_out.logits,
            batch["chosen_input_ids"],
            batch["chosen_response_mask"],
        )
        ref_rejected_logps = sequence_logprob(
            ref_rejected_out.logits,
            batch["rejected_input_ids"],
            batch["rejected_response_mask"],
        )

    return compute_dpo_loss(
        chosen_logps,
        rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta,
        loss_type=loss_type,
    )


def evaluate_dpo(
    model,
    ref_model,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int,
    beta: float,
    loss_type: str = "sigmoid",
):
    model.eval()
    totals = {
        "loss": 0.0,
        "chosen_reward": 0.0,
        "rejected_reward": 0.0,
        "reward_margin": 0.0,
        "reward_accuracy": 0.0,
    }
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            metrics = compute_dpo_batch_metrics(model, ref_model, batch, beta, loss_type=loss_type)
            for key in totals:
                totals[key] += float(metrics[key].item())
            count += 1
    model.train()
    if count == 0:
        return {key: float("inf") for key in totals}
    return {key: value / float(count) for key, value in totals.items()}
