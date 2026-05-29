import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel

from pytorch_alignment.dpo_utils import (
    PreferenceDataset,
    compute_dpo_batch_metrics,
    dpo_collate_batch,
)


class SimpleCharTokenizer:
    def __init__(self):
        alphabet = (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            " .,!?;:-_+/\\'\"()[]{}\n"
        )
        self.pad_token_id = 0
        self.eos_token_id = 1
        self._stoi = {ch: idx + 2 for idx, ch in enumerate(alphabet)}
        self.vocab_size = len(self._stoi) + 2

    def encode(self, text, add_special_tokens=False):
        ids = [self._stoi.get(ch, self.eos_token_id) for ch in text]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids


def build_tiny_policy(vocab_size: int):
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=128,
        n_ctx=128,
        n_embd=64,
        n_layer=2,
        n_head=2,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_cfg)


def summarize_gradients(model):
    trainable = {}
    frozen_with_grad = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad_norm = 0.0 if param.grad is None else float(param.grad.norm().item())
            trainable[name] = grad_norm
        elif param.grad is not None:
            frozen_with_grad.append(name)
    nonzero_trainable = {k: v for k, v in trainable.items() if v > 0.0}
    return trainable, nonzero_trainable, frozen_with_grad


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "review-stage" / "synthetic_dpo_smoke.jsonl"
    tokenizer = SimpleCharTokenizer()
    dataset = PreferenceDataset(str(data_path), tokenizer, seq_len=96)
    batch = dpo_collate_batch([dataset[i] for i in range(len(dataset))])

    device = torch.device("cpu")
    torch.manual_seed(42)

    policy_model = build_tiny_policy(tokenizer.vocab_size).to(device)
    reference_model = build_tiny_policy(tokenizer.vocab_size).to(device)
    reference_model.load_state_dict(policy_model.state_dict())
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    batch = {k: v.to(device) for k, v in batch.items()}

    policy_model.train()
    metrics = compute_dpo_batch_metrics(policy_model, reference_model, batch, beta=0.1)
    metrics["loss"].backward()

    trainable, nonzero_trainable, frozen_with_grad = summarize_gradients(policy_model)
    ref_grad_params = [
        name for name, param in reference_model.named_parameters() if param.grad is not None
    ]

    report = {
        "num_samples": len(dataset),
        "loss": float(metrics["loss"].item()),
        "chosen_reward": float(metrics["chosen_reward"].item()),
        "rejected_reward": float(metrics["rejected_reward"].item()),
        "reward_margin": float(metrics["reward_margin"].item()),
        "reward_accuracy": float(metrics["reward_accuracy"].item()),
        "policy_trainable_params": len(trainable),
        "policy_nonzero_grad_params": len(nonzero_trainable),
        "policy_nonzero_grad_examples": list(nonzero_trainable.items())[:8],
        "policy_frozen_params_with_grad": frozen_with_grad[:8],
        "reference_grad_params": ref_grad_params[:8],
        "status": "ok"
        if torch.isfinite(metrics["loss"]) and nonzero_trainable and not ref_grad_params
        else "failed",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
