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


def build_policy_and_reference(vocab_size: int, device: torch.device):
    torch.manual_seed(42)
    policy_model = build_tiny_policy(vocab_size).to(device)
    reference_model = build_tiny_policy(vocab_size).to(device)
    reference_model.load_state_dict(policy_model.state_dict())
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    return policy_model, reference_model


def summarize_dataset(path: Path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    chosen_lens = [len(rec["chosen"]) for rec in records]
    rejected_lens = [len(rec["rejected"]) for rec in records]
    return {
        "num_samples": len(records),
        "avg_chosen_chars": sum(chosen_lens) / max(1, len(chosen_lens)),
        "avg_rejected_chars": sum(rejected_lens) / max(1, len(rejected_lens)),
        "avg_abs_length_gap": sum(abs(c - r) for c, r in zip(chosen_lens, rejected_lens)) / max(1, len(records)),
    }


def evaluate_dataset(path: Path, tokenizer, steps: int, lr: float, beta: float):
    device = torch.device("cpu")
    dataset = PreferenceDataset(str(path), tokenizer, seq_len=96)
    batch = dpo_collate_batch([dataset[i] for i in range(len(dataset))])
    batch = {k: v.to(device) for k, v in batch.items()}

    policy_model, reference_model = build_policy_and_reference(tokenizer.vocab_size, device)
    optimizer = torch.optim.AdamW(
        [param for param in policy_model.parameters() if param.requires_grad],
        lr=lr,
    )

    initial = compute_dpo_batch_metrics(policy_model, reference_model, batch, beta)
    initial_metrics = {k: float(v.item()) for k, v in initial.items()}

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        metrics = compute_dpo_batch_metrics(policy_model, reference_model, batch, beta)
        metrics["loss"].backward()
        optimizer.step()

    final = compute_dpo_batch_metrics(policy_model, reference_model, batch, beta)
    final_metrics = {k: float(v.item()) for k, v in final.items()}

    return {
        "dataset_summary": summarize_dataset(path),
        "initial": initial_metrics,
        "final": final_metrics,
        "delta": {
            "loss": final_metrics["loss"] - initial_metrics["loss"],
            "reward_margin": final_metrics["reward_margin"] - initial_metrics["reward_margin"],
            "reward_accuracy": final_metrics["reward_accuracy"] - initial_metrics["reward_accuracy"],
        },
    }


def main():
    repo_root = Path(__file__).resolve().parents[1]
    tokenizer = SimpleCharTokenizer()
    beta = 0.1
    steps = 60
    lr = 5e-3

    weak_path = repo_root / "review-stage" / "data" / "synthetic_dpo_smoke.jsonl"
    strong_path = repo_root / "review-stage" / "data" / "synthetic_dpo_high_quality.jsonl"

    report = {
        "algorithm": {
            "name": "reference-based Direct Preference Optimization (DPO)",
            "objective": "maximize chosen-vs-rejected policy log-ratio relative to a frozen reference model",
            "loss_form": "L = -log(sigmoid(beta * ((logpi_c-logpi_r) - (logref_c-logref_r))))",
            "response_scoring": "sum of token log-probabilities over the response span only",
            "trainable_params": "LoRA policy adapter only; reference model frozen",
        },
        "experiment": {
            "optimizer": "AdamW",
            "steps": steps,
            "learning_rate": lr,
            "beta": beta,
            "device": "cpu",
            "tiny_model": "GPT2Config(n_layer=2, n_head=2, n_embd=64) + LoRA(r=4, alpha=8)",
        },
        "datasets": {
            "existing_smoke": evaluate_dataset(weak_path, tokenizer, steps, lr, beta),
            "high_quality": evaluate_dataset(strong_path, tokenizer, steps, lr, beta),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
