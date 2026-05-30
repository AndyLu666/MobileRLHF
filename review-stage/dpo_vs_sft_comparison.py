import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel

from pytorch_alignment.dpo_utils import (
    PreferenceDataset,
    compute_dpo_batch_metrics,
    dpo_collate_batch,
    sequence_logprob,
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


class ChosenOnlyDataset:
    def __init__(self, path: Path, tokenizer, seq_len: int):
        self.samples = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                prompt_ids = tokenizer.encode(rec["prompt"], add_special_tokens=False)
                chosen_ids = tokenizer.encode(rec["chosen"], add_special_tokens=False)
                ids = (prompt_ids + chosen_ids)[:seq_len]
                response_mask = ([0] * len(prompt_ids) + [1] * len(chosen_ids))[:seq_len]
                pad_len = seq_len - len(ids)
                if pad_len > 0:
                    ids += [tokenizer.pad_token_id] * pad_len
                    response_mask += [0] * pad_len
                input_ids = torch.tensor(ids, dtype=torch.long)
                labels = torch.full_like(input_ids, -100)
                mask_tensor = torch.tensor(response_mask, dtype=torch.long)
                labels = torch.where(mask_tensor > 0, input_ids, labels)
                attention_mask = torch.tensor(
                    [1 if t != tokenizer.pad_token_id else 0 for t in ids],
                    dtype=torch.long,
                )
                self.samples.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                    }
                )

    def collate(self):
        return {
            key: torch.stack([sample[key] for sample in self.samples], dim=0)
            for key in self.samples[0].keys()
        }


def evaluate_preference_policy(model, batch):
    chosen_out = model(
        input_ids=batch["chosen_input_ids"],
        attention_mask=batch["chosen_attention_mask"],
    )
    rejected_out = model(
        input_ids=batch["rejected_input_ids"],
        attention_mask=batch["rejected_attention_mask"],
    )
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
    pair_margin = chosen_logps - rejected_logps
    chosen_lengths = batch["chosen_response_mask"][:, 1:].sum(dim=-1).clamp_min(1).to(torch.float32)
    rejected_lengths = batch["rejected_response_mask"][:, 1:].sum(dim=-1).clamp_min(1).to(torch.float32)
    return {
        "pair_accuracy": float((pair_margin > 0).to(torch.float32).mean().item()),
        "mean_pair_margin": float(pair_margin.mean().item()),
        "chosen_logprob_per_token": float((chosen_logps / chosen_lengths).mean().item()),
        "rejected_logprob_per_token": float((rejected_logps / rejected_lengths).mean().item()),
    }


def train_dpo(model, ref_model, batch, steps: int, lr: float, beta: float):
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=lr,
    )
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        metrics = compute_dpo_batch_metrics(model, ref_model, batch, beta)
        metrics["loss"].backward()
        optimizer.step()


def train_sft(model, batch, steps: int, lr: float):
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=lr,
    )
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(**batch)
        out.loss.backward()
        optimizer.step()


def main():
    repo_root = Path(__file__).resolve().parents[1]
    device = torch.device("cpu")
    tokenizer = SimpleCharTokenizer()
    seq_len = 96
    steps = 60
    lr = 5e-3
    beta = 0.1
    data_path = repo_root / "review-stage" / "synthetic_dpo_high_quality.jsonl"

    preference_dataset = PreferenceDataset(str(data_path), tokenizer, seq_len=seq_len)
    preference_batch = dpo_collate_batch([preference_dataset[i] for i in range(len(preference_dataset))])
    preference_batch = {k: v.to(device) for k, v in preference_batch.items()}

    chosen_only_dataset = ChosenOnlyDataset(data_path, tokenizer, seq_len=seq_len)
    sft_batch = {k: v.to(device) for k, v in chosen_only_dataset.collate().items()}

    dpo_model, ref_model = build_policy_and_reference(tokenizer.vocab_size, device)
    sft_model, _ = build_policy_and_reference(tokenizer.vocab_size, device)

    before_metrics = evaluate_preference_policy(dpo_model, preference_batch)
    train_dpo(dpo_model, ref_model, preference_batch, steps=steps, lr=lr, beta=beta)
    train_sft(sft_model, sft_batch, steps=steps, lr=lr)
    dpo_after = evaluate_preference_policy(dpo_model, preference_batch)
    sft_after = evaluate_preference_policy(sft_model, preference_batch)

    report = {
        "setup": {
            "algorithm_a": "DPO on preference pairs",
            "algorithm_b": "SFT on chosen responses only",
            "steps": steps,
            "learning_rate": lr,
            "beta": beta,
            "dataset": str(data_path.relative_to(repo_root)),
        },
        "before_training": before_metrics,
        "after_dpo": dpo_after,
        "after_sft": sft_after,
        "comparison": {
            "dpo_minus_sft_pair_accuracy": dpo_after["pair_accuracy"] - sft_after["pair_accuracy"],
            "dpo_minus_sft_pair_margin": dpo_after["mean_pair_margin"] - sft_after["mean_pair_margin"],
            "dpo_minus_sft_chosen_logprob_per_token": dpo_after["chosen_logprob_per_token"] - sft_after["chosen_logprob_per_token"],
            "dpo_minus_sft_rejected_logprob_per_token": dpo_after["rejected_logprob_per_token"] - sft_after["rejected_logprob_per_token"],
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
