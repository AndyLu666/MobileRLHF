import argparse
import json

import torch
from peft import PeftModel
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from pytorch_alignment.dpo_utils import PreferenceDataset, dpo_collate_batch, sequence_logprob


def build_model(base_dir: str, adapter_dir: str | None, device: torch.device):
    model = GPT2LMHeadModel.from_pretrained(base_dir)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on preference pairs")
    parser.add_argument("--base_model_dir", required=True)
    parser.add_argument("--adapter_dir", default="")
    parser.add_argument("--preference_jsonl", required=True)
    parser.add_argument("--seq_len", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2TokenizerFast.from_pretrained(args.base_model_dir)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    dataset = PreferenceDataset(args.preference_jsonl, tokenizer, args.seq_len)
    batch = dpo_collate_batch([dataset[i] for i in range(len(dataset))])
    batch = {k: v.to(device) for k, v in batch.items()}
    model = build_model(args.base_model_dir, args.adapter_dir or None, device)

    with torch.no_grad():
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

    margin = chosen_logps - rejected_logps
    chosen_lengths = batch["chosen_response_mask"][:, 1:].sum(dim=-1).clamp_min(1).to(torch.float32)
    rejected_lengths = batch["rejected_response_mask"][:, 1:].sum(dim=-1).clamp_min(1).to(torch.float32)
    report = {
        "num_samples": len(dataset),
        "pair_accuracy": float((margin > 0).to(torch.float32).mean().item()),
        "mean_pair_margin": float(margin.mean().item()),
        "chosen_logprob_per_token": float((chosen_logps / chosen_lengths).mean().item()),
        "rejected_logprob_per_token": float((rejected_logps / rejected_lengths).mean().item()),
        "adapter_dir": args.adapter_dir or "",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
