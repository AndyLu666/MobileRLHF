import argparse
import json
import math
import os
import random
from typing import Iterable, List

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import LoraConfig, PeftModel, get_peft_model
except ImportError as e:  # pragma: no cover - import-time guard
    raise SystemExit("Please install peft to run this script: pip install peft") from e

from pytorch_alignment.dpo_utils import (
    BinaryFeedbackDataset,
    PreferenceDataset,
    binary_feedback_collate_batch,
    compute_dpo_batch_metrics,
    compute_kto_batch_metrics,
    dpo_collate_batch,
    evaluate_dpo,
    evaluate_kto,
)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WikiTextDataset(Dataset):
    """
    Mirrors the C++ WikiText2Dataset: concat lines with EOS, chunk into fixed windows.
    Labels equal input_ids; loss does the shift internally.
    """

    def __init__(
        self,
        path: str,
        tokenizer: AutoTokenizer,
        seq_len: int,
        eos_token_id: int,
        data_fraction: float = 1.0,
        insert_eos_between_lines: bool = True,
        drop_last: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        tokens = self._load(path, tokenizer, eos_token_id, insert_eos_between_lines)
        if data_fraction < 1.0:
            keep = max(seq_len + 1, int(len(tokens) * data_fraction))
            tokens = tokens[:keep]
        self.chunks = self._chunk(tokens, drop_last, eos_token_id)

    def _load(
        self,
        path: str,
        tokenizer: AutoTokenizer,
        eos_token_id: int,
        insert_eos_between_lines: bool,
    ) -> List[int]:
        tokens: List[int] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line == "":
                    if insert_eos_between_lines:
                        tokens.append(eos_token_id)
                    continue
                ids = tokenizer.encode(line, add_special_tokens=False)
                tokens.extend(ids)
                if insert_eos_between_lines:
                    tokens.append(eos_token_id)
        return tokens

    def _chunk(self, tokens: List[int], drop_last: bool, pad_id: int) -> List[torch.Tensor]:
        # Align with C++ version: need seq_len+1 tokens available to form a valid chunk
        # C++ uses: for s in range(0, N - (S+1) + 1, stride) where need = S+1
        # HuggingFace does the shift internally (logits[:-1] vs labels[1:])
        chunks: List[torch.Tensor] = []
        n = len(tokens)
        need = self.seq_len + 1  # Align with C++: need seq_len+1 tokens available
        for start in range(0, n - need + 1, self.seq_len):
            window = tokens[start : start + self.seq_len]
            chunks.append(torch.tensor(window, dtype=torch.long))
        return chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int):
        ids = self.chunks[idx]
        attn = torch.ones_like(ids, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": attn, "labels": ids.clone()}


def collate_batch(batch: List[dict]) -> dict:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


class JsonlMaskedDataset(Dataset):
    """
    JSONL dataset {"ids": [...], "mask": [...]} with labels masked in-place.
    Aligns with C++ JSONL mode used for MMLU (no extra shift done here).
    """

    def __init__(self, path: str, seq_len: int, pad_id: int):
        self.samples = []
        self.seq_len = seq_len
        self.pad_id = pad_id
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    ids = rec.get("ids", [])
                    mask = rec.get("mask", [])
                    if not isinstance(ids, list) or not isinstance(mask, list):
                        continue
                    if len(ids) != len(mask) or not ids:
                        continue
                    ids = ids[:seq_len]
                    mask = mask[:seq_len]
                    if len(ids) < seq_len:
                        pad_n = seq_len - len(ids)
                        ids = ids + [pad_id] * pad_n
                        mask = mask + [0] * pad_n
                    ids_t = torch.tensor(ids, dtype=torch.long)
                    attn = torch.ones_like(ids_t, dtype=torch.long)
                    labels = torch.full_like(ids_t, -100)
                    mask_t = torch.tensor(mask, dtype=torch.long)
                    labels = torch.where(mask_t > 0, ids_t, labels)
                    self.samples.append({"input_ids": ids_t, "attention_mask": attn, "labels": labels})
                except Exception:
                    continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def cycle(loader: Iterable):
    while True:
        for item in loader:
            yield item


def build_target_modules(target_mode: str, override: str) -> List[str]:
    if override:
        return [m.strip() for m in override.split(",") if m.strip()]
    if target_mode == "attn":
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if target_mode == "light":
        return ["q_proj", "v_proj"]
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def make_scheduler(step: int, total_steps: int, warmup_steps: int, base_lr: float, mode: str) -> float:
    # Align with C++ Gemma: step is 1-indexed in C++, so we use step+1 here
    # C++ uses: if (step <= warmup_steps) return lr * step / warmup_steps
    # PyTorch step is 0-indexed, so step+1 corresponds to C++ step
    step_1indexed = step + 1
    if warmup_steps > 0 and step_1indexed <= warmup_steps:
        return base_lr * float(step_1indexed) / float(max(1, warmup_steps))
    remain = max(1, total_steps - warmup_steps)
    progress = float(step_1indexed - warmup_steps) / float(remain)
    progress = min(max(progress, 0.0), 1.0)
    if mode == "cosine":
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (1.0 - progress)


def evaluate(model, dataloader: DataLoader, device: torch.device, max_batches: int) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            losses.append(out.loss.item())
    model.train()
    if not losses:
        return float("inf")
    return math.exp(sum(losses) / len(losses))


def main():
    parser = argparse.ArgumentParser(description="PyTorch Gemma LoRA finetune (alignment build)")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--jsonl_train", type=str, default="")
    parser.add_argument("--jsonl_valid", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./gemma_lora_pt")
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16", "auto"],
        help="torch dtype for model weights (use float32 to mirror C++ alignment)",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--data_fraction", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_scheduler", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--target_mode", type=str, default="full", choices=["full", "attn", "light"])
    parser.add_argument("--lora_targets", type=str, default="")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_mode", type=str, default="sft", choices=["sft", "dpo", "kto"])
    parser.add_argument("--preference_jsonl", type=str, default="")
    parser.add_argument("--preference_valid_jsonl", type=str, default="")
    parser.add_argument("--binary_feedback_jsonl", type=str, default="")
    parser.add_argument("--binary_feedback_valid_jsonl", type=str, default="")
    parser.add_argument("--reference_model_dir", type=str, default="")
    parser.add_argument("--dpo_beta", type=float, default=0.1)
    parser.add_argument("--dpo_loss_type", type=str, default="sigmoid", choices=["sigmoid", "robust", "ipo", "sigmoid_norm"])
    parser.add_argument("--kto_desirable_weight", type=float, default=1.0)
    parser.add_argument("--kto_undesirable_weight", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.torch_dtype == "float32":
        torch_dtype = torch.float32
    elif args.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = None  # let HF decide

    tok = AutoTokenizer.from_pretrained(args.model_dir, padding_side="right")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    use_jsonl = bool(args.jsonl_train)
    use_dpo = args.train_mode == "dpo"
    use_kto = args.train_mode == "kto"
    if use_dpo:
        if not args.preference_jsonl:
            raise SystemExit("--preference_jsonl is required when --train_mode dpo")
        train_dataset = PreferenceDataset(args.preference_jsonl, tok, args.seq_len)
        eval_dataset = (
            PreferenceDataset(args.preference_valid_jsonl, tok, args.seq_len)
            if args.preference_valid_jsonl
            else train_dataset
        )
        collate_fn = dpo_collate_batch
    elif use_kto:
        if not args.binary_feedback_jsonl:
            raise SystemExit("--binary_feedback_jsonl is required when --train_mode kto")
        train_dataset = BinaryFeedbackDataset(args.binary_feedback_jsonl, tok, args.seq_len)
        eval_dataset = (
            BinaryFeedbackDataset(args.binary_feedback_valid_jsonl, tok, args.seq_len)
            if args.binary_feedback_valid_jsonl
            else train_dataset
        )
        collate_fn = binary_feedback_collate_batch
    elif use_jsonl:
        train_dataset: Dataset = JsonlMaskedDataset(args.jsonl_train, args.seq_len, tok.pad_token_id)
        eval_dataset: Dataset = (
            JsonlMaskedDataset(args.jsonl_valid, args.seq_len, tok.pad_token_id)
            if args.jsonl_valid
            else train_dataset
        )
        collate_fn = collate_batch
    else:
        train_dataset = WikiTextDataset(
            os.path.join(args.data_dir, "wiki.train.raw"),
            tok,
            seq_len=args.seq_len,
            eos_token_id=tok.eos_token_id,
            data_fraction=args.data_fraction,
            insert_eos_between_lines=True,
            drop_last=True,
        )
        eval_dataset = WikiTextDataset(
            os.path.join(args.data_dir, "wiki.valid.raw"),
            tok,
            seq_len=args.seq_len,
            eos_token_id=tok.eos_token_id,
            data_fraction=1.0,
            insert_eos_between_lines=True,
            drop_last=False,
        )
        collate_fn = collate_batch

    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, args.batch),
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=max(1, args.batch),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch_dtype)
    ref_model = None
    if use_dpo or use_kto:
        ref_dir = args.reference_model_dir or args.model_dir
        ref_model = AutoModelForCausalLM.from_pretrained(ref_dir, torch_dtype=torch_dtype).to(device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
    target_modules = build_target_modules(args.target_mode, args.lora_targets)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

    micro_per_epoch = math.ceil(len(train_loader))
    total_updates = (
        math.ceil(micro_per_epoch / max(1, args.grad_accum)) * args.epochs
    )
    if args.max_steps and args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = total_updates
    warmup_steps = int(total_steps * args.warmup_ratio)

    print("\n========== PyTorch Gemma LoRA Finetune (alignment) ==========")
    print(f"Train sequences: {len(train_dataset)}, Eval sequences: {len(eval_dataset)}")
    print(f"Total steps: {total_steps}, grad_accum: {args.grad_accum}, warmup_steps: {warmup_steps}")
    print(f"LoRA r/alpha/dropout: {args.lora_r}/{args.lora_alpha}/{args.lora_dropout}")
    print(f"Targets: {','.join(target_modules)}")

    model.to(device)
    model.train()

    global_step = 0
    ema_loss = None
    token_counter = 0
    train_iter = cycle(train_loader)

    while global_step < total_steps:
        accum_loss = 0.0
        accum_tokens = 0
        optimizer.zero_grad()
        for _ in range(max(1, args.grad_accum)):
            batch = next(train_iter)
            batch = {k: v.to(device) for k, v in batch.items()}
            if use_dpo:
                dpo_metrics = compute_dpo_batch_metrics(model, ref_model, batch, args.dpo_beta, args.dpo_loss_type)
                loss = dpo_metrics["loss"]
            elif use_kto:
                kto_metrics = compute_kto_batch_metrics(
                    model,
                    ref_model,
                    batch,
                    args.dpo_beta,
                    desirable_weight=args.kto_desirable_weight,
                    undesirable_weight=args.kto_undesirable_weight,
                )
                loss = kto_metrics["loss"]
            else:
                out = model(**batch)
                loss = out.loss
            (loss / max(1, args.grad_accum)).backward()
            accum_loss += loss.item()
            if use_dpo:
                accum_tokens += int(batch["chosen_attention_mask"].sum().item())
            elif use_kto:
                accum_tokens += int(batch["attention_mask"].sum().item())
            else:
                accum_tokens += int(batch["attention_mask"].sum().item())

        if args.max_grad_norm and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)

        lr_cur = make_scheduler(global_step, total_steps, warmup_steps, args.learning_rate, args.lr_scheduler)
        for g in optimizer.param_groups:
            g["lr"] = lr_cur
        optimizer.step()

        global_step += 1
        token_counter += accum_tokens
        avg_loss = accum_loss / float(max(1, args.grad_accum))
        if ema_loss is None:
            ema_loss = avg_loss
        else:
            beta = 0.9
            ema_loss = beta * ema_loss + (1.0 - beta) * avg_loss

        if global_step % max(1, args.logging_steps) == 0:
            if use_dpo:
                print(
                    f"[Train] step {global_step}/{total_steps} "
                    f"lr {lr_cur:.6f} loss {avg_loss:.4f} "
                    f"reward_acc {dpo_metrics['reward_accuracy'].item():.4f} "
                    f"reward_margin {dpo_metrics['reward_margin'].item():.4f} "
                    f"chosen_reward {dpo_metrics['chosen_reward'].item():.4f} "
                    f"rejected_reward {dpo_metrics['rejected_reward'].item():.4f} "
                    f"tokens {accum_tokens}"
                )
            elif use_kto:
                print(
                    f"[Train] step {global_step}/{total_steps} "
                    f"lr {lr_cur:.6f} loss {avg_loss:.4f} "
                    f"reward_acc {kto_metrics['reward_accuracy'].item():.4f} "
                    f"reward_margin {kto_metrics['reward_margin'].item():.4f} "
                    f"chosen_reward {kto_metrics['chosen_reward'].item():.4f} "
                    f"rejected_reward {kto_metrics['rejected_reward'].item():.4f} "
                    f"tokens {accum_tokens}"
                )
            else:
                ppl = math.exp(avg_loss)
                print(
                    f"[Train] step {global_step}/{total_steps} "
                    f"lr {lr_cur:.6f} loss {avg_loss:.4f} ppl {ppl:.2f} tokens {accum_tokens}"
                )

        if args.eval_steps > 0 and global_step % args.eval_steps == 0:
            if use_dpo:
                eval_metrics = evaluate_dpo(model, ref_model, eval_loader, device, args.eval_batches, args.dpo_beta, args.dpo_loss_type)
                print(
                    f"[Eval] step {global_step}/{total_steps} "
                    f"loss {eval_metrics['loss']:.4f} "
                    f"reward_acc {eval_metrics['reward_accuracy']:.4f} "
                    f"reward_margin {eval_metrics['reward_margin']:.4f} "
                    f"ema_loss {ema_loss:.4f} total_tokens {token_counter}"
                )
            elif use_kto:
                eval_metrics = evaluate_kto(
                    model,
                    ref_model,
                    eval_loader,
                    device,
                    args.eval_batches,
                    args.dpo_beta,
                    desirable_weight=args.kto_desirable_weight,
                    undesirable_weight=args.kto_undesirable_weight,
                )
                print(
                    f"[Eval] step {global_step}/{total_steps} "
                    f"loss {eval_metrics['loss']:.4f} "
                    f"reward_acc {eval_metrics['reward_accuracy']:.4f} "
                    f"reward_margin {eval_metrics['reward_margin']:.4f} "
                    f"ema_loss {ema_loss:.4f} total_tokens {token_counter}"
                )
            else:
                valid_ppl = evaluate(model, eval_loader, device, args.eval_batches)
                print(
                    f"[Eval] step {global_step}/{total_steps} valid_ppl {valid_ppl:.2f} "
                    f"ema_loss {ema_loss:.4f} total_tokens {token_counter}"
                )

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"\n🎉 Gemma LoRA training done. Saved adapter to {args.output_dir}")
    print(f"Total steps {global_step}, total tokens {token_counter}, final EMA loss {ema_loss:.4f}")


if __name__ == "__main__":
    main()
