from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel

from pytorch_alignment.dpo_utils import BinaryFeedbackDataset
from pytorch_alignment.kto_trainer import KtoTrainingConfig, train_kto


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


def build_tiny_policy(vocab_size: int, seq_len: int):
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=seq_len,
        n_ctx=seq_len,
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


def resolve_qwen_model_dir(repo_root: Path, explicit_model_dir: Path | None) -> Path:
    if explicit_model_dir is not None:
        return explicit_model_dir.expanduser().resolve()
    env_model_dir = os.environ.get("QWEN_MODEL_DIR")
    if env_model_dir:
        return Path(env_model_dir).expanduser().resolve()
    return (repo_root / "examples/qwen_lora_finetune/pretrained").resolve()


def build_qwen_lora_config(rank: int, alpha: int, dropout: float) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _dtype_for_device(device: torch.device):
    if device.type == "cuda":
        return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def _ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def build_qwen_policy_and_reference(
    model_dir: Path,
    device: torch.device,
    *,
    rank: int,
    alpha: int,
    dropout: float,
):
    torch_dtype = _dtype_for_device(device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    _ensure_pad_token(tokenizer)
    policy_base = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=torch_dtype,
        trust_remote_code=True,
    )
    policy_base.config.use_cache = False
    policy_model = get_peft_model(policy_base, build_qwen_lora_config(rank, alpha, dropout))

    reference_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=torch_dtype,
        trust_remote_code=True,
    )
    reference_model.config.use_cache = False
    return tokenizer, policy_model, reference_model


def run_tiny_kto_from_jsonl(
    *,
    kto_jsonl: Path,
    eval_jsonl: Path,
    out_dir: Path,
    max_steps: int,
    batch_size: int,
    seq_len: int,
    beta: float,
    lr: float,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = SimpleCharTokenizer()
    train_dataset = BinaryFeedbackDataset(str(kto_jsonl), tokenizer, seq_len=seq_len)
    eval_dataset = BinaryFeedbackDataset(str(eval_jsonl), tokenizer, seq_len=seq_len)
    if len(train_dataset) == 0:
        raise ValueError(f"no KTO training samples loaded from {kto_jsonl}")

    device = torch.device("cpu")
    torch.manual_seed(seed)
    policy_model = build_tiny_policy(tokenizer.vocab_size, seq_len)
    reference_model = build_tiny_policy(tokenizer.vocab_size, seq_len)
    reference_model.load_state_dict(policy_model.state_dict())

    report = train_kto(
        policy_model,
        reference_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=KtoTrainingConfig(
            max_steps=max_steps,
            batch_size=batch_size,
            beta=beta,
            lr=lr,
            desirable_weight=desirable_weight,
            undesirable_weight=undesirable_weight,
            eval_max_batches=10,
            seed=seed,
            metrics_path=out_dir / "metrics.ndjson",
        ),
        device=device,
    )
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def run_qwen_kto_from_jsonl(
    *,
    kto_jsonl: Path,
    eval_jsonl: Path,
    out_dir: Path,
    model_dir: Path,
    max_steps: int,
    batch_size: int,
    seq_len: int,
    beta: float,
    lr: float,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
    seed: int = 42,
    device_name: str = "auto",
    lora_rank: int = 4,
    lora_alpha: int = 8,
    lora_dropout: float = 0.0,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(device_name)
    torch.manual_seed(seed)
    tokenizer, policy_model, reference_model = build_qwen_policy_and_reference(
        model_dir,
        device,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )
    train_dataset = BinaryFeedbackDataset(str(kto_jsonl), tokenizer, seq_len=seq_len)
    eval_dataset = BinaryFeedbackDataset(str(eval_jsonl), tokenizer, seq_len=seq_len)
    if len(train_dataset) == 0:
        raise ValueError(f"no KTO training samples loaded from {kto_jsonl}")

    report = train_kto(
        policy_model,
        reference_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=KtoTrainingConfig(
            max_steps=max_steps,
            batch_size=batch_size,
            beta=beta,
            lr=lr,
            desirable_weight=desirable_weight,
            undesirable_weight=undesirable_weight,
            eval_max_batches=10,
            seed=seed,
            metrics_path=out_dir / "metrics.ndjson",
        ),
        device=device,
    )
    report["backend"] = "qwen"
    report["model_dir"] = str(model_dir)
    report["device"] = str(device)
    report["lora_rank"] = lora_rank
    report["lora_alpha"] = lora_alpha
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run KTO alignment smoke training. Default backend: Qwen3-0.6B.")
    parser.add_argument("--backend", choices=["qwen", "tiny"], default="qwen")
    parser.add_argument("--model_dir", type=Path, default=None, help="Qwen3-0.6B HuggingFace model directory.")
    parser.add_argument("--kto_jsonl", type=Path, default=repo_root / "android-visualizer/app/src/main/assets/mft_demo_data/fedrl_health_kto_train.jsonl")
    parser.add_argument("--eval_jsonl", type=Path, default=repo_root / "android-visualizer/app/src/main/assets/mft_demo_data/fedrl_health_kto_eval.jsonl")
    parser.add_argument("--out_dir", type=Path, default=repo_root / "runs/kto_health_smoke")
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=192)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--desirable_weight", type=float, default=1.0)
    parser.add_argument("--undesirable_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda.")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if args.backend == "tiny":
        report = run_tiny_kto_from_jsonl(
            kto_jsonl=args.kto_jsonl,
            eval_jsonl=args.eval_jsonl,
            out_dir=args.out_dir,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            beta=args.beta,
            lr=args.lr,
            desirable_weight=args.desirable_weight,
            undesirable_weight=args.undesirable_weight,
            seed=args.seed,
        )
        report["backend"] = "tiny"
    else:
        report = run_qwen_kto_from_jsonl(
            kto_jsonl=args.kto_jsonl,
            eval_jsonl=args.eval_jsonl,
            out_dir=args.out_dir,
            model_dir=resolve_qwen_model_dir(repo_root, args.model_dir),
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            beta=args.beta,
            lr=args.lr,
            desirable_weight=args.desirable_weight,
            undesirable_weight=args.undesirable_weight,
            seed=args.seed,
            device_name=args.device,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
