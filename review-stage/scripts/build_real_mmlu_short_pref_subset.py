import argparse
import json
from pathlib import Path

from transformers import GPT2TokenizerFast


def kept_at_seq_len(tokenizer, rec: dict, seq_len: int) -> tuple[bool, int]:
    prompt_ids = tokenizer.encode(rec["prompt"], add_special_tokens=False)
    chosen_ids = tokenizer.encode(rec["chosen"], add_special_tokens=False)
    rejected_ids = tokenizer.encode(rec["rejected"], add_special_tokens=False)

    chosen_resp = 0 if len(prompt_ids) >= seq_len else min(len(chosen_ids), seq_len - len(prompt_ids))
    rejected_resp = 0 if len(prompt_ids) >= seq_len else min(len(rejected_ids), seq_len - len(prompt_ids))
    return chosen_resp > 0 and rejected_resp > 0, len(prompt_ids)


def select_rows(input_path: Path, tokenizer, seq_len: int, limit: int) -> list[dict]:
    kept_rows: list[tuple[int, int, dict]] = []
    with input_path.open("r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin, 1):
            rec = json.loads(line)
            ok, prompt_len = kept_at_seq_len(tokenizer, rec, seq_len)
            if ok:
                kept_rows.append((prompt_len, idx, rec))
    kept_rows.sort(key=lambda x: (x[0], x[1]))
    return [rec for _, _, rec in kept_rows[:limit]]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fout:
        for rec in rows:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a short real-MMLU preference subset that survives truncation.")
    parser.add_argument("--pretrained-dir", default="gpt2_small_lora_finetune/pretrained/gpt2")
    parser.add_argument("--train-input", default="review-stage/data/real_mmlu_pref_train.jsonl")
    parser.add_argument("--eval-input", default="review-stage/data/real_mmlu_pref_eval.jsonl")
    parser.add_argument("--train-output", default="review-stage/data/real_mmlu_pref_short_train.jsonl")
    parser.add_argument("--eval-output", default="review-stage/data/real_mmlu_pref_short_eval.jsonl")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=16)
    parser.add_argument("--eval-size", type=int, default=8)
    args = parser.parse_args()

    tok = GPT2TokenizerFast.from_pretrained(args.pretrained_dir)
    tok.pad_token = tok.eos_token

    train_rows = select_rows(Path(args.train_input), tok, args.seq_len, args.train_size)
    eval_rows = select_rows(Path(args.eval_input), tok, args.seq_len, args.eval_size)

    write_jsonl(Path(args.train_output), train_rows)
    write_jsonl(Path(args.eval_output), eval_rows)

    print(
        json.dumps(
            {
                "status": "ok",
                "seq_len": args.seq_len,
                "train_size": len(train_rows),
                "eval_size": len(eval_rows),
                "train_output": args.train_output,
                "eval_output": args.eval_output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
