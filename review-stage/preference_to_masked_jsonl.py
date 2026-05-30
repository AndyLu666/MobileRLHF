import argparse
import json

from transformers import GPT2TokenizerFast


def main():
    parser = argparse.ArgumentParser(description="Convert preference JSONL to chosen-only masked JSONL")
    parser.add_argument("--input", required=True, help="Preference JSONL with prompt/chosen/rejected")
    parser.add_argument("--output", required=True, help="Output JSONL with ids/mask for chosen-only SFT")
    parser.add_argument("--tokenizer_dir", required=True, help="Directory loadable by GPT2TokenizerFast")
    parser.add_argument("--seq_len", type=int, default=128)
    args = parser.parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer_dir)
    kept = 0
    skipped = 0

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            prompt = rec["prompt"]
            chosen = rec["chosen"]
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            chosen_ids = tokenizer.encode(chosen, add_special_tokens=False)
            ids = (prompt_ids + chosen_ids)[: args.seq_len]
            mask = ([0] * len(prompt_ids) + [1] * len(chosen_ids))[: args.seq_len]
            if not any(mask):
                skipped += 1
                continue
            fout.write(json.dumps({"ids": ids, "mask": mask}) + "\n")
            kept += 1

    print(json.dumps({"status": "ok", "kept": kept, "skipped": skipped, "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
