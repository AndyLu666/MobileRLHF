import argparse
import json
import os
import random
import re
from typing import Dict, List


SYSTEM_PROMPT = (
    "You generate a rejected answer for preference training. "
    "The rejected answer must stay relevant to the prompt and stylistically similar to the chosen answer, "
    "but should be slightly worse: less accurate, less complete, or containing a subtle mistake. "
    "Do not produce nonsense unless the chosen answer is extremely short. "
    "Return JSON with a single key: rejected."
)


def perturb_numbers(text: str) -> str:
    def repl(match):
        value = match.group(0)
        try:
            num = int(value)
            return str(num + 1 if num >= 0 else num - 1)
        except Exception:
            return value

    return re.sub(r"\b\d+\b", repl, text, count=1)


def weaken_absolute_claims(text: str) -> str:
    swaps = [
        (" is ", " can be "),
        (" are ", " may be "),
        (" improves ", " may improve "),
        (" reduces ", " may reduce "),
        (" keeps ", " can keep "),
        (" makes ", " may make "),
        (" helps ", " can help "),
    ]
    out = text
    for src, dst in swaps:
        if src in out:
            return out.replace(src, dst, 1)
    return out


def drop_supporting_detail(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) >= 3:
        return " ".join(sentences[:-1]).strip()
    return text


def substitute_near_word(text: str) -> str:
    pairs = [
        ("memory", "attention"),
        ("learning", "practice"),
        ("warm", "hot"),
        ("privacy", "security"),
        ("reliable", "simple"),
        ("faster", "easier"),
        ("warmer", "cooler"),
        ("limited", "shared"),
        ("stable", "quick"),
        ("generalize", "fit"),
        ("trust", "interest"),
    ]
    out = text
    for src, dst in pairs:
        if src in out:
            return out.replace(src, dst, 1)
    return out


def heuristic_rejected(prompt: str, chosen: str, seed: int) -> str:
    random.seed(seed)
    variants = [
        perturb_numbers(chosen),
        weaken_absolute_claims(chosen),
        drop_supporting_detail(chosen),
        substitute_near_word(chosen),
    ]
    variants = [v for v in variants if v and v != chosen]
    if not variants:
        return chosen.rstrip(".") + ", but not always."
    candidate = random.choice(variants)
    if candidate == chosen:
        candidate = chosen.rstrip(".") + ", but not always."
    return candidate


def openai_compatible_rejected(prompt: str, chosen: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps({"prompt": prompt, "chosen": chosen}, ensure_ascii=False),
            },
        ],
    )
    text = response.output_text.strip()
    try:
        obj = json.loads(text)
        return obj["rejected"]
    except Exception:
        return text


def main():
    parser = argparse.ArgumentParser(description="Generate rejected answers from chosen-only SFT JSONL")
    parser.add_argument("--input", required=True, help="Chosen-only JSONL with prompt/chosen")
    parser.add_argument("--output", required=True, help="Preference JSONL with prompt/chosen/rejected")
    parser.add_argument("--mode", choices=["heuristic", "openai"], default="heuristic")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows: List[Dict[str, str]] = []
    with open(args.input, "r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            if not line.strip():
                continue
            rec = json.loads(line)
            prompt = rec["prompt"]
            chosen = rec["chosen"]
            if args.mode == "openai":
                rejected = openai_compatible_rejected(prompt, chosen, args.model)
            else:
                rejected = heuristic_rejected(prompt, chosen, args.seed + idx)
            rows.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "generator_mode": args.mode,
                }
            )

    with open(args.output, "w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"status": "ok", "count": len(rows), "mode": args.mode, "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
