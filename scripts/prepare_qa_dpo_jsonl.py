#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from qa_preference_tools import (
        FakeNegativeGenerator,
        QwenNegativeGenerator,
        ToolStats,
        iter_jsonl,
        normalize_qa_row,
        too_similar,
        write_jsonl_row,
    )
except ImportError:
    import sys

    ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(ROOT))
    from qa_preference_tools import (
        FakeNegativeGenerator,
        QwenNegativeGenerator,
        ToolStats,
        iter_jsonl,
        normalize_qa_row,
        too_similar,
        write_jsonl_row,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert QA JSONL to DPO prompt/chosen/rejected JSONL.")
    parser.add_argument("--input", required=True, help="Input QA JSONL.")
    parser.add_argument("--output", required=True, help="Output DPO preference JSONL.")
    parser.add_argument("--backend", choices=["qwen", "fake"], default="qwen")
    parser.add_argument("--model_dir", default="", help="Local Qwen model/tokenizer directory for --backend qwen.")
    parser.add_argument("--mode", choices=["generate-negative"], default="generate-negative")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="", help="Optional torch device override.")
    parser.add_argument("--fake_rejected", default="incorrect answer", help="Deterministic rejected text for tests.")
    parser.add_argument("--similarity_threshold", type=float, default=0.85)
    parser.add_argument("--skip_existing_rejected", action="store_true")
    parser.add_argument("--overwrite_rejected", action="store_true")
    parser.add_argument("--stats", default="", help="Optional stats JSON path.")
    return parser.parse_args()


def build_generator(args: argparse.Namespace):
    if args.backend == "fake":
        return FakeNegativeGenerator(args.fake_rejected)
    if not args.model_dir:
        raise SystemExit("--model_dir is required when --backend qwen")
    return QwenNegativeGenerator(
        model_dir=args.model_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        device=args.device,
    )


def convert(args: argparse.Namespace) -> ToolStats:
    if args.skip_existing_rejected and args.overwrite_rejected:
        raise SystemExit("Choose only one of --skip_existing_rejected or --overwrite_rejected")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator = build_generator(args)
    stats = ToolStats()

    with output_path.open("w", encoding="utf-8") as out:
        for line_no, row in iter_jsonl(input_path, stats):
            if row is None:
                continue
            qa = normalize_qa_row(row, line_no)
            if qa is None:
                stats.skipped_missing_fields += 1
                continue
            existing_rejected = row.get("rejected")
            if existing_rejected and args.skip_existing_rejected:
                stats.skipped_existing_rejected += 1
                continue
            rejected = str(existing_rejected).strip() if existing_rejected and not args.overwrite_rejected else ""
            if not rejected:
                rejected = generator.generate(qa.prompt, qa.response).strip()
            if too_similar(rejected, qa.response, threshold=args.similarity_threshold):
                stats.skipped_generation_guard += 1
                continue
            write_jsonl_row(
                out,
                {
                    "prompt": qa.prompt,
                    "chosen": qa.response,
                    "rejected": rejected,
                    "source": generator.source,
                    "meta": {
                        "input_line": line_no,
                        "backend": args.backend,
                        "mode": args.mode,
                    },
                },
            )
            stats.kept += 1
    return stats


def main() -> None:
    args = parse_args()
    stats = convert(args)
    stats_json = json.dumps(stats.to_dict(), ensure_ascii=False, indent=2)
    if args.stats:
        Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats).write_text(stats_json + "\n", encoding="utf-8")
    print(stats_json)


if __name__ == "__main__":
    main()
