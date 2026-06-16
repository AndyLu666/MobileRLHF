#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

try:
    from qa_preference_tools import (
        FakeAnswerJudge,
        GoldAnswerJudge,
        QwenAnswerJudge,
        ToolStats,
        iter_jsonl,
        normalize_qa_row,
        pick_text,
        write_jsonl_row,
    )
except ImportError:
    import sys

    ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(ROOT))
    from qa_preference_tools import (
        FakeAnswerJudge,
        GoldAnswerJudge,
        QwenAnswerJudge,
        ToolStats,
        iter_jsonl,
        normalize_qa_row,
        pick_text,
        write_jsonl_row,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Judge QA responses as single-sided preference feedback. The main "
            "output is prompt/response/label JSONL for BinaryFeedback/KTO-style "
            "training; --pair_output is an optional pairwise-DPO export."
        )
    )
    parser.add_argument("--input", required=True, help="Input QA JSONL.")
    parser.add_argument("--output", required=True, help="Judged output JSONL.")
    parser.add_argument("--pair_output", default="", help="Optional DPO prompt/chosen/rejected JSONL.")
    parser.add_argument("--max_pairs_per_prompt", type=int, default=1, help="Max grouped DPO pairs per prompt; 0 means all.")
    parser.add_argument("--backend", choices=["qwen", "gold", "fake"], default="gold")
    parser.add_argument("--mode", choices=["single", "pair", "gold"], default="single")
    parser.add_argument("--model_dir", default="", help="Local Qwen model/tokenizer directory for --backend qwen.")
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="", help="Optional torch device override.")
    parser.add_argument("--stats", default="", help="Optional stats JSON path.")
    return parser.parse_args()


def build_judge(args: argparse.Namespace):
    if args.backend == "fake":
        return FakeAnswerJudge()
    if args.backend == "gold":
        return GoldAnswerJudge()
    if not args.model_dir:
        raise SystemExit("--model_dir is required when --backend qwen")
    return QwenAnswerJudge(
        model_dir=args.model_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        device=args.device,
    )


def convert(args: argparse.Namespace) -> ToolStats:
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pair_path = Path(args.pair_output) if args.pair_output else None
    if pair_path:
        pair_path.parent.mkdir(parents=True, exist_ok=True)

    judge = build_judge(args)
    stats = ToolStats()
    grouped = defaultdict(lambda: {"chosen": [], "rejected": []})

    pair_handle = pair_path.open("w", encoding="utf-8") if pair_path else None
    try:
        with output_path.open("w", encoding="utf-8") as out:
            for line_no, row in iter_jsonl(input_path, stats):
                if row is None:
                    continue
                qa = normalize_qa_row(row, line_no)
                if qa is None:
                    stats.skipped_missing_fields += 1
                    continue
                try:
                    judgment = judge.judge(qa.prompt, qa.response, qa.gold)
                except Exception as exc:
                    stats.skipped_judge_parse += 1
                    write_jsonl_row(
                        out,
                        {
                            "prompt": qa.prompt,
                            "response": qa.response,
                            "label": "unknown",
                            "is_correct": None,
                            "confidence": 0.0,
                            "reason": f"judge failed: {exc}",
                            "source": getattr(judge, "source", "answer_judge"),
                            "training_format": "single_feedback",
                            "meta": {"input_line": line_no, "backend": args.backend},
                        },
                    )
                    stats.unknown_labels += 1
                    continue

                judged_row = {
                    "prompt": qa.prompt,
                    "response": qa.response,
                    "is_correct": judgment.is_correct,
                    "label": judgment.label,
                    "confidence": judgment.confidence,
                    "reason": judgment.reason,
                    "source": judgment.source,
                    "training_format": "single_feedback",
                    "meta": {"input_line": line_no, "backend": args.backend, "mode": args.mode},
                }
                if judgment.raw:
                    judged_row["meta"]["raw_judge_output"] = judgment.raw
                write_jsonl_row(out, judged_row)
                stats.kept += 1
                if judgment.is_correct:
                    stats.chosen_labels += 1
                else:
                    stats.unchosen_labels += 1

                if pair_handle:
                    rejected = pick_text(row, ("rejected", "unchosen", "bad_response", "response_b"))
                    chosen = pick_text(row, ("chosen", "good_response", "response_a"))
                    if args.mode == "pair" and chosen and rejected:
                        write_jsonl_row(
                            pair_handle,
                            {
                                "prompt": qa.prompt,
                                "chosen": chosen,
                                "rejected": rejected,
                                "source": f"{judgment.source}_pair",
                                "meta": {"input_line": line_no},
                            },
                        )
                        stats.exported_pairs += 1
                    elif rejected and judgment.is_correct:
                        write_jsonl_row(
                            pair_handle,
                            {
                                "prompt": qa.prompt,
                                "chosen": qa.response,
                                "rejected": rejected,
                                "source": f"{judgment.source}_pair",
                                "meta": {"input_line": line_no},
                            },
                        )
                        stats.exported_pairs += 1
                    else:
                        branch = "chosen" if judgment.is_correct else "rejected"
                        grouped[qa.prompt][branch].append(
                            {
                                "response": qa.response,
                                "line_no": line_no,
                                "confidence": judgment.confidence,
                            }
                        )
            if pair_handle:
                for prompt, bucket in grouped.items():
                    chosen_rows = bucket["chosen"]
                    rejected_rows = bucket["rejected"]
                    if not chosen_rows or not rejected_rows:
                        continue
                    pair_count = 0
                    for chosen_row in chosen_rows:
                        for rejected_row in rejected_rows:
                            if args.max_pairs_per_prompt and pair_count >= args.max_pairs_per_prompt:
                                break
                            write_jsonl_row(
                                pair_handle,
                                {
                                    "prompt": prompt,
                                    "chosen": chosen_row["response"],
                                    "rejected": rejected_row["response"],
                                    "source": f"{getattr(judge, 'source', 'answer_judge')}_grouped_pair",
                                    "meta": {
                                        "chosen_line": chosen_row["line_no"],
                                        "rejected_line": rejected_row["line_no"],
                                        "chosen_confidence": chosen_row["confidence"],
                                        "rejected_confidence": rejected_row["confidence"],
                                    },
                                },
                            )
                            stats.exported_pairs += 1
                            pair_count += 1
                        if args.max_pairs_per_prompt and pair_count >= args.max_pairs_per_prompt:
                            break
    finally:
        if pair_handle:
            pair_handle.close()
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
