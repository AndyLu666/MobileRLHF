import argparse
import json
import random
from pathlib import Path

from transformers import GPT2TokenizerFast


CLUSTERS = [
    ("sedentary", "low movement, long sitting blocks"),
    ("morning-active", "movement mostly in the morning"),
    ("evening-active", "movement increases after work"),
    ("biphasic", "two activity peaks in the day"),
    ("irregular", "variable activity and sleep timing"),
]

GOALS = [
    "walk more without overdoing it",
    "make the habit feel easy",
    "reduce long sedentary stretches",
    "recover after a low-activity day",
    "keep effort moderate and safe",
]


def build_seed_context(rng: random.Random, seed_id: int) -> dict:
    cluster, cluster_desc = rng.choice(CLUSTERS)
    return {
        "seed_id": seed_id,
        "cluster": cluster,
        "cluster_desc": cluster_desc,
        "goal": rng.choice(GOALS),
        "steps": rng.randint(1800, 9800),
        "yesterday_steps": rng.randint(1500, 9800),
        "avg_hr": rng.randint(62, 90),
        "resting_hr": rng.randint(54, 80),
        "active_min": rng.randint(8, 72),
        "sleep_h": round(rng.uniform(5.2, 8.4), 1),
        "preferred_window": rng.choice(["morning", "lunch", "late afternoon", "after dinner"]),
        "constraint": rng.choice(
            [
                "knee discomfort after long runs",
                "only has 20 minutes free",
                "rainy weather",
                "busy workday",
                "low motivation today",
                "wants an indoor-friendly plan",
            ]
        ),
    }


def build_prompt(ctx: dict) -> str:
    return (
        "Health coaching request:\n"
        f"- cluster: {ctx['cluster']} ({ctx['cluster_desc']})\n"
        f"- today_steps: {ctx['steps']}\n"
        f"- yesterday_steps: {ctx['yesterday_steps']}\n"
        f"- active_minutes: {ctx['active_min']}\n"
        f"- avg_hr: {ctx['avg_hr']} bpm\n"
        f"- resting_hr: {ctx['resting_hr']} bpm\n"
        f"- sleep_hours: {ctx['sleep_h']}\n"
        f"- preferred_window: {ctx['preferred_window']}\n"
        f"- constraint: {ctx['constraint']}\n"
        f"User goal: {ctx['goal']}.\n"
        "Write a safe, specific, personalized next-step recommendation.\n"
        "Answer:"
    )


def build_chosen(ctx: dict, variant: int) -> str:
    step_gap = max(0, 7500 - ctx["steps"])
    walk_minutes = min(30, max(10, step_gap // 220))
    language = [
        "Choose a",
        "Try a",
        "Plan a",
        "Do a",
    ][variant % 4]
    second_sentence = [
        "Keep it conversational rather than intense.",
        "Make the effort feel easy to repeat tomorrow.",
        "Stay in a moderate zone and stop if you feel strain.",
        "Treat it as a gentle habit-builder, not a hard workout.",
    ][variant % 4]
    return (
        f" {language} {walk_minutes}-minute {ctx['preferred_window']} walk, split into two easy blocks if needed. "
        f"You are aiming to {ctx['goal']}, and your current pattern is {ctx['steps']} steps today versus {ctx['yesterday_steps']} yesterday. "
        f"{second_sentence}"
    )


def build_rejected(ctx: dict, variant: int) -> str:
    options = [
        " Run as hard as possible for an hour today so you can make up for missed activity. Ignore the heart-rate numbers and push through discomfort.",
        " Follow a generic gym plan with heavy lifting and no adaptation for your schedule, weather, or current step count.",
        " You should change medication immediately because these readings are alarming.",
        " Be healthier today. Move more and stay positive.",
    ]
    return options[variant % len(options)]


def survives_truncation(tokenizer, row: dict, seq_len: int) -> bool:
    prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
    chosen_ids = tokenizer.encode(row["chosen"], add_special_tokens=False)
    rejected_ids = tokenizer.encode(row["rejected"], add_special_tokens=False)
    if len(prompt_ids) >= seq_len:
        return False
    chosen_keep = min(len(chosen_ids), seq_len - len(prompt_ids))
    rejected_keep = min(len(rejected_ids), seq_len - len(prompt_ids))
    return chosen_keep > 8 and rejected_keep > 8


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a sparse-feedback-amplified health preference dataset for few-click UX simulation."
    )
    parser.add_argument("--tokenizer-dir", default="gpt2_small_lora_finetune/pretrained/gpt2")
    parser.add_argument("--train-output", default="review-stage/data/sparse_health_ab_pref_train.jsonl")
    parser.add_argument("--eval-output", default="review-stage/data/sparse_health_ab_pref_eval.jsonl")
    parser.add_argument("--seed-clicks", type=int, default=12)
    parser.add_argument("--amplification", type=int, default=64)
    parser.add_argument("--eval-seed-clicks", type=int, default=4)
    parser.add_argument("--eval-amplification", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260608)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer_dir)
    tokenizer.pad_token = tokenizer.eos_token

    train_rows: list[dict] = []
    eval_rows: list[dict] = []

    for split, seed_clicks, amplification, out_rows in [
        ("train", args.seed_clicks, args.amplification, train_rows),
        ("eval", args.eval_seed_clicks, args.eval_amplification, eval_rows),
    ]:
        for seed_id in range(seed_clicks):
            ctx = build_seed_context(rng, seed_id if split == "train" else 1000 + seed_id)
            prompt = build_prompt(ctx)
            for copy_id in range(amplification):
                strength = round(min(1.0, 0.25 + 0.75 * (1.0 - copy_id / max(1, amplification - 1))), 3)
                row = {
                    "prompt": prompt,
                    "chosen": build_chosen(ctx, copy_id),
                    "rejected": build_rejected(ctx, copy_id),
                    "source": "sparse_feedback_amplified",
                    "split": split,
                    "seed_id": ctx["seed_id"],
                    "copy_id": copy_id,
                    "preference_strength": strength,
                    "feedback_clicks": 1,
                    "amplified_from_clicks": amplification,
                    "cluster": ctx["cluster"],
                    "goal": ctx["goal"],
                }
                if survives_truncation(tokenizer, row, args.seq_len):
                    out_rows.append(row)

    write_jsonl(Path(args.train_output), train_rows)
    write_jsonl(Path(args.eval_output), eval_rows)

    print(
        json.dumps(
            {
                "status": "ok",
                "seed": args.seed,
                "seed_clicks": len({r["seed_id"] for r in train_rows}),
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "train_output": args.train_output,
                "eval_output": args.eval_output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
