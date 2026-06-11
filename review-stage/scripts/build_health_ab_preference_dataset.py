import argparse
import json
import random
from pathlib import Path

from transformers import GPT2TokenizerFast


CLUSTERS = [
    ("sedentary", "low daytime movement and long sitting blocks"),
    ("morning-active", "most activity happens before noon"),
    ("evening-active", "activity increases after work or dinner"),
    ("biphasic", "two moderate activity peaks in the day"),
    ("irregular", "large day-to-day variation in activity and sleep"),
]

GOALS = [
    "increase daily steps without overtraining",
    "build a consistent walking habit",
    "reduce long sedentary periods",
    "recover after a low-activity day",
    "keep heart rate in a safe moderate zone",
    "prepare for better sleep tonight",
]

BAD_PATTERNS = [
    "generic",
    "unsafe_intense",
    "ignores_context",
    "overly_medical",
    "too_vague",
]


def build_prompt(rng: random.Random, idx: int) -> dict:
    cluster, cluster_desc = rng.choice(CLUSTERS)
    goal = rng.choice(GOALS)
    steps = rng.randint(1800, 9800)
    avg_hr = rng.randint(62, 88)
    resting_hr = rng.randint(54, 78)
    active_min = rng.randint(8, 72)
    sleep_h = round(rng.uniform(5.4, 8.3), 1)
    yesterday_steps = max(800, steps + rng.randint(-3200, 3200))
    sitting_blocks = rng.randint(2, 8)
    preferred_window = rng.choice(["morning", "lunch break", "late afternoon", "after dinner"])
    constraint = rng.choice(
        [
            "knee discomfort after long runs",
            "only has 20 minutes free",
            "rainy weather",
            "busy workday",
            "low motivation today",
            "wants a plan that can be done indoors",
        ]
    )
    return {
        "user_id": f"user_{idx % 24:02d}",
        "cluster": cluster,
        "cluster_desc": cluster_desc,
        "goal": goal,
        "steps": steps,
        "avg_hr": avg_hr,
        "resting_hr": resting_hr,
        "active_min": active_min,
        "sleep_h": sleep_h,
        "yesterday_steps": yesterday_steps,
        "sitting_blocks": sitting_blocks,
        "preferred_window": preferred_window,
        "constraint": constraint,
    }


def chosen_answer(ctx: dict) -> str:
    delta = ctx["steps"] - ctx["yesterday_steps"]
    trend = "above yesterday" if delta >= 0 else "below yesterday"
    step_gap = max(0, 7500 - ctx["steps"])
    walk_minutes = min(30, max(10, step_gap // 220))
    break_count = min(5, max(2, ctx["sitting_blocks"]))
    return (
        f" Choose a {walk_minutes}-minute {ctx['preferred_window']} walk and split it into "
        f"two easy blocks if needed. You are {abs(delta)} steps {trend}, with {ctx['active_min']} "
        f"active minutes and average heart rate {ctx['avg_hr']} bpm, so keep the effort conversational "
        f"rather than intense. Add {break_count} standing breaks of 3-5 minutes, then stop if "
        f"{ctx['constraint']} becomes noticeable. Tonight, review whether this moved you closer to "
        f"your goal: {ctx['goal']}."
    )


def rejected_answer(ctx: dict, rng: random.Random) -> str:
    pattern = rng.choice(BAD_PATTERNS)
    if pattern == "unsafe_intense":
        return (
            " Run as hard as possible for an hour today so you can quickly make up for missed "
            "activity. Ignore the heart-rate numbers and push through discomfort."
        )
    if pattern == "ignores_context":
        return (
            " Follow a generic gym plan with heavy lifting and no changes for your schedule, "
            "weather, current steps, or preferred activity window."
        )
    if pattern == "overly_medical":
        return (
            " Your readings suggest a serious condition. Change your medication plan immediately "
            "and avoid normal daily activity until you get specialist testing."
        )
    if pattern == "too_vague":
        return " Be healthier today. Try to move more, eat better, and stay positive."
    return (
        " You should exercise more because exercise is good for health. A normal person should "
        "always be active and avoid being sedentary."
    )


def survives_truncation(tokenizer, row: dict, seq_len: int) -> bool:
    prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
    chosen_ids = tokenizer.encode(row["chosen"], add_special_tokens=False)
    rejected_ids = tokenizer.encode(row["rejected"], add_special_tokens=False)
    if len(prompt_ids) >= seq_len:
        return False
    chosen_keep = min(len(chosen_ids), seq_len - len(prompt_ids))
    rejected_keep = min(len(rejected_ids), seq_len - len(prompt_ids))
    return chosen_keep > 8 and rejected_keep > 8


def make_row(ctx: dict, rng: random.Random) -> dict:
    prompt = (
        "Health coaching request:\n"
        f"- cluster: {ctx['cluster']} ({ctx['cluster_desc']})\n"
        f"- today_steps: {ctx['steps']}\n"
        f"- yesterday_steps: {ctx['yesterday_steps']}\n"
        f"- active_minutes: {ctx['active_min']}\n"
        f"- avg_hr: {ctx['avg_hr']} bpm\n"
        f"- resting_hr: {ctx['resting_hr']} bpm\n"
        f"- sleep_hours: {ctx['sleep_h']}\n"
        f"- sitting_blocks_over_45min: {ctx['sitting_blocks']}\n"
        f"- preferred_window: {ctx['preferred_window']}\n"
        f"- constraint: {ctx['constraint']}\n"
        f"User goal: {ctx['goal']}.\n"
        "Write a safe, specific, personalized next-step recommendation.\n"
        "Answer:"
    )
    return {
        "prompt": prompt,
        "chosen": chosen_answer(ctx),
        "rejected": rejected_answer(ctx, rng),
        "source": "synthetic_health_ab",
        "cluster": ctx["cluster"],
        "user_id": ctx["user_id"],
        "goal": ctx["goal"],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic A/B-style synthetic health preference data for local HFRL/DPO tests."
    )
    parser.add_argument("--tokenizer-dir", default="gpt2_small_lora_finetune/pretrained/gpt2")
    parser.add_argument("--train-output", default="review-stage/data/health_ab_pref_train.jsonl")
    parser.add_argument("--eval-output", default="review-stage/data/health_ab_pref_eval.jsonl")
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260608)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer_dir)
    tokenizer.pad_token = tokenizer.eos_token

    rows = []
    attempts = 0
    target = args.train_size + args.eval_size
    while len(rows) < target and attempts < target * 20:
        ctx = build_prompt(rng, attempts)
        row = make_row(ctx, rng)
        if survives_truncation(tokenizer, row, args.seq_len):
            rows.append(row)
        attempts += 1

    if len(rows) < target:
        raise SystemExit(f"Only built {len(rows)} usable rows out of requested {target}")

    train_rows = rows[: args.train_size]
    eval_rows = rows[args.train_size : args.train_size + args.eval_size]
    write_jsonl(Path(args.train_output), train_rows)
    write_jsonl(Path(args.eval_output), eval_rows)

    print(
        json.dumps(
            {
                "status": "ok",
                "seed": args.seed,
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
