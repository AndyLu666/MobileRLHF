import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List


LETTERS = ("A", "B", "C", "D")


@dataclass
class McqExample:
    prompt: str
    chosen: str
    category: str


def format_prompt(question: str, options: List[str]) -> str:
    lines = [f"Question: {question}"]
    for letter, option in zip(LETTERS, options):
        lines.append(f"{letter}. {option}")
    lines.append("Answer:")
    return "\n".join(lines)


def arithmetic_examples() -> List[McqExample]:
    rows: List[McqExample] = []
    for a in range(12, 40):
        for b in range(3, 10):
            correct = a + b
            wrong = [correct - 1, correct + 1, correct + 2]
            options = [str(correct), str(wrong[0]), str(wrong[1]), str(wrong[2])]
            answer = "A"
            rows.append(
                McqExample(
                    prompt=format_prompt(f"What is {a} + {b}?", options),
                    chosen=f" {answer}",
                    category="arithmetic_add",
                )
            )
    for a in range(24, 60, 3):
        for b in range(2, 9):
            correct = a - b
            wrong = [correct + 1, correct - 1, correct + 2]
            options = [str(wrong[0]), str(correct), str(wrong[1]), str(wrong[2])]
            answer = "B"
            rows.append(
                McqExample(
                    prompt=format_prompt(f"What is {a} - {b}?", options),
                    chosen=f" {answer}",
                    category="arithmetic_sub",
                )
            )
    for a in range(3, 13):
        for b in range(4, 10):
            correct = a * b
            wrong = [correct - a, correct + b, correct + 1]
            options = [str(wrong[0]), str(wrong[1]), str(correct), str(wrong[2])]
            answer = "C"
            rows.append(
                McqExample(
                    prompt=format_prompt(f"What is {a} x {b}?", options),
                    chosen=f" {answer}",
                    category="arithmetic_mul",
                )
            )
    for quotient in range(3, 21):
        for divisor in range(2, 8):
            dividend = quotient * divisor
            correct = quotient
            wrong = [correct + 1, max(1, correct - 1), correct + 2]
            options = [str(correct), str(wrong[0]), str(wrong[1]), str(wrong[2])]
            rows.append(
                McqExample(
                    prompt=format_prompt(f"What is {dividend} / {divisor}?", options),
                    chosen=" A",
                    category="arithmetic_div",
                )
            )
    return rows


def unit_examples() -> List[McqExample]:
    rows: List[McqExample] = []
    month_days = [
        ("January", 31),
        ("April", 30),
        ("June", 30),
        ("September", 30),
        ("November", 30),
        ("February", 28),
    ]
    for month, days in month_days:
        options = [str(days + 1), str(days - 1), str(days + 2), str(days)]
        rows.append(
            McqExample(
                prompt=format_prompt(f"How many days are in {month} during a non-leap year?", options),
                chosen=" D",
                category="calendar_days",
            )
        )
    for hours in range(2, 26, 2):
        correct = hours * 60
        options = [str(correct), str(correct + 30), str(correct - 30), str(correct + 60)]
        rows.append(
            McqExample(
                prompt=format_prompt(f"How many minutes are in {hours} hours?", options),
                chosen=" A",
                category="time_minutes",
            )
        )
    for meters in range(3, 33, 3):
        correct = meters * 100
        options = [str(correct - 10), str(correct), str(correct + 10), str(correct + 100)]
        rows.append(
            McqExample(
                prompt=format_prompt(f"How many centimeters are in {meters} meters?", options),
                chosen=" B",
                category="metric_conversion",
            )
        )
    for kilograms in range(2, 22, 2):
        correct = kilograms * 1000
        options = [str(correct + 100), str(correct - 100), str(correct), str(correct + 1000)]
        rows.append(
            McqExample(
                prompt=format_prompt(f"How many grams are in {kilograms} kilograms?", options),
                chosen=" C",
                category="mass_conversion",
            )
        )
    return rows


def geography_examples() -> List[McqExample]:
    rows: List[McqExample] = []
    facts = [
        ("France", "Paris", ["Lyon", "Marseille", "Nice"], "A"),
        ("Japan", "Tokyo", ["Osaka", "Kyoto", "Nagoya"], "B"),
        ("Italy", "Rome", ["Milan", "Naples", "Turin"], "C"),
        ("Spain", "Madrid", ["Barcelona", "Seville", "Valencia"], "D"),
        ("Germany", "Berlin", ["Munich", "Hamburg", "Bonn"], "A"),
        ("Canada", "Ottawa", ["Toronto", "Vancouver", "Montreal"], "B"),
        ("Australia", "Canberra", ["Sydney", "Melbourne", "Perth"], "C"),
        ("Brazil", "Brasilia", ["Rio de Janeiro", "Sao Paulo", "Salvador"], "D"),
        ("India", "New Delhi", ["Mumbai", "Bengaluru", "Chennai"], "A"),
        ("Egypt", "Cairo", ["Alexandria", "Giza", "Luxor"], "B"),
    ]
    for country, capital, distractors, answer in facts:
        if answer == "A":
            options = [capital, distractors[0], distractors[1], distractors[2]]
        elif answer == "B":
            options = [distractors[0], capital, distractors[1], distractors[2]]
        elif answer == "C":
            options = [distractors[0], distractors[1], capital, distractors[2]]
        else:
            options = [distractors[0], distractors[1], distractors[2], capital]
        rows.append(
            McqExample(
                prompt=format_prompt(f"What is the capital of {country}?", options),
                chosen=f" {answer}",
                category="geography_capital",
            )
        )
    return rows


def science_examples() -> List[McqExample]:
    rows: List[McqExample] = []
    facts = [
        ("Which planet is known as the Red Planet?", ["Mars", "Venus", "Jupiter", "Mercury"], "A"),
        ("What gas do plants primarily absorb for photosynthesis?", ["Oxygen", "Carbon dioxide", "Nitrogen", "Hydrogen"], "B"),
        ("Which organ pumps blood through the human body?", ["Lung", "Liver", "Heart", "Kidney"], "C"),
        ("What force pulls objects toward Earth?", ["Magnetism", "Friction", "Electricity", "Gravity"], "D"),
        ("Which state of matter has a fixed shape and fixed volume?", ["Solid", "Liquid", "Gas", "Plasma"], "A"),
        ("What part of the cell contains genetic material?", ["Cytoplasm", "Nucleus", "Membrane", "Ribosome"], "B"),
        ("Which energy source comes from the sun?", ["Wind energy", "Hydroelectric energy", "Solar energy", "Geothermal energy"], "C"),
        ("What do bees help many plants do?", ["Sleep", "Migrate", "Hibernate", "Pollinate"], "D"),
    ]
    for question, options, answer in facts:
        rows.append(
            McqExample(
                prompt=format_prompt(question, options),
                chosen=f" {answer}",
                category="science_fact",
            )
        )
    return rows


def computing_examples() -> List[McqExample]:
    rows: List[McqExample] = []
    facts = [
        ("Which tool is primarily used to track code changes over time?", ["Git", "Docker", "Nginx", "SQLite"], "A"),
        ("What does CPU stand for?", ["Central Process Unit", "Central Processing Unit", "Compute Performance Utility", "Core Program Unit"], "B"),
        ("Which layer is responsible for routing packets between networks?", ["Physical layer", "Transport layer", "Network layer", "Session layer"], "C"),
        ("Which command is commonly used to list files in a Unix shell?", ["pwd", "cat", "cd", "ls"], "D"),
        ("What is the main purpose of unit tests?", ["Validate small pieces of code", "Deploy an application", "Encrypt network traffic", "Resize datasets"], "A"),
        ("Which file typically describes Python package dependencies?", ["README.md", "requirements.txt", "Dockerfile", "LICENSE"], "B"),
        ("What is a common benefit of version control branches?", ["They speed up CPUs", "They remove bugs automatically", "They isolate changes for safer work", "They compress images"], "C"),
        ("Which memory is usually fastest for a CPU core to access?", ["Hard disk", "Network storage", "Main memory", "Cache"], "D"),
    ]
    for question, options, answer in facts:
        rows.append(
            McqExample(
                prompt=format_prompt(question, options),
                chosen=f" {answer}",
                category="computing_fact",
            )
        )
    return rows


def language_examples() -> List[McqExample]:
    rows: List[McqExample] = []
    facts = [
        ("Which word is closest in meaning to 'brief'?", ["short", "heavy", "noisy", "late"], "A"),
        ("Which sentence uses correct punctuation?", ["Lets eat, grandma.", "Let's eat, grandma.", "Lets eat grandma.", "Let's eat grandma."], "B"),
        ("Which word is a noun?", ["run", "blue", "library", "quickly"], "C"),
        ("Which option is the best summary of a paragraph?", ["A random detail", "A repeated typo", "A character name only", "The main idea in fewer words"], "D"),
    ]
    for question, options, answer in facts:
        rows.append(
            McqExample(
                prompt=format_prompt(question, options),
                chosen=f" {answer}",
                category="language_fact",
            )
        )
    return rows


def logic_examples() -> List[McqExample]:
    rows: List[McqExample] = []
    for start in range(2, 22):
        seq = [start, start + 2, start + 4]
        correct = start + 6
        options = [str(correct), str(correct + 1), str(correct - 1), str(correct + 2)]
        rows.append(
            McqExample(
                prompt=format_prompt(
                    f"Which number best completes the pattern {seq[0]}, {seq[1]}, {seq[2]}, ?",
                    options,
                ),
                chosen=" A",
                category="logic_sequence_even",
            )
        )
    for start in range(3, 18):
        seq = [start, start * 2, start * 4]
        correct = start * 8
        options = [str(correct + start), str(correct), str(correct - start), str(correct + 2 * start)]
        rows.append(
            McqExample(
                prompt=format_prompt(
                    f"Which number best completes the pattern {seq[0]}, {seq[1]}, {seq[2]}, ?",
                    options,
                ),
                chosen=" B",
                category="logic_sequence_double",
            )
        )
    return rows


def fraction_examples() -> List[McqExample]:
    rows: List[McqExample] = []
    for numerator in range(1, 11):
        denominator = numerator + 2
        question = f"Which fraction is equivalent to {numerator}/{denominator}?"
        correct = f"{numerator * 2}/{denominator * 2}"
        options = [
            f"{numerator + 1}/{denominator * 2}",
            correct,
            f"{numerator * 2}/{denominator * 2 + 1}",
            f"{numerator + 2}/{denominator + 2}",
        ]
        rows.append(
            McqExample(
                prompt=format_prompt(question, options),
                chosen=" B",
                category="fraction_equivalent",
            )
        )
    for base in range(10, 31):
        percent = 10
        correct = base // 10
        question = f"What is {percent}% of {base}?"
        options = [str(correct), str(correct + 1), str(max(0, correct - 1)), str(correct + 2)]
        rows.append(
            McqExample(
                prompt=format_prompt(question, options),
                chosen=" A",
                category="percent_ten",
            )
        )
    for base in range(12, 33):
        question = f"What is half of {base}?"
        correct = base / 2
        options = [str(int(correct + 1)), str(int(correct - 1)), str(correct), str(int(correct + 2))]
        rows.append(
            McqExample(
                prompt=format_prompt(question, options),
                chosen=" C",
                category="fraction_half",
            )
        )
    return rows


def build_examples() -> List[McqExample]:
    rows = []
    rows.extend(arithmetic_examples())
    rows.extend(unit_examples())
    rows.extend(geography_examples())
    rows.extend(science_examples())
    rows.extend(computing_examples())
    rows.extend(language_examples())
    rows.extend(logic_examples())
    rows.extend(fraction_examples())
    return rows


def write_jsonl(path: Path, rows: List[McqExample]) -> None:
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(
                json.dumps(
                    {"prompt": row.prompt, "chosen": row.chosen, "category": row.category},
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic MMLU-style chosen-only SFT data")
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--train-size", type=int, default=480)
    parser.add_argument("--eval-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rows = build_examples()
    random.Random(args.seed).shuffle(rows)
    needed = args.train_size + args.eval_size
    if len(rows) < needed:
        raise ValueError(f"Requested {needed} examples, but only {len(rows)} are available")

    train_rows = rows[: args.train_size]
    eval_rows = rows[args.train_size : needed]
    write_jsonl(Path(args.train_output), train_rows)
    write_jsonl(Path(args.eval_output), eval_rows)

    print(
        json.dumps(
            {
                "status": "ok",
                "train_size": len(train_rows),
                "eval_size": len(eval_rows),
                "categories": sorted({row.category for row in rows}),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
