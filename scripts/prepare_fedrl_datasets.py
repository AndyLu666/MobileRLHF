#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ANDROID_DATA = ROOT / "android-visualizer" / "app" / "src" / "main" / "assets" / "mft_demo_data"
CONVERTER = ROOT / "scripts" / "convert_dpo_to_kto_jsonl.py"


DATASETS = [
    {
        "id": "fedrl_health",
        "display_name": "Curated health preference dataset",
        "train_source": ANDROID_DATA / "fedrl_health_dpo_train.jsonl",
        "eval_source": ANDROID_DATA / "fedrl_health_dpo_eval.jsonl",
        "formal": True,
    },
    {
        "id": "fedrl_hf_fitness",
        "display_name": "HF fitness preference fallback dataset",
        "train_source": ANDROID_DATA / "hf_fitness_pref_train.jsonl",
        "eval_source": ANDROID_DATA / "hf_fitness_pref_eval.jsonl",
        "formal": False,
    },
]


def count_jsonl(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def read_first(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"{path} is empty")


def validate_dpo(path: Path) -> None:
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = [key for key in ("prompt", "chosen", "rejected") if not str(row.get(key, "")).strip()]
        if missing:
            raise ValueError(f"{path}:{line_no}: missing DPO fields: {missing}")


def validate_kto(path: Path) -> None:
    labels = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = [key for key in ("prompt", "response", "label") if key not in row]
        if missing:
            raise ValueError(f"{path}:{line_no}: missing KTO fields: {missing}")
        if not isinstance(row["label"], bool):
            raise ValueError(f"{path}:{line_no}: label must be boolean")
        labels.append(row["label"])
    if labels and (all(labels) or not any(labels)):
        raise ValueError(f"{path}: KTO labels must include both true and false")


def convert_to_kto(dpo_path: Path, kto_path: Path, stats_path: Path) -> dict[str, Any]:
    subprocess.run(
        [
            sys.executable,
            str(CONVERTER),
            "--input",
            str(dpo_path),
            "--output",
            str(kto_path),
            "--stats",
            str(stats_path),
        ],
        check=True,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(stats_path.read_text(encoding="utf-8"))


def copy_file(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def prepare_dataset(spec: dict[str, Any]) -> dict[str, Any]:
    dataset_id = spec["id"]
    train_source = Path(spec["train_source"])
    eval_source = Path(spec["eval_source"])
    if not train_source.exists():
        raise FileNotFoundError(train_source)
    if not eval_source.exists():
        raise FileNotFoundError(eval_source)

    dpo_train = ANDROID_DATA / f"{dataset_id}_dpo_train.jsonl"
    dpo_eval = ANDROID_DATA / f"{dataset_id}_dpo_eval.jsonl"
    kto_train = ANDROID_DATA / f"{dataset_id}_kto_train.jsonl"
    kto_eval = ANDROID_DATA / f"{dataset_id}_kto_eval.jsonl"

    copy_file(train_source, dpo_train)
    copy_file(eval_source, dpo_eval)
    validate_dpo(dpo_train)
    validate_dpo(dpo_eval)
    with tempfile.TemporaryDirectory(prefix="fedrl_kto_stats_") as tmp:
        tmp_dir = Path(tmp)
        train_convert = convert_to_kto(dpo_train, kto_train, tmp_dir / f"{dataset_id}_train_stats.json")
        eval_convert = convert_to_kto(dpo_eval, kto_eval, tmp_dir / f"{dataset_id}_eval_stats.json")
    validate_kto(kto_train)
    validate_kto(kto_eval)

    android_outputs = [
        str(path.relative_to(ROOT))
        for path in (dpo_train, dpo_eval, kto_train, kto_eval)
    ]

    return {
        "id": dataset_id,
        "display_name": spec["display_name"],
        "formal": bool(spec["formal"]),
        "dpo": {
            "schema": "jsonl: prompt/chosen/rejected",
            "train": str(dpo_train.relative_to(ROOT)),
            "eval": str(dpo_eval.relative_to(ROOT)),
            "train_rows": count_jsonl(dpo_train),
            "eval_rows": count_jsonl(dpo_eval),
            "sample_keys": sorted(read_first(dpo_train).keys()),
        },
        "kto": {
            "schema": "jsonl: prompt/response/label",
            "train": str(kto_train.relative_to(ROOT)),
            "eval": str(kto_eval.relative_to(ROOT)),
            "train_rows": count_jsonl(kto_train),
            "eval_rows": count_jsonl(kto_eval),
            "android_runtime_note": (
                "Reference asset for validation and Python smoke runs. "
                "The Android runtime accepts binary KTO rows directly and "
                "explicitly unpairs DPO rows before canonical KTO batching."
            ),
            "train_conversion": train_convert,
            "eval_conversion": eval_convert,
            "sample_keys": sorted(read_first(kto_train).keys()),
        },
        "android_assets": android_outputs,
    }


def main() -> int:
    manifest = {
        "format": "fedrl_preference_dataset_manifest.v1",
        "primary_dataset": "fedrl_health",
        "fallback_dataset": "fedrl_hf_fitness",
        "datasets": [prepare_dataset(spec) for spec in DATASETS],
        "phone_feedback_dataset": {
            "id": "phone_qwen_feedback",
            "schema": "mixed jsonl: prompt/chosen/rejected or prompt/response/label",
            "path_on_phone": "files/mft_demo_data/inference_feedback.jsonl",
            "note": (
                "Generated by the Android app after local A/B or thumbs feedback; "
                "raw rows stay on device."
            ),
        },
    }
    manifest_path = ANDROID_DATA / "fedrl_preference_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
