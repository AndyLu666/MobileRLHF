#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def serial_key(serial: str) -> str:
    return serial.replace(":", "_").replace("/", "_")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def last_metric(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    last = ""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = line
    return json.loads(last) if last else {}


def monitor_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "monitor_samples": 0,
            "monitor_first_utc": "",
            "monitor_last_utc": "",
            "monitor_elapsed_s": "",
            "monitor_max_vmrss_kb": "",
            "monitor_max_vmhwm_kb": "",
        }
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    if not rows:
        return {
            "monitor_samples": 0,
            "monitor_first_utc": "",
            "monitor_last_utc": "",
            "monitor_elapsed_s": "",
            "monitor_max_vmrss_kb": "",
            "monitor_max_vmhwm_kb": "",
        }

    def max_int(field: str) -> int | str:
        values = []
        for row in rows:
            raw = row.get(field, "")
            if raw:
                try:
                    values.append(int(raw))
                except ValueError:
                    pass
        return max(values) if values else ""

    return {
        "monitor_samples": len(rows),
        "monitor_first_utc": rows[0].get("timestamp_utc", ""),
        "monitor_last_utc": rows[-1].get("timestamp_utc", ""),
        "monitor_elapsed_s": rows[-1].get("elapsed_s", ""),
        "monitor_max_vmrss_kb": max_int("vmrss_kb"),
        "monitor_max_vmhwm_kb": max_int("vmhwm_kb"),
    }


def summarize(round_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report_path in sorted(round_dir.glob("*/*/report.json")):
        run_dir = report_path.parent
        serial = run_dir.parent.name
        report = load_json(report_path)
        metric_path = run_dir / "metrics.ndjson"
        metric = last_metric(metric_path)
        monitor_path = round_dir / "monitor" / f"{serial_key(serial)}.csv"
        row: dict[str, Any] = {
            "serial": serial,
            "run_name": run_dir.name,
            "status": report.get("status", ""),
            "objective": report.get("objective", ""),
            "feedback_pairs": report.get("feedback_pairs", ""),
            "binary_feedback_rows": report.get("binary_feedback_rows", ""),
            "max_steps": report.get("max_steps", ""),
            "batch_size": report.get("batch_size", ""),
            "lora_rank": report.get("lora_rank", ""),
            "lora_alpha": report.get("lora_alpha", ""),
            "cached_reference_pairs": report.get("cached_reference_pairs", ""),
            "cached_reference_batches": report.get("cached_reference_batches", ""),
            "total_elapsed_ms": report.get("total_elapsed_ms", ""),
            "max_train_rss_mb": report.get("max_train_rss_mb", ""),
            "metrics_rows": count_lines(metric_path),
            "last_step": metric.get("step", ""),
            "last_loss": metric.get("loss", ""),
            "last_reward_accuracy": metric.get("reward_accuracy", ""),
            "last_reward_margin": metric.get("reward_margin", ""),
            "last_kl_estimate": metric.get("kl_estimate", ""),
        }
        row.update(monitor_summary(monitor_path))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a collected FedRL phone round.")
    parser.add_argument("--round-dir", required=True, type=Path)
    args = parser.parse_args()
    rows = summarize(args.round_dir)
    write_csv(args.round_dir / "device_summary.csv", rows)
    (args.round_dir / "summary.json").write_text(
        json.dumps({"round_dir": str(args.round_dir), "devices": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
