#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FORMAT = "mobile_finetuner.lora_adapter_jsonl.v1"


def read_adapter(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    header: dict[str, Any] | None = None
    tensors: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "format" in row:
                header = row
                if row["format"] != FORMAT:
                    raise ValueError(f"{path}: unsupported adapter format {row['format']!r}")
                continue
            name = row.get("name")
            shape = row.get("shape")
            data = row.get("data")
            if not isinstance(name, str) or not isinstance(shape, list) or not isinstance(data, list):
                raise ValueError(f"{path}:{line_no}: malformed tensor row")
            if name in tensors:
                raise ValueError(f"{path}:{line_no}: duplicate tensor {name}")
            tensors[name] = {"shape": shape, "data": [float(x) for x in data]}
    if header is None:
        raise ValueError(f"{path}: missing adapter header")
    if not tensors:
        raise ValueError(f"{path}: no tensors found")
    return header, tensors


def report_weight(adapter: Path, explicit_weight: float | None, weight_key: str) -> float:
    if explicit_weight is not None:
        if explicit_weight <= 0:
            raise ValueError("explicit weights must be positive")
        return explicit_weight
    report = adapter.with_name("report.json")
    if report.exists():
        row = json.loads(report.read_text(encoding="utf-8"))
        if weight_key == "auto":
            objective = str(row.get("objective", "")).lower()
            keys = (
                ["binary_feedback_rows", "train_samples", "feedback_pairs"]
                if objective == "kto"
                else ["feedback_pairs", "train_samples", "binary_feedback_rows"]
            )
        else:
            keys = [weight_key]
        for key in keys:
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                return float(value)
    return 1.0


def parse_input(raw: str) -> tuple[Path, float | None]:
    if "::" not in raw:
        return Path(raw), None
    path, weight = raw.rsplit("::", 1)
    return Path(path), float(weight)


def aggregate(inputs: list[tuple[Path, float]], output: Path, manifest: Path) -> None:
    loaded = []
    for adapter, weight in inputs:
        _, tensors = read_adapter(adapter)
        loaded.append((adapter, weight, tensors))

    reference_names = list(loaded[0][2].keys())
    reference_shapes = {name: loaded[0][2][name]["shape"] for name in reference_names}
    for adapter, _, tensors in loaded[1:]:
        if list(tensors.keys()) != reference_names:
            raise ValueError(f"{adapter}: tensor names/order differ from {loaded[0][0]}")
        for name in reference_names:
            if tensors[name]["shape"] != reference_shapes[name]:
                raise ValueError(f"{adapter}: shape mismatch for {name}")

    total_weight = sum(weight for _, weight, _ in loaded)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as out:
        out.write(json.dumps({"format": FORMAT, "aggregation": "fedavg_jsonl_v1"}, separators=(",", ":")) + "\n")
        for name in reference_names:
            size = len(loaded[0][2][name]["data"])
            acc = [0.0] * size
            for _, weight, tensors in loaded:
                data = tensors[name]["data"]
                if len(data) != size:
                    raise ValueError(f"data length mismatch for {name}")
                scale = weight / total_weight
                for i, value in enumerate(data):
                    acc[i] += scale * value
            out.write(json.dumps(
                {"name": name, "shape": reference_shapes[name], "data": acc},
                separators=(",", ":"),
            ) + "\n")

    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "output": str(output),
        "aggregation": "fedavg_jsonl_v1",
        "weight_sum": total_weight,
        "clients": [
            {"adapter": str(adapter), "weight": weight}
            for adapter, weight, _ in loaded
        ],
        "tensor_count": len(reference_names),
        "weight_rule": "explicit adapter::weight or report.json weight key",
    }, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="FedAvg MobileFineTuner JSONL LoRA adapters.")
    parser.add_argument("--input", action="append", required=True,
                        help="Adapter path, optionally suffixed as path::weight.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--weight-key",
        default="auto",
        help=(
            "report.json numeric field used when no explicit weight is supplied; "
            "'auto' uses binary_feedback_rows for KTO and feedback_pairs for DPO"
        ),
    )
    args = parser.parse_args()

    inputs = []
    for raw in args.input:
        path, explicit = parse_input(raw)
        inputs.append((path, report_weight(path, explicit, args.weight_key)))
    manifest = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    aggregate(inputs, args.output, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
