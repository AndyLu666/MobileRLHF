# Project Structure

This repository contains the native MobileFineTuner core, model-specific
training examples, Android integration, and research-stage alignment workflows.
Use this map to decide where new work should live.

## Top-Level Directories

| Path | Purpose | Put new work here when... |
| --- | --- | --- |
| `operator/` | Reusable C++ training core: tensor/autograd, model graphs, tokenizer adapters, LoRA, optimizers, datasets, and CMake package export. | You are changing native training behavior or the public C++/JNI training surface. |
| `examples/` | Model-specific native CLIs and scripts for GPT-2, Gemma, and Qwen. | You need a runnable native experiment for one model family. |
| `android-visualizer/` | Android app, SDK/AAR module, sample app, Gradle config, and Android QA scripts. | You are changing phone UI, Android packaging, or Java/Kotlin/JNI app integration. |
| `pytorch_alignment/` | Python reference utilities for alignment data, DPO/KTO loss behavior, and small PyTorch checks. | You are prototyping or testing alignment math before native/Android integration. |
| `scripts/` | Maintained repo-level automation. These scripts should be stable enough to document and re-run. | You are adding a reusable command that other developers should call directly. |
| `review-stage/` | Research-stage notes, small curated datasets, issue drafts, experiment scripts, and one-off validation harnesses. | You need an exploratory artifact that is useful to track but not yet a stable public entrypoint. |
| `docs/` | Stable architecture, API, Android, model asset, and workflow documentation. | The information should survive beyond a single experiment. |
| `tests/` | Python unit tests for maintained helper code. | You are changing Python behavior in `pytorch_alignment/` or `scripts/`. |
| `figures/` | Source diagrams and generated figure descriptions. | You are creating paper or README diagrams. |
| `briefing/` and `.briefing/` | Meeting/reporting artifacts and dated summaries. | You are preparing a presentation, checkpoint report, or handoff summary. |
| `runs/` | Local experiment outputs: metrics, logs, adapters, plots, and generated reports. Ignored by Git. | You are producing run artifacts that should not be committed. |
| `data/` | Local external datasets. Ignored by Git. | You are staging downloaded or generated datasets that are too large or too local for source control. |
| `third_party/` | External native dependencies used by the build. | You are updating vendored native dependencies or platform libraries. |

## Current Alignment Workflow

The active DPO-to-KTO work uses a two-level split:

- Stable utilities live in `pytorch_alignment/` and `scripts/`.
- Curated research data and smoke runners live in `review-stage/`.

The current files are:

```text
pytorch_alignment/dpo_utils.py
pytorch_alignment/kto_trainer.py
pytorch_alignment/kto_cli.py
scripts/convert_dpo_to_kto_jsonl.py
scripts/run_kto_health_smoke.sh
review-stage/data/health_dpo_curated_30.jsonl
review-stage/data/health_dpo_curated_train.jsonl
review-stage/data/health_dpo_curated_eval.jsonl
review-stage/data/health_kto_train.jsonl
review-stage/data/health_kto_eval.jsonl
review-stage/scripts/kto_sanity_check.py
```

For the 20-30 sample Qwen3-0.6B KTO smoke target, run:

```bash
bash scripts/run_kto_health_smoke.sh
```

This regenerates KTO binary JSONL from the curated DPO pairs and runs both
target Qwen3-0.6B configurations:

- batch size 4, 5 steps
- batch size 2, 10 steps

## Where Not To Put Things

- Do not put generated metrics, adapters, or logs in `scripts/`, `docs/`, or
  `review-stage/data/`; use `runs/`.
- Do not put stable reusable scripts under `review-stage/scripts/`; promote
  them to `scripts/` once they become maintained entrypoints.
- Do not put model weights or downloaded benchmark datasets in Git-tracked
  directories.
- Do not change `operator/` for alignment-method experiments until the PyTorch
  smoke path has proven the loss and backprop behavior.
