# KTO Structure and Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the MobileFineTuner repository easier to navigate and promote the current KTO smoke path from scattered files into a documented, repeatable workflow.

**Architecture:** Keep core code paths stable and avoid large directory moves. Add a top-level directory map, document the PyTorch alignment/KTO area, and provide a maintained one-command smoke runner that converts curated DPO pairs into KTO binary samples and runs both target smoke configurations.

**Tech Stack:** Markdown docs, Python 3, shell scripts, existing `pt_env`, PyTorch/Transformers/PEFT tiny GPT-2 smoke.

## Global Constraints

- Do not modify unrelated Android working-tree changes.
- Keep `operator/`, `examples/`, and `android-visualizer/` in place.
- Keep generated run outputs under ignored `runs/`.
- Use `PYTHONPYCACHEPREFIX=/Users/anakinju/DKU/MobileFineTuner/.pycache` for Python verification on this machine.

---

### Task 1: Repository Navigation

**Files:**
- Create: `PROJECT_STRUCTURE.md`
- Modify: `README.md`

**Interfaces:**
- Produces: A top-level map explaining what each folder is for.

- [ ] Add `PROJECT_STRUCTURE.md` with current directory responsibilities.
- [ ] Link it from `README.md`.
- [ ] Verify links and Markdown readability.

### Task 2: PyTorch Alignment Area

**Files:**
- Create: `pytorch_alignment/README.md`
- Modify: `scripts/README.md`

**Interfaces:**
- Produces: Clear ownership for `pytorch_alignment/dpo_utils.py`, KTO data conversion, and smoke validation.

- [ ] Document DPO/KTO utilities and expected data formats.
- [ ] Document which scripts are stable and which are review-stage experiments.
- [ ] Keep file paths compatible with current tests.

### Task 3: KTO One-Command Workflow

**Files:**
- Create: `scripts/run_kto_health_smoke.sh`
- Modify: `scripts/README.md`

**Interfaces:**
- Consumes: `scripts/convert_dpo_to_kto_jsonl.py`, `review-stage/scripts/kto_sanity_check.py`, `review-stage/data/health_dpo_curated_train.jsonl`, `review-stage/data/health_dpo_curated_eval.jsonl`.
- Produces: `review-stage/data/health_kto_train.jsonl`, `review-stage/data/health_kto_eval.jsonl`, `runs/kto_health_smoke_b4_s5_seq192/`, `runs/kto_health_smoke_b2_s10_seq192/`.

- [ ] Add a shell wrapper that regenerates KTO data and runs both target smoke configs.
- [ ] Verify it exits nonzero on failure.
- [ ] Run the wrapper successfully.

### Task 4: Verification

**Files:**
- Verify: `.gitignore`, `pytorch_alignment/dpo_utils.py`, `scripts/*.py`, `review-stage/scripts/kto_sanity_check.py`, docs.

**Interfaces:**
- Produces: Fresh test/build evidence.

- [ ] Run `tests/test_dpo_utils.py`.
- [ ] Run `py_compile`.
- [ ] Run `bash -n scripts/run_kto_health_smoke.sh`.
- [ ] Run `git diff --check`.
