# Feedback Mode and Demo UI Refactor Design

Date: 2026-07-05
Repo: MobileFineTuner
Target: Android visualizer demo app

## Goal

Refactor the Android demo app into a clear guided workflow that supports both
A/B preference feedback and single-answer thumbs up/down feedback, then makes
KTO training progress and effect visible while training is running.

The app should be usable for a paper demo without explaining hidden tabs or raw
logs. The first screen should answer four questions:

1. What prompt am I testing?
2. Which feedback mode am I using?
3. Can I start KTO training from collected feedback?
4. Is training moving in the right direction?

## Product Shape

Use a single primary workflow instead of many equally prominent panels:

1. Generate
2. Feedback
3. Train
4. Track

The main demo screen exposes only the common path. Advanced settings stay
available but move into a collapsed section.

Primary tabs become:

- Demo: generate, collect feedback, start training, and view live progress.
- Progress: detailed charts and run artifacts.
- Artifacts: files, JSONL paths, reports, logs, and export evidence.

The existing low-level metrics and logs remain accessible, but they are no
longer the first thing a demo user sees.

## Feedback Modes

The app supports two collection modes from the same prompt box.

### A/B Compare

The model generates two candidates:

- Candidate A
- Candidate B

The user selects `Prefer A` or `Prefer B`. The app saves a pair feedback record.

This keeps compatibility with the current pairwise flow and existing DPO-style
datasets.

### Single Rating

The model generates one candidate response. The user selects:

- Thumbs up: desirable response
- Thumbs down: undesirable response

The app saves a single KTO-style label record. It does not fabricate a rejected
answer.

## Feedback Data Schema

Feedback JSONL supports three shapes:

### Legacy Pair

Existing records with `prompt`, `chosen`, and `rejected` continue to work.

### Typed Pair

```json
{
  "type": "pair",
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "chosen_label": "A",
  "source": "android_qwen_inference_feedback"
}
```

### Typed Single

```json
{
  "type": "single",
  "prompt": "...",
  "response": "...",
  "label": true,
  "source": "android_qwen_inference_feedback"
}
```

For KTO training, pair records are converted internally:

- chosen -> label true
- rejected -> label false

Single records are used directly:

- thumbs up -> label true
- thumbs down -> label false

## Training UI

The training area should show live evidence while the run is active.

Visible during training:

- step progress, for example `3 / 10`
- current phase: preparing data, training, saving adapter, complete, failed
- reward accuracy chart
- reward margin chart
- loss chart
- peak memory / RSS value
- last status message

Reward accuracy and reward margin are the primary signals. Loss is shown, but
not presented as the main evidence that the model is better.

## Effect UI

After training, the app shows a compact before/after summary:

- feedback samples used
- final reward accuracy
- final reward margin
- loss trend
- peak memory
- adapter output path

If before/after evaluation exists, show the improvement summary. If it does not
exist, clearly label the run as a smoke validation rather than a quality claim.

## Code Boundaries

Target files:

- `TrainingModels.kt`: add feedback mode and single-rating state.
- `FeedbackJsonl.kt`: encode and parse pair and single feedback records.
- `ExperimentViewModel.kt`: add single thumbs feedback handling and mixed
  pair/single KTO ingestion.
- `DashboardScreen.kt`: restructure the paper/demo console around the guided
  workflow.
- `PaperDemoConsoleState.kt`: keep UI state derivation out of the composable.
- UI tests and JVM tests for record encoding, state derivation, and parser
  compatibility.

The refactor should keep existing native KTO/DPO training commands intact. It
changes Android orchestration and feedback formatting, not the core native
optimizer in this pass.

## Error Handling

The UI must make common failure states obvious:

- no model loaded
- inference failed
- no feedback samples available
- training failed
- metrics file missing
- run completed but no before/after eval exists

Each failure state should show a short action-oriented message.

## Testing

Required checks:

- `FeedbackJsonl` encodes typed pair records.
- `FeedbackJsonl` encodes typed single records.
- Legacy pair records still parse.
- Pair records convert into two KTO rows.
- Single records convert into one KTO row.
- UI state reports correct workflow step and progress.
- Android JVM tests pass.
- Android debug build succeeds.

Preferred manual check:

- install debug APK through adb
- verify Demo screen has mode switch, prompt box, feedback buttons, train
  button, and live chart area
- run a short 5-step or 10-step KTO smoke demo and verify metrics are visible

## Non-goals

- Do not claim the model quality improves from a 5-10 step smoke run alone.
- Do not remove legacy DPO pair support.
- Do not redesign the native C++ trainer in this pass.
- Do not make hidden raw logs the primary demo surface.

