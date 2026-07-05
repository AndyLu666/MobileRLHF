# Paper Demo Console Design

## Goal

Build the Android visualizer's main screen as a paper-ready KTO demo console: it should explain the local feedback-to-training workflow in one screenshot while still letting the developer quickly verify whether the Android run worked.

## Target Audience

- MobileECom WIP paper reviewers: need a clear visual story, not raw engineering clutter.
- Project developers: need fast access to run status, training parameters, KTO trend metrics, logs, and artifact paths.

## Main Product Shape

The app opens directly into a focused `Demo` screen. The screen presents the full workflow as:

1. `Generate`: Qwen3-0.6B produces A/B health suggestions locally or through the configured bridge.
2. `Prefer`: the user chooses A or B, producing preference pairs.
3. `Train KTO`: Android native C++ KTO training runs on the saved preference dataset.
4. `Evidence`: the app surfaces the run artifacts and trend metrics.

Existing detailed views remain available, but the primary first impression is the KTO workflow, not the old multi-chart dashboard.

## Layout

### Top Bar

- Title: `MobileFineTuner`
- Subtitle: `On-device preference tuning with Qwen3-0.6B`
- Right action: refresh icon
- Visual style: flat, restrained, no decorative blobs or gradient ornaments.

### Workflow Strip

A compact horizontal stepper shows:

- `Generate`
- `Prefer`
- `Train`
- `Evidence`

Each step has a small status label: `Ready`, `Saved`, `Running`, `Done`, or `Needs input`.

### Primary Demo Panel

This is the main interaction area.

- Prompt input
- Model path / bridge selector
- `Generate A/B` button
- Candidate A and Candidate B panels
- A/B preference buttons
- saved pair count

The A/B panels use simple labels and consistent spacing. They should be readable in screenshots without requiring zoom.

### KTO Run Panel

This panel controls and summarizes Android training.

- Dataset source: `Qwen A/B feedback`
- Pair count
- Training config: `steps`, `batch`, `seq_len`, `rank`, `alpha`, `lr`
- `Train KTO` button
- Stop button
- Current run name and status

The run panel should avoid hidden required controls. The default demo path should be obvious and runnable without tuning.

### Evidence Panel

This panel is the paper-facing result summary.

Required fields:

- `Objective`: `KTO`
- `Model`: `Qwen3-0.6B`
- `Pairs`: latest pair count
- `Steps`: latest max steps
- `Batch`: latest batch size
- `Loss`: latest loss
- `Reward Accuracy`: latest reward accuracy
- `Reward Margin`: latest reward margin
- `Trend`: `reward_accuracy` and `reward_margin` before to after, when `report.json` provides it
- `Artifacts`: short paths for `report.json`, `metrics.ndjson`, and `before_after_eval.json`

Metric wording must not overclaim. The UI should say `training trend` rather than `quality improvement` for smoke runs.

### Details Access

The old Overview, Training, Comparison, and Logs views should remain available behind a `Details` tab or secondary section. This keeps developer visibility without overwhelming the paper screenshot.

## Visual Direction

- Use a clean light UI with white surfaces, subtle gray dividers, and small accent colors.
- Avoid large decorative gradients, orbs, oversized hero typography, and nested cards.
- Use cards only for repeated or clearly framed items: A/B candidates, KTO run status, evidence summary.
- Use compact typography so text fits on a phone screenshot.
- Prefer icons for actions where possible: refresh, run, stop, folder/details.
- Keep border radius at 8dp or less for cards and controls unless existing Material components require otherwise.

## Data Flow

1. `NativeInferenceState` provides prompt, model/bridge settings, generated candidates, saved pair count, and feedback path.
2. A/B preference selection appends to `inference_feedback.jsonl`.
3. `NativeDemoRunnerState` provides current KTO run status and training source.
4. The latest selected `RunSnapshot` provides live metrics from `metrics.ndjson`.
5. `report.json` / `before_after_eval.json`, when available, provide trend fields for the Evidence panel.

If report parsing is not currently available in model state, the first implementation should surface available `RunSnapshot` metrics and artifact paths, then add report-derived trend fields as a focused follow-up.

## Error Handling

- If no feedback pairs exist, show `Generate and choose A/B before training`.
- If training is running, disable conflicting controls and show the active run name.
- If the model path is missing, show a concise inline error next to the model selector.
- If report artifacts are missing, show `Waiting for report.json` instead of blank metrics.
- If a smoke run trend exists, label it as `training trend`.

## Testing

Use TDD for behavior helpers and UI-state formatting where possible.

Required checks:

- Unit test for workflow step status mapping.
- Unit test for demo metric formatting so missing values render as `N/A`.
- Gradle unit tests pass.
- Android debug APK builds.
- Manual/emulator smoke check: Demo screen renders, A/B generation controls are visible, `Train KTO` remains reachable, and evidence fields fit without overlap.

## Out of Scope

- New training algorithm work.
- New native C++ trainer changes.
- Claims of real quality improvement from 5-10 step smoke runs.
- Full paper writing or figure generation.

