## Goal

Move from local simulated feedback to real interactive Android feedback.

## Acceptance Criteria

- User selects A/B responses repeatedly on Android.
- App appends DPO-compatible preference JSONL.
- Native trainer is launched on device or through an adb-shell binary path.
- Visualizer reads the resulting HFRL metrics.

## Current Status

Deferred for the local-only batch. This requires Android phone access and user feedback, so it should be tracked separately from local DPO/data/performance work.

## Related Issues

- Existing issue #10: DPO multi-user preference collection and aggregation flow
- Existing issue #11: C++ DPO native mobile-side trainer
- Existing issue #16: C++ DPO validation

## Device Dependency

Android phone required.
