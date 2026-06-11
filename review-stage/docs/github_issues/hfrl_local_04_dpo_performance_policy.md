## Goal

Use paired chosen/rejected forwards and cached reference logps where they improve runtime without changing DPO semantics.

## Context

Existing measurements:

- 4-forward baseline, 4 steps: `real=13.08s`, `loss=0.696726`, `reward_margin=-0.00714569`, `nonzero_grad=24`
- paired 2-forward, 4 steps: `real=10.76s`, `loss=0.696727`, `reward_margin=-0.00714724`, `nonzero_grad=24`
- cached reference, 20 steps: paired `real=54.04s`, cached `real=56.68s`
- cached reference, 50 steps: paired `real=131.45s`, cached `real=105.21s`

## Acceptance Criteria

- Keep paired 2-forward mode enabled for local HFRL runs.
- Keep cached reference logps opt-in for longer static-feedback runs, because cache precompute can dominate when the train/eval set is larger.
- Include loss/reward/gradient/RSS metrics in performance reports, not only wall time.
- Preserve non-cached smoke mode for quick correctness checks.

## Local Result

The 220-step health A/B run completed with cached reference logps, but the reference-cache precompute cost was too high for this generated 512/128 split:

- `ref_cache_seconds=1036.218`
- `train_loop_seconds=801.1588`

Decision: `DPO_PAIR_FORWARD=1` remains the default. `CACHE_REF_LOGPS=0` is the baseline default; use `CACHE_REF_LOGPS=1` only after a short amortization check or when reusing a static feedback set across longer runs.

## Device Dependency

No Android phone required.
