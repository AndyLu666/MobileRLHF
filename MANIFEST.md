# Release Manifest

This source tree is prepared for the maintained MobileRLHF DPO/KTO phone
federation workflow.

Tracked public assets:

- Android app and SDK source under `android-visualizer/`
- Native MobileFineTuner source under `operator/`
- FedRL DPO/KTO JSONL assets under
  `android-visualizer/app/src/main/assets/mft_demo_data/`
- adb federation scripts under `scripts/android/`
- FedAvg scripts under `scripts/federated/`
- Public docs under `docs/`

Intentionally untracked local artifacts:

- `runs/`
- `review-stage/`
- Android/Gradle build outputs
- IDE metadata
- model weights and downloaded datasets

The archived local artifacts from this cleanup live outside the repository
under the Downloads directory.
