# Trainer-Level DPO vs SFT

## Purpose
- Run a matched GPT-2 LoRA comparison between:
- DPO on preference pairs
- chosen-only SFT derived from the same preference train set

## Files
- `run_gpt2_dpo_vs_sft.sh`: end-to-end harness
- `preference_to_masked_jsonl.py`: converts preference JSONL to chosen-only masked JSONL for SFT
- `eval_preference_model.py`: evaluates an adapter on a held-out preference set

## Inputs
- A complete Hugging Face GPT-2 directory with config and tokenizer files
- Train preference set
- Held-out eval preference set

## Default command
```bash
bash review-stage/run_gpt2_dpo_vs_sft.sh
```

## Current status
- Harness added
- Not executed in this sandbox because the local GPT-2 directory only contains weights and lacks tokenizer/config files required by the existing trainer path

## Next
- Run the harness locally with a complete GPT-2 model directory
- Compare `dpo_eval.json` and `sft_eval.json`
