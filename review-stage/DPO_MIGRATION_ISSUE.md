# Feature: Replace SFT Loss With DPO-Based Preference Optimization

## Background

Current fine-tuning logic in MobileFineTuner uses standard supervised fine-tuning (SFT):

- Dataset provides a single target answer
- Loss is token-level cross entropy against labels
- Optimization objective is next-token prediction

This setup is not suitable for preference-driven personalization or RLHF-style alignment. We want to migrate the training pipeline from answer supervision to Direct Preference Optimization (DPO).

---

## Goal

Replace the existing label-based supervised loss with DPO preference loss.

The new pipeline should:

- Train from `(prompt, chosen, rejected)` preference tuples
- Compare log-probabilities between preferred and rejected responses
- Optimize the policy model against a frozen reference model
- Support LoRA fine-tuning
- Preserve current lightweight/mobile-friendly training infrastructure

---

## Scope

### In Scope

- PyTorch alignment trainers
- Dataset format changes
- DPO loss implementation
- Reference model loading
- Batch collation for paired samples
- Training/eval logging updates
- LoRA compatibility
- CLI flags for DPO hyperparameters

### Out of Scope (initial phase)

- PPO/RLHF rollout infrastructure
- Reward model training
- Online preference collection
- C++ runtime DPO implementation
- Multi-turn conversational preference trees

---

## Existing SFT Flow

Current training flow:

1. Dataset returns:
   - input_ids
   - attention_mask
   - labels

2. Trainer computes:
   - logits = model(...)
   - loss = cross_entropy(logits, labels)

3. Backpropagation:
   - loss.backward()

---

## Target DPO Flow

New dataset format:

```json
{
  "prompt": "...",
  "chosen": "preferred answer",
  "rejected": "bad answer"
}
```

Training flow:

1. Tokenize:
   - prompt + chosen
   - prompt + rejected

2. Compute:
   - policy_logps_chosen
   - policy_logps_rejected
   - ref_logps_chosen
   - ref_logps_rejected

3. DPO objective:

```python
pi_logratios = policy_chosen - policy_rejected
ref_logratios = ref_chosen - ref_rejected
loss = -logsigmoid(beta * (pi_logratios - ref_logratios)).mean()
```

4. Optimize only policy LoRA weights.

---

## Planned Architecture Changes

### 1. New Dataset Class

Add:

- `PreferenceDataset`
- `DPOSample`
- `dpo_collate_batch`

Responsibilities:

- Read JSONL preference data
- Build paired tokenized samples
- Generate masks for response-only logprob computation

---

### 2. Reference Model

Add frozen reference model:

```python
ref_model = AutoModelForCausalLM.from_pretrained(...)
ref_model.eval()
```

Requirements:

- no gradients
- optional shared quantization
- same tokenizer/config as policy model

---

### 3. DPO Loss Module

Add helper:

```python
compute_dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta,
)
```

Include:

- masking utilities
- sequence logprob aggregation
- numerical stability handling

---

### 4. CLI Changes

New args:

```bash
--train_mode dpo
--dpo_beta 0.1
--preference_jsonl xxx.jsonl
--reference_model_dir xxx
```

Backward compatibility:

- existing SFT remains available
- default mode can remain `sft`

---

### 5. Metrics

Add:

- chosen reward margin
- rejected reward margin
- preference accuracy
- implicit reward statistics

Retain:

- loss
- lr
- grad norm

Perplexity becomes optional / less meaningful.

---

## Implementation Order

### Phase 1 — Infrastructure

- [ ] Add DPO dataset parser
- [ ] Add pairwise collator
- [ ] Add sequence logprob utility
- [ ] Add DPO loss utility

### Phase 2 — Trainer Integration

- [ ] Add reference model loading
- [ ] Add DPO forward pass
- [ ] Add DPO optimizer step
- [ ] Add metric logging

### Phase 3 — CLI + Config

- [ ] Add training mode switch
- [ ] Add DPO hyperparameters
- [ ] Add validation checks

### Phase 4 — Validation

- [ ] Verify gradients only update LoRA params
- [ ] Verify frozen reference model
- [ ] Run tiny preference dataset sanity test
- [ ] Compare against baseline SFT

---

## Technical Risks

### Memory Overhead

DPO requires:

- policy chosen forward
- policy rejected forward
- reference chosen forward
- reference rejected forward

Mitigation:

- batch size reduction
- gradient accumulation
- shared tokenization
- no_grad for reference model

---

### Numerical Stability

Potential issue:

- exploding log-ratio differences

Mitigation:

- clamp values
- fp32 accumulation
- stable logsigmoid

---

## Acceptance Criteria

The migration is complete when:

- Training no longer depends on token labels
- Preference JSONL datasets train successfully
- DPO loss decreases during training
- LoRA checkpoints save/load correctly
- Existing SFT mode still works
- Small-scale sanity runs complete without OOM

---

## Initial Target Files

Primary likely touch points:

- `pytorch_alignment/qwen_lora_finetune.py`
- `pytorch_alignment/gpt2_full_finetune.py`
- `scripts/train_gemma3_lora_torch.py`
- related dataset/collator utilities

Future extension:

- C++ mobile runtime DPO support
- online RLHF pipeline
- reward model integration
