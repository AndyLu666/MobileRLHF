import tempfile
import unittest
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel

from pytorch_alignment.dpo_utils import (
    BinaryFeedbackDataset,
    PreferenceDataset,
    binary_feedback_collate_batch,
    compute_dpo_batch_metrics,
    dpo_collate_batch,
    evaluate_dpo,
    is_desirable_label,
)


class SimpleCharTokenizer:
    def __init__(self):
        alphabet = (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            " .,!?;:-_+/\\'\"()[]{}\n"
        )
        self.pad_token_id = 0
        self.eos_token_id = 1
        self._stoi = {ch: idx + 2 for idx, ch in enumerate(alphabet)}
        self.vocab_size = len(self._stoi) + 2

    def encode(self, text, add_special_tokens=False):
        ids = [self._stoi.get(ch, self.eos_token_id) for ch in text]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids


def build_tiny_policy(vocab_size: int):
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=128,
        n_ctx=128,
        n_embd=64,
        n_layer=2,
        n_head=2,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_cfg)


class DpoUtilsTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = SimpleCharTokenizer()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_path = Path(self.tmpdir.name) / "dpo.jsonl"
        self.data_path.write_text(
            "\n".join(
                [
                    '{"prompt":"User: Say hello.\\nAssistant:","chosen":" Hello there.","rejected":" No."}',
                    '{"prompt":"User: What is 2+2?\\nAssistant:","chosen":" 2+2 is 4.","rejected":" 2+2 is 5."}',
                    '{"prompt":"User: Thank me.\\nAssistant:","chosen":" Thank you!","rejected":" Whatever."}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_preference_dataset_and_collator(self):
        dataset = PreferenceDataset(str(self.data_path), self.tokenizer, seq_len=64)
        self.assertEqual(len(dataset), 3)
        batch = dpo_collate_batch([dataset[0], dataset[1]])
        self.assertEqual(batch["chosen_input_ids"].shape, torch.Size([2, 64]))
        self.assertTrue(torch.any(batch["chosen_response_mask"] > 0))
        self.assertTrue(torch.any(batch["rejected_response_mask"] > 0))

    def test_binary_feedback_dataset_accepts_judged_labels(self):
        feedback_path = Path(self.tmpdir.name) / "feedback.jsonl"
        feedback_path.write_text(
            "\n".join(
                [
                    '{"prompt":"User: What is 2+2?\\nAssistant:","response":" 4","label":"chosen"}',
                    '{"prompt":"User: What is 2+2?\\nAssistant:","response":" 5","label":"unchosen"}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        self.assertTrue(is_desirable_label("chosen"))
        self.assertTrue(is_desirable_label("correct"))
        self.assertFalse(is_desirable_label("unchosen"))
        dataset = BinaryFeedbackDataset(str(feedback_path), self.tokenizer, seq_len=64)
        self.assertEqual(len(dataset), 2)
        batch = binary_feedback_collate_batch([dataset[0], dataset[1]])
        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 64]))
        self.assertEqual(batch["label"].tolist(), [1.0, -1.0])

    def test_dpo_metrics_backward_and_eval(self):
        dataset = PreferenceDataset(str(self.data_path), self.tokenizer, seq_len=64)
        batch = dpo_collate_batch([dataset[i] for i in range(len(dataset))])
        batch = {k: v.to("cpu") for k, v in batch.items()}

        torch.manual_seed(42)
        policy_model = build_tiny_policy(self.tokenizer.vocab_size)
        reference_model = build_tiny_policy(self.tokenizer.vocab_size)
        reference_model.load_state_dict(policy_model.state_dict())
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False

        metrics = compute_dpo_batch_metrics(policy_model, reference_model, batch, beta=0.1)
        self.assertTrue(torch.isfinite(metrics["loss"]))
        metrics["loss"].backward()

        nonzero_policy_grads = 0
        for _, param in policy_model.named_parameters():
            if param.requires_grad and param.grad is not None and torch.count_nonzero(param.grad) > 0:
                nonzero_policy_grads += 1
        self.assertGreater(nonzero_policy_grads, 0)

        for _, param in reference_model.named_parameters():
            self.assertIsNone(param.grad)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=dpo_collate_batch,
        )
        eval_metrics = evaluate_dpo(policy_model, reference_model, dataloader, torch.device("cpu"), 10, beta=0.1)
        self.assertIn("loss", eval_metrics)
        self.assertIn("reward_accuracy", eval_metrics)
        self.assertTrue(eval_metrics["loss"] < float("inf"))


if __name__ == "__main__":
    unittest.main()
