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
    compute_kto_batch_metrics,
    compute_dpo_batch_metrics,
    dpo_collate_batch,
    evaluate_dpo,
    is_desirable_label,
)
from pytorch_alignment.kto_cli import build_qwen_lora_config, resolve_qwen_model_dir, run_tiny_kto_from_jsonl
from pytorch_alignment.kto_trainer import KtoTrainingConfig, train_kto
from scripts.convert_dpo_to_kto_jsonl import convert_dpo_pairs_to_kto_records


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

    def test_convert_dpo_pairs_to_balanced_kto_records(self):
        kto_path = Path(self.tmpdir.name) / "kto.jsonl"
        stats = convert_dpo_pairs_to_kto_records(self.data_path, kto_path)

        self.assertEqual(stats["pairs"], 3)
        self.assertEqual(stats["written"], 6)
        self.assertEqual(stats["desirable"], 3)
        self.assertEqual(stats["undesirable"], 3)

        records = [
            __import__("json").loads(line)
            for line in kto_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(records[0]["response"], " Hello there.")
        self.assertIs(records[0]["label"], True)
        self.assertEqual(records[1]["response"], " No.")
        self.assertIs(records[1]["label"], False)
        self.assertEqual({record["feedback_format"] for record in records}, {"kto_binary"})

    def test_kto_metrics_use_binary_feedback_names_and_weights(self):
        feedback_path = Path(self.tmpdir.name) / "feedback.jsonl"
        feedback_path.write_text(
            "\n".join(
                [
                    '{"prompt":"User: How should I recover after a low sleep night?\\nAssistant:","response":" Take a short walk, hydrate, and keep intensity light today.","label":true}',
                    '{"prompt":"User: How should I recover after a low sleep night?\\nAssistant:","response":" Ignore fatigue and do maximum intensity intervals immediately.","label":false}',
                    '{"prompt":"User: My steps are low today.\\nAssistant:","response":" Add a 10 minute walk after dinner and stretch before bed.","label":true}',
                    '{"prompt":"User: My steps are low today.\\nAssistant:","response":" Do nothing and worry about it tomorrow.","label":false}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        dataset = BinaryFeedbackDataset(str(feedback_path), self.tokenizer, seq_len=96)
        batch = binary_feedback_collate_batch([dataset[i] for i in range(len(dataset))])
        batch = {k: v.to("cpu") for k, v in batch.items()}

        torch.manual_seed(7)
        policy_model = build_tiny_policy(self.tokenizer.vocab_size)
        reference_model = build_tiny_policy(self.tokenizer.vocab_size)
        reference_model.load_state_dict(policy_model.state_dict())
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False

        base_metrics = compute_kto_batch_metrics(
            policy_model,
            reference_model,
            batch,
            beta=0.1,
            desirable_weight=1.0,
            undesirable_weight=1.0,
        )
        weighted_metrics = compute_kto_batch_metrics(
            policy_model,
            reference_model,
            batch,
            beta=0.1,
            desirable_weight=1.0,
            undesirable_weight=2.0,
        )

        self.assertTrue(torch.isfinite(base_metrics["loss"]))
        self.assertIn("desirable_reward", base_metrics)
        self.assertIn("undesirable_reward", base_metrics)
        self.assertIn("kl_center", base_metrics)
        self.assertIn("reward_margin", base_metrics)
        self.assertNotIn("chosen_reward", base_metrics)
        self.assertGreater(weighted_metrics["loss"].item(), base_metrics["loss"].item())

        base_metrics["loss"].backward()
        nonzero_policy_grads = sum(
            1
            for _, param in policy_model.named_parameters()
            if param.requires_grad and param.grad is not None and torch.count_nonzero(param.grad) > 0
        )
        self.assertGreater(nonzero_policy_grads, 0)
        for _, param in reference_model.named_parameters():
            self.assertIsNone(param.grad)

    def test_train_kto_runs_multiple_steps_and_writes_metrics(self):
        feedback_path = Path(self.tmpdir.name) / "feedback.jsonl"
        feedback_path.write_text(
            "\n".join(
                [
                    '{"prompt":"User: My steps are low.\\nAssistant:","response":" Take a gentle 10 minute walk after dinner.","label":true}',
                    '{"prompt":"User: My steps are low.\\nAssistant:","response":" Do nothing and feel guilty.","label":false}',
                    '{"prompt":"User: I slept poorly.\\nAssistant:","response":" Keep training light and protect bedtime.","label":true}',
                    '{"prompt":"User: I slept poorly.\\nAssistant:","response":" Do hard intervals anyway.","label":false}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        dataset = BinaryFeedbackDataset(str(feedback_path), self.tokenizer, seq_len=96)

        torch.manual_seed(11)
        policy_model = build_tiny_policy(self.tokenizer.vocab_size)
        reference_model = build_tiny_policy(self.tokenizer.vocab_size)
        reference_model.load_state_dict(policy_model.state_dict())
        metrics_path = Path(self.tmpdir.name) / "metrics.ndjson"

        result = train_kto(
            policy_model,
            reference_model,
            train_dataset=dataset,
            eval_dataset=dataset,
            config=KtoTrainingConfig(
                max_steps=3,
                batch_size=2,
                beta=0.1,
                lr=1e-3,
                metrics_path=metrics_path,
                seed=11,
            ),
            device=torch.device("cpu"),
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["train_samples"], 4)
        self.assertEqual(result["max_steps"], 3)
        self.assertEqual(result["last_train_metrics"]["step"], 3)
        self.assertGreater(result["last_train_metrics"]["policy_nonzero_grad_params"], 0)
        self.assertEqual(result["last_train_metrics"]["reference_grad_params"], 0)
        self.assertIn("eval_metrics", result)
        self.assertEqual(len(metrics_path.read_text(encoding="utf-8").splitlines()), 3)

    def test_tiny_kto_cli_runner_writes_report_and_metrics(self):
        feedback_path = Path(self.tmpdir.name) / "feedback.jsonl"
        feedback_path.write_text(
            "\n".join(
                [
                    '{"prompt":"User: I am tired.\\nAssistant:","response":" Keep activity easy and sleep earlier tonight.","label":true}',
                    '{"prompt":"User: I am tired.\\nAssistant:","response":" Push as hard as possible immediately.","label":false}',
                    '{"prompt":"User: My stress is high.\\nAssistant:","response":" Try slow breathing and a short walk.","label":true}',
                    '{"prompt":"User: My stress is high.\\nAssistant:","response":" Ignore stress and keep working all night.","label":false}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        out_dir = Path(self.tmpdir.name) / "tiny_cli"

        result = run_tiny_kto_from_jsonl(
            kto_jsonl=feedback_path,
            eval_jsonl=feedback_path,
            out_dir=out_dir,
            max_steps=2,
            batch_size=2,
            seq_len=96,
            beta=0.1,
            lr=1e-3,
            seed=13,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["max_steps"], 2)
        self.assertTrue((out_dir / "report.json").is_file())
        self.assertTrue((out_dir / "metrics.ndjson").is_file())
        self.assertEqual(len((out_dir / "metrics.ndjson").read_text(encoding="utf-8").splitlines()), 2)

    def test_qwen_backend_defaults_to_local_qwen3_0_6b(self):
        repo_root = Path(__file__).resolve().parents[1]
        model_dir = resolve_qwen_model_dir(repo_root, None)
        self.assertEqual(model_dir.name, "pretrained")
        self.assertEqual(model_dir.parent.name, "qwen_lora_finetune")
        if not (model_dir / "config.json").is_file():
            self.skipTest(f"local Qwen model assets are not available at {model_dir}")
        lora_config = build_qwen_lora_config(rank=2, alpha=4, dropout=0.0)
        self.assertEqual(lora_config.r, 2)
        self.assertEqual(lora_config.lora_alpha, 4)
        self.assertEqual(set(lora_config.target_modules), {"q_proj", "k_proj", "v_proj", "o_proj"})

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
