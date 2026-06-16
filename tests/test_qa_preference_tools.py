import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.qa_preference_tools import parse_judgment_text_fallback


REPO_ROOT = Path(__file__).resolve().parents[1]


class QaPreferenceToolTest(unittest.TestCase):
    def test_prepare_qa_dpo_jsonl_with_fake_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            input_path = tmpdir / "qa.jsonl"
            output_path = tmpdir / "dpo.jsonl"
            input_path.write_text(
                json.dumps({"prompt": "地球是圆的还是方的？", "response": "圆的"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/prepare_qa_dpo_jsonl.py"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--backend",
                    "fake",
                    "--fake_rejected",
                    "方的",
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["prompt"], "地球是圆的还是方的？")
            self.assertEqual(rows[0]["chosen"], "圆的")
            self.assertEqual(rows[0]["rejected"], "方的")

    def test_judge_qa_answers_with_gold_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            input_path = tmpdir / "qa.jsonl"
            output_path = tmpdir / "judged.jsonl"
            pair_path = tmpdir / "pairs.jsonl"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {"prompt": "地球是圆的还是方的？", "response": "圆的", "gold": "圆的"},
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {"prompt": "地球是圆的还是方的？", "response": "方的", "gold": "圆的"},
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {"prompt": "水在标准大气压下多少摄氏度沸腾？", "response": "100 摄氏度", "gold": "100 摄氏度"},
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {"prompt": "水在标准大气压下多少摄氏度沸腾？", "response": "0 摄氏度", "gold": "100 摄氏度"},
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/judge_qa_answers.py"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--pair_output",
                    str(pair_path),
                    "--backend",
                    "gold",
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["label"] for row in rows], ["chosen", "unchosen", "chosen", "unchosen"])
            self.assertEqual([row["is_correct"] for row in rows], [True, False, True, False])
            self.assertTrue(all(row["training_format"] == "single_feedback" for row in rows))

            pairs = [json.loads(line) for line in pair_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(pairs), 2)
            self.assertEqual(pairs[0]["prompt"], "地球是圆的还是方的？")
            self.assertEqual(pairs[0]["chosen"], "圆的")
            self.assertEqual(pairs[0]["rejected"], "方的")
            self.assertEqual(pairs[1]["chosen"], "100 摄氏度")
            self.assertEqual(pairs[1]["rejected"], "0 摄氏度")

    def test_qwen_judge_text_fallback(self):
        good = parse_judgment_text_fallback("The response is correct and answers the question.", "qwen_answer_judge")
        bad = parse_judgment_text_fallback("The response is incorrect and does not answer the question.", "qwen_answer_judge")
        self.assertTrue(good.is_correct)
        self.assertFalse(bad.is_correct)

    def test_unique_prompts_keep_single_feedback_without_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            input_path = tmpdir / "unique.jsonl"
            output_path = tmpdir / "judged.jsonl"
            pair_path = tmpdir / "pairs.jsonl"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps({"prompt": "What does DNS do?", "response": "It maps names to addresses.", "gold": "It maps names to addresses."}),
                        json.dumps({"prompt": "What is binary search?", "response": "It checks random items.", "gold": "It halves a sorted search space."}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/judge_qa_answers.py"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--pair_output",
                    str(pair_path),
                    "--backend",
                    "gold",
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            pairs = [json.loads(line) for line in pair_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(pairs), 0)
            self.assertEqual([row["training_format"] for row in rows], ["single_feedback", "single_feedback"])


if __name__ == "__main__":
    unittest.main()
