import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


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
                    "--backend",
                    "gold",
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["label"] for row in rows], ["chosen", "unchosen"])
            self.assertEqual([row["is_correct"] for row in rows], [True, False])


if __name__ == "__main__":
    unittest.main()
