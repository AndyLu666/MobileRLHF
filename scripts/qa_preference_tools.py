from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


PROMPT_KEYS = ("prompt", "question", "instruction", "input")
RESPONSE_KEYS = ("response", "answer", "chosen", "output")
GOLD_KEYS = ("gold", "gold_answer", "reference", "reference_answer", "answer")


@dataclass
class QARecord:
    prompt: str
    response: str
    raw: Dict[str, Any]
    line_no: int
    gold: Optional[str] = None


@dataclass
class ToolStats:
    total: int = 0
    kept: int = 0
    skipped_invalid_json: int = 0
    skipped_missing_fields: int = 0
    skipped_existing_rejected: int = 0
    skipped_generation_guard: int = 0
    skipped_judge_parse: int = 0
    exported_pairs: int = 0
    chosen_labels: int = 0
    unchosen_labels: int = 0
    unknown_labels: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "total": self.total,
            "kept": self.kept,
            "skipped_invalid_json": self.skipped_invalid_json,
            "skipped_missing_fields": self.skipped_missing_fields,
            "skipped_existing_rejected": self.skipped_existing_rejected,
            "skipped_generation_guard": self.skipped_generation_guard,
            "skipped_judge_parse": self.skipped_judge_parse,
            "exported_pairs": self.exported_pairs,
            "chosen_labels": self.chosen_labels,
            "unchosen_labels": self.unchosen_labels,
            "unknown_labels": self.unknown_labels,
        }


def compact_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def comparable_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.casefold())


def pick_text(row: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = compact_text(value)
        if text:
            return text
    return None


def normalize_qa_row(row: Dict[str, Any], line_no: int) -> Optional[QARecord]:
    prompt = pick_text(row, PROMPT_KEYS)
    response = pick_text(row, RESPONSE_KEYS)
    if not prompt or not response:
        return None
    gold = pick_text(row, GOLD_KEYS)
    return QARecord(prompt=prompt, response=response, gold=gold, raw=row, line_no=line_no)


def iter_jsonl(path: Path, stats: ToolStats) -> Iterable[tuple[int, Optional[Dict[str, Any]]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            stats.total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats.skipped_invalid_json += 1
                yield line_no, None
                continue
            if not isinstance(row, dict):
                stats.skipped_invalid_json += 1
                yield line_no, None
                continue
            yield line_no, row


def write_jsonl_row(handle, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def too_similar(left: str, right: str, threshold: float = 0.85) -> bool:
    left_cmp = comparable_text(left)
    right_cmp = comparable_text(right)
    if not left_cmp or not right_cmp:
        return True
    if left_cmp == right_cmp:
        return True
    return SequenceMatcher(None, left_cmp, right_cmp).ratio() >= threshold


def leaks_answer(candidate: str, answer: str) -> bool:
    candidate_cmp = comparable_text(candidate)
    answer_cmp = comparable_text(answer)
    if not candidate_cmp or not answer_cmp:
        return True
    if len(answer_cmp) <= 2:
        return answer_cmp in candidate_cmp
    return answer_cmp in candidate_cmp or candidate_cmp in answer_cmp


class NegativeGenerator:
    source = "negative_generator"

    def generate(self, prompt: str, chosen: str) -> str:
        raise NotImplementedError


class FakeNegativeGenerator(NegativeGenerator):
    source = "fake_negative_generation"

    def __init__(self, text: str = "incorrect answer"):
        self.text = text

    def generate(self, prompt: str, chosen: str) -> str:
        if self.text and not too_similar(self.text, chosen):
            return self.text
        return f"not {chosen}"


class QwenNegativeGenerator(NegativeGenerator):
    source = "qwen_negative_generation"

    def __init__(
        self,
        model_dir: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        device: str = "",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if seed >= 0:
            torch.manual_seed(seed)
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
        if not device:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model.to(device)
        self.model.eval()

    def generate(self, prompt: str, chosen: str) -> str:
        instruction = (
            "Create one rejected answer for preference training.\n"
            "The rejected answer must be factually wrong or clearly low quality.\n"
            "Do not include the correct answer. Do not explain. Do not add markdown.\n"
            "Return exactly one JSON object with this schema: "
            "{\"rejected\": \"short wrong answer\"}\n\n"
            f"Question: {prompt}\n"
            f"Correct answer to avoid: {chosen}\n"
            "JSON:"
        )
        inputs = self.tokenizer(instruction, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
        raw = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return clean_rejected_answer(raw)


@dataclass
class Judgment:
    is_correct: bool
    confidence: float
    reason: str
    source: str
    raw: Optional[str] = None

    @property
    def label(self) -> str:
        return "chosen" if self.is_correct else "unchosen"


class AnswerJudge:
    source = "answer_judge"

    def judge(self, prompt: str, response: str, gold: Optional[str] = None) -> Judgment:
        raise NotImplementedError


class FakeAnswerJudge(AnswerJudge):
    source = "fake_answer_judge"

    def judge(self, prompt: str, response: str, gold: Optional[str] = None) -> Judgment:
        bad_markers = ("wrong", "incorrect", "方", "false")
        is_correct = not any(marker in response.casefold() for marker in bad_markers)
        return Judgment(is_correct=is_correct, confidence=0.5, reason="fake deterministic label", source=self.source)


class GoldAnswerJudge(AnswerJudge):
    source = "gold_answer_judge"

    def judge(self, prompt: str, response: str, gold: Optional[str] = None) -> Judgment:
        if not gold:
            return Judgment(False, 0.0, "missing gold answer", self.source)
        response_cmp = comparable_text(response)
        gold_cmp = comparable_text(gold)
        is_correct = False
        if response_cmp and gold_cmp:
            if response_cmp == gold_cmp:
                is_correct = True
            else:
                length_ratio = min(len(response_cmp), len(gold_cmp)) / max(len(response_cmp), len(gold_cmp))
                is_correct = length_ratio >= 0.8 and SequenceMatcher(None, response_cmp, gold_cmp).ratio() >= 0.92
        return Judgment(
            is_correct=is_correct,
            confidence=1.0 if is_correct else 0.8,
            reason="matched gold answer" if is_correct else "did not match gold answer",
            source=self.source,
        )


class QwenAnswerJudge(AnswerJudge):
    source = "qwen_answer_judge"

    def __init__(
        self,
        model_dir: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        device: str = "",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if seed >= 0:
            torch.manual_seed(seed)
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
        if not device:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model.to(device)
        self.model.eval()

    def judge(self, prompt: str, response: str, gold: Optional[str] = None) -> Judgment:
        gold_block = f"\nReference answer: {gold}" if gold else ""
        instruction = (
            "Judge whether the response correctly answers the question. "
            "Focus on factual adequacy, not style. "
            "Return strict JSON only with keys is_correct, confidence, reason. "
            "Example: {\"is_correct\": true, \"confidence\": 0.7, \"reason\": \"answers the question\"}\n\n"
            f"Question: {prompt}\n"
            f"Response: {response}"
            f"{gold_block}\n"
            "JSON:"
        )
        inputs = self.tokenizer(instruction, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
        raw = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        try:
            judgment = parse_judgment_json(raw, self.source)
        except ValueError:
            judgment = parse_judgment_text_fallback(raw, self.source)
        if gold:
            gold_judgment = GoldAnswerJudge().judge(prompt, response, gold)
            if gold_judgment.is_correct != judgment.is_correct:
                return Judgment(
                    is_correct=gold_judgment.is_correct,
                    confidence=max(gold_judgment.confidence, judgment.confidence),
                    reason=(
                        "gold sanity override: "
                        f"{gold_judgment.reason}; qwen_reason={judgment.reason}"
                    ),
                    source=f"{self.source}_gold_sanity",
                    raw=raw,
                )
        return judgment


def parse_judgment_json(raw: str, source: str) -> Judgment:
    obj = extract_first_json_object(raw)
    if obj is None:
        raise ValueError("judge output did not contain a JSON object")
    data = json.loads(obj)
    if "is_correct" not in data:
        raise ValueError("judge JSON missing is_correct")
    confidence = float(data.get("confidence", 0.0))
    confidence = min(1.0, max(0.0, confidence))
    reason = compact_text(data.get("reason", ""))
    return Judgment(
        is_correct=bool(data["is_correct"]),
        confidence=confidence,
        reason=reason,
        source=source,
        raw=raw,
    )


def extract_first_json_object(raw: str) -> Optional[str]:
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : idx + 1]
    return None


def clean_rejected_answer(raw: str) -> str:
    obj = extract_first_json_object(raw)
    if obj:
        try:
            data = json.loads(obj)
            value = compact_text(data.get("rejected", ""))
            if value:
                return value
        except json.JSONDecodeError:
            pass

    text = raw.strip()
    for marker in (
        "\nAnswer:",
        "\nCorrect answer:",
        "\nExplanation:",
        "\nThe correct answer",
        "\nThe answer",
        "\nYou did not",
        "\nOkay,",
        "\nLet's",
    ):
        pos = text.find(marker)
        if pos >= 0:
            text = text[:pos]
    text = text.strip().strip('"').strip()
    return compact_text(text)


def parse_judgment_text_fallback(raw: str, source: str) -> Judgment:
    text = raw.strip()
    lowered = text.casefold()
    negative_hits = sum(
        marker in lowered
        for marker in (
            "incorrect",
            "not correct",
            "does not answer",
            "doesn't answer",
            "wrong",
            "false",
        )
    )
    positive_hits = sum(
        marker in lowered
        for marker in (
            "is correct",
            "correct answer",
            "answers the question",
            "true",
        )
    )
    is_correct = positive_hits > negative_hits
    reason = compact_text(text[:240]) if text else "qwen text fallback produced no explanation"
    return Judgment(
        is_correct=is_correct,
        confidence=0.35,
        reason=f"text fallback: {reason}",
        source=source,
        raw=raw,
    )
