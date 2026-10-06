#!/usr/bin/env python3
"""Evaluate local simulation outputs with an OpenAI judge model."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "simulation"
DEFAULT_JUDGE_MODEL = "gpt-5.6-terra"

CORRECTNESS_JUDGE_PROMPT = """\
You are an expert evaluator for question answering systems.

Given a prompt, expected answer, model answer, and optional retrieved context,
return a strict JSON object with this structure:

{
  "correct": boolean,
  "answer_relevant": boolean,
  "context_relevant": boolean,
  "correctness_reason": string,
  "relevancy_reason": string,
  "context_relevancy_reason": string
}

Definitions:
- correct: model answer is factually aligned with expected answer for the prompt.
- answer_relevant: model answer is relevant to the prompt even if incomplete.
- context_relevant: retrieved context is useful and relevant for answering the prompt.

Rules:
1. Respond with JSON only (no markdown).
2. If model answer is empty, set correct=false and answer_relevant=false.
3. If context is empty, set context_relevant=false with reason "No context provided.".
4. Keep reasons short and specific.
"""


@dataclass
class EvaluationRecord:
    id: Any
    batch_id: str
    question: str
    model_answer: str
    expected_answer: str
    correct: bool
    answer_relevant: bool
    context_relevant: bool
    relevancy_score: float
    context_relevancy_score: float
    correctness_reason: str
    relevancy_reason: str
    context_relevancy_reason: str
    source_file: str
    processing_time: Optional[float]
    prompt_token_count: Optional[int]
    response_token_count: Optional[int]
    error: Optional[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "batch_id": self.batch_id,
            "question": self.question,
            "model_answer": self.model_answer,
            "expected_answer": self.expected_answer,
            "correct": self.correct,
            "answer_relevant": self.answer_relevant,
            "context_relevant": self.context_relevant,
            "relevancy_score": self.relevancy_score,
            "context_relevancy_score": self.context_relevancy_score,
            "correctness_reason": self.correctness_reason,
            "relevancy_reason": self.relevancy_reason,
            "context_relevancy_reason": self.context_relevancy_reason,
            "source_file": self.source_file,
            "processing_time": self.processing_time,
            "prompt_token_count": self.prompt_token_count,
            "response_token_count": self.response_token_count,
            "error": self.error,
        }


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def find_output_files(results_dir: Path) -> list[Path]:
    return sorted(path for path in results_dir.rglob("TC_OUTPUT_*.json") if path.is_file())


def extract_json_object(value: str) -> str:
    patterns = (
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
        r"(\{.*\})",
    )
    for pattern in patterns:
        match = re.search(pattern, value, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


def safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in {"true", "yes", "1"}:
            return True
        if value.lower() in {"false", "no", "0"}:
            return False
    return default


def flatten_context(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    chunks.append(text)
            elif isinstance(item, dict):
                text = str(item.get("context") or item.get("text") or "").strip()
                if text:
                    chunks.append(text)
            else:
                text = str(item).strip()
                if text:
                    chunks.append(text)
        return "\n\n".join(chunks)
    return str(value).strip()


def build_eval_prompt(prompt: str, expected_answer: str, model_answer: str, context: str) -> str:
    return f"""\
Evaluate this model output.

Prompt:
{prompt}

Expected Answer:
{expected_answer}

Model Answer:
{model_answer}

Retrieved Context:
{context if context else 'No context provided.'}
"""


def judge_answer(llm: OpenAI, prompt: str, expected_answer: str, model_answer: str, context: str) -> dict[str, Any]:
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=CORRECTNESS_JUDGE_PROMPT),
        ChatMessage(role=MessageRole.USER, content=build_eval_prompt(prompt, expected_answer, model_answer, context)),
    ]
    response = llm.chat(messages=messages)
    raw = response.message.content if response.message and response.message.content else ""
    parsed_json = extract_json_object(raw)
    if not parsed_json:
        return {
            "correct": False,
            "answer_relevant": False,
            "context_relevant": False,
            "correctness_reason": "Judge response did not contain JSON.",
            "relevancy_reason": "Judge response did not contain JSON.",
            "context_relevancy_reason": "Judge response did not contain JSON.",
        }

    try:
        data = json.loads(parsed_json)
    except json.JSONDecodeError:
        return {
            "correct": False,
            "answer_relevant": False,
            "context_relevant": False,
            "correctness_reason": "Judge JSON could not be parsed.",
            "relevancy_reason": "Judge JSON could not be parsed.",
            "context_relevancy_reason": "Judge JSON could not be parsed.",
        }

    return {
        "correct": safe_bool(data.get("correct"), default=False),
        "answer_relevant": safe_bool(data.get("answer_relevant"), default=False),
        "context_relevant": safe_bool(data.get("context_relevant"), default=False),
        "correctness_reason": str(data.get("correctness_reason") or "No reason provided."),
        "relevancy_reason": str(data.get("relevancy_reason") or "No reason provided."),
        "context_relevancy_reason": str(data.get("context_relevancy_reason") or "No reason provided."),
    }


def process_output_file(output_file: Path, llm: OpenAI) -> list[dict[str, Any]]:
    payload = read_json(output_file)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list content in: {output_file}")

    results: list[dict[str, Any]] = []
    total_items = len(payload)
    for index, item in enumerate(payload, start=1):
        if total_items and (index == 1 or index % 10 == 0 or index == total_items):
            print(f"    Progress: {index}/{total_items} ({index / total_items:.1%})")

        if not isinstance(item, dict):
            continue

        question = str(item.get("prompt") or "")
        expected_answer = str(item.get("expected_output") or "")
        model_answer = str(item.get("model_response") or "")
        context_text = flatten_context(item.get("context"))
        batch_id = str(item.get("batch_id") or output_file.stem.replace("TC_OUTPUT_", ""))

        if not model_answer.strip():
            judged = {
                "correct": False,
                "answer_relevant": False,
                "context_relevant": bool(context_text),
                "correctness_reason": "Model answer is empty.",
                "relevancy_reason": "Model answer is empty.",
                "context_relevancy_reason": "Context exists but answer is empty." if context_text else "No context provided.",
            }
        else:
            judged = judge_answer(
                llm=llm,
                prompt=question,
                expected_answer=expected_answer,
                model_answer=model_answer,
                context=context_text,
            )

        correct = safe_bool(judged.get("correct"), default=False)
        answer_relevant = safe_bool(judged.get("answer_relevant"), default=False)
        context_relevant = safe_bool(judged.get("context_relevant"), default=False)
        correctness_reason = str(judged.get("correctness_reason") or "No reason provided.")
        relevancy_reason = str(judged.get("relevancy_reason") or "No reason provided.")
        context_relevancy_reason = str(judged.get("context_relevancy_reason") or "No reason provided.")

        record = EvaluationRecord(
            id=item.get("question_id"),
            batch_id=batch_id,
            question=question,
            model_answer=model_answer,
            expected_answer=expected_answer,
            correct=correct,
            answer_relevant=answer_relevant,
            context_relevant=context_relevant,
            relevancy_score=1.0 if answer_relevant else 0.0,
            context_relevancy_score=1.0 if context_relevant else 0.0,
            correctness_reason=correctness_reason,
            relevancy_reason=relevancy_reason,
            context_relevancy_reason=context_relevancy_reason,
            source_file=str(output_file),
            processing_time=item.get("processing_time"),
            prompt_token_count=item.get("prompt_token_count"),
            response_token_count=item.get("response_token_count"),
            error=item.get("error"),
        )
        results.append(record.as_dict())

    return results


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "total_questions": 0,
            "correct": 0,
            "answer_relevant": 0,
            "context_relevant": 0,
            "accuracy": 0.0,
            "answer_relevancy_rate": 0.0,
            "context_relevancy_rate": 0.0,
        }

    total = len(records)
    correct = sum(1 for row in records if row.get("correct") is True)
    answer_relevant = sum(1 for row in records if row.get("answer_relevant") is True)
    context_relevant = sum(1 for row in records if row.get("context_relevant") is True)

    return {
        "total_questions": total,
        "correct": correct,
        "answer_relevant": answer_relevant,
        "context_relevant": context_relevant,
        "accuracy": round(correct / total, 4),
        "answer_relevancy_rate": round(answer_relevant / total, 4),
        "context_relevancy_rate": round(context_relevant / total, 4),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate simulation result JSON files with an OpenAI judge model.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_DATA_DIR / "output",
        help="Folder containing one or more TC_OUTPUT_<test-case>.json files.",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=None,
        help="Destination folder for evaluation output. Defaults to <results-dir>/evaluation.",
    )
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL, help="OpenAI model for judgement.")
    parser.add_argument("--openai-api-key", default=None, help="Optional API key override; defaults to OPENAI_API_KEY.")
    parser.add_argument("--max-output-tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--verbose-openai", action="store_true", help="Enable LlamaIndex OpenAI verbose logging.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    results_dir = args.results_dir.resolve()
    evaluation_dir = (args.evaluation_dir or results_dir / "evaluation").resolve()

    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required. Set it in the environment or pass --openai-api-key.")

    output_files = find_output_files(results_dir)
    if not output_files:
        raise FileNotFoundError(f"No TC_OUTPUT_*.json files found in: {results_dir}")

    llm = OpenAI(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_output_tokens,
        verbose=args.verbose_openai,
    )

    overall_records: list[dict[str, Any]] = []
    per_file_summary: dict[str, Any] = {}

    total_files = len(output_files)
    for file_index, output_file in enumerate(output_files, start=1):
        print(f"Evaluating file {file_index}/{total_files}: {output_file}")
        file_records = process_output_file(output_file, llm)
        output_name = output_file.name.replace("TC_OUTPUT_", "TC_EVAL_")
        eval_path = evaluation_dir / output_name
        write_json(eval_path, file_records)
        summary = summarize(file_records)
        summary["evaluation_file"] = str(eval_path)
        per_file_summary[output_file.name] = summary
        overall_records.extend(file_records)
        print(
            f"  -> {summary['correct']}/{summary['total_questions']} correct"
            f" ({summary['accuracy']:.2%} accuracy)"
        )

    overall_summary = summarize(overall_records)
    write_json(
        evaluation_dir / "EVAL_SUMMARY.json",
        {
            "results_dir": str(results_dir),
            "evaluation_dir": str(evaluation_dir),
            "judge_model": args.model,
            "files": per_file_summary,
            "overall": overall_summary,
        },
    )

    print("Evaluation complete.")
    print(f"Evaluation files written to: {evaluation_dir}")
    print(
        f"Overall accuracy: {overall_summary['correct']}/{overall_summary['total_questions']}"
        f" ({overall_summary['accuracy']:.2%})"
    )


if __name__ == "__main__":
    main()
