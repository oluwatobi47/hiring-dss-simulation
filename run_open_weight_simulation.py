#!/usr/bin/env python3
"""Run the shared hiring workflow against an OpenAI-compatible model server.

The server owns model loading, quantization and the model's native chat template.
No inference requests are sent to OpenAI by this runner.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlsplit
from model_comparison_registry import TOKENIZERS


ROOT = Path(__file__).resolve().parent
MODELS = {
    **TOKENIZERS,
    "qwen38_27b": "Qwen/Qwen3.8-27B",
    "gpt_oss_20b": "openai/gpt-oss-20b",
}


def json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object.")
    return parsed


def build_arg_parser(profile: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    if profile is None:
        parser.add_argument("--profile", choices=MODELS, required=True)
    else:
        parser.set_defaults(profile=profile)
    parser.add_argument("--api-base", default="http://localhost:8000/v1",
                        help="OpenAI-compatible server URL, including /v1.")
    parser.add_argument("--api-key", default=None,
                        help="Defaults to OPEN_WEIGHT_API_KEY, then 'local'.")
    parser.add_argument("--model", help="Server model ID/alias. Defaults to the profile's HF ID.")
    parser.add_argument("--tokenizer", help="HF tokenizer ID or local directory; defaults to profile HF ID.")
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--context-window", type=int, default=32768,
                        help="Must not exceed the context length configured on the server.")
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--retrieval-temperature", type=float, default=0.3)
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"],
                        help="gpt-oss only; defaults to medium for that profile.")
    parser.add_argument("--chat-template-kwargs", type=json_object, default={},
                        help='Server-supported template options as JSON, e.g. {"enable_thinking": false}.')
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "simulation")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="New or empty directory; existing runs are never appended to.")
    parser.add_argument("--embedding-model-name", default="BAAI/bge-m3")
    parser.add_argument("--embedding-cache-dir", type=Path, default=ROOT / "model" / "embedding")
    parser.add_argument("--test-case", action="append")
    parser.add_argument("--question-limit", type=int)
    parser.add_argument("--similarity-top-k", type=int, default=2)
    parser.add_argument("--max-tool-use", type=int, default=1)
    parser.add_argument("--checkpoint-label", default="unspecified",
                        help="Record the actual server checkpoint/revision and quantization.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def prepare_args(parser: argparse.ArgumentParser, argv: list[str] | None) -> argparse.Namespace:
    args = parser.parse_args(argv)
    endpoint = urlsplit(args.api_base)
    if endpoint.scheme not in {"http", "https"} or not endpoint.netloc:
        parser.error("--api-base must be an HTTP(S) server URL.")
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        parser.error("--api-base must not include credentials, query parameters or fragments.")
    if endpoint.hostname == "api.openai.com":
        parser.error("Use an open-weight model server, not api.openai.com.")
    for name in ("context_window", "max_output_tokens", "similarity_top_k", "max_tool_use", "timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.question_limit is not None and args.question_limit <= 0:
        parser.error("--question-limit must be positive.")
    if args.max_output_tokens >= args.context_window:
        parser.error("--max-output-tokens must be smaller than --context-window.")
    if args.reasoning_effort and args.profile != "gpt_oss_20b":
        parser.error("--reasoning-effort is supported here only for gpt_oss_20b.")
    if args.temperature < 0 or args.retrieval_temperature < 0:
        parser.error("Temperatures must be nonnegative.")
    args.model = args.model or MODELS[args.profile]
    args.tokenizer = args.tokenizer or MODELS[args.profile]
    if args.profile == "gpt_oss_20b":
        args.reasoning_effort = args.reasoning_effort or "medium"
    if args.output_dir is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model)
        args.output_dir = args.data_dir / "output" / "open_weight" / f"workflow_{slug}" / run_id
    args.api_key = args.api_key or os.environ.get("OPEN_WEIGHT_API_KEY") or "local"
    # Compatibility with the shared workflow; credentials are supplied to the adapter directly.
    args.openai_api_key = None
    args.disable_temperature = False
    args.verbose_openai = args.verbose
    return args


def build_llm_factory(args: argparse.Namespace, tokenizer: Any):
    from llama_index.llms.openai_like import OpenAILike

    def factory(*, model, system_prompt, temperature, max_output_tokens, reasoning_effort, verbose):
        additional: dict[str, Any] = {}
        if reasoning_effort:
            additional["reasoning_effort"] = reasoning_effort
        if args.chat_template_kwargs:
            additional["extra_body"] = {"chat_template_kwargs": args.chat_template_kwargs}
        return OpenAILike(
            model=model,
            api_base=args.api_base,
            api_key=args.api_key,
            context_window=args.context_window,
            max_tokens=max_output_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            is_chat_model=True,
            is_function_calling_model=False,
            tokenizer=tokenizer,
            additional_kwargs=additional,
            timeout=args.timeout,
            verbose=verbose,
        )

    return factory


async def run(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from run_openai_simulation import main_async, validate_inputs

    validate_inputs(args.data_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty --output-dir to avoid mixing runs: {args.output_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)
    llm_factory = build_llm_factory(args, tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key not in {"api_key", "openai_api_key"}}
    config["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    config["tokenizer_resolved_revision"] = tokenizer.init_kwargs.get("_commit_hash")
    config["token_count_note"] = "Client tokenizer estimates; server templates and hidden reasoning may differ."
    (args.output_dir / "RUN_CONFIG.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    await main_async(
        args,
        llm_factory=llm_factory,
        tokenizer=lambda text: tokenizer.encode(text, add_special_tokens=False),
        provider="open-weight",
        require_openai_key=False,
    )


def main(profile: str | None = None, argv: list[str] | None = None) -> None:
    args = prepare_args(build_arg_parser(profile), argv)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
