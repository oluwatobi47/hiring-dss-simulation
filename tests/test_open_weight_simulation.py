"""Offline checks for model selection, request configuration and workflow reuse."""

import argparse
import asyncio
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from run_open_weight_simulation import MODELS, build_arg_parser, build_llm_factory, prepare_args, run


class OpenWeightTests(unittest.TestCase):
    def args(self, profile="qwen35_9b", *flags):
        return prepare_args(build_arg_parser(profile), list(flags))

    def test_profiles_and_default_output_isolation(self):
        paths = set()
        for profile, model in MODELS.items():
            args = self.args(profile)
            self.assertEqual(args.model, model)
            self.assertEqual(args.tokenizer, model)
            self.assertIn("open_weight", args.output_dir.parts)
            paths.add(args.output_dir)
            self.assertEqual(args.reasoning_effort, "medium" if profile == "gpt_oss_20b" else None)
        self.assertEqual(len(paths), 4)

    def test_invalid_configuration(self):
        for flags in [
            ["--question-limit", "0"],
            ["--context-window", "1024"],
            ["--reasoning-effort", "high"],
            ["--api-base", "https://api.openai.com/v1"],
            ["--api-base", "http://user:secret@localhost/v1"],
            ["--chat-template-kwargs", "[]"],
        ]:
            with self.subTest(flags=flags), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.args("qwen35_9b", *flags)

    def test_adapter_uses_chat_endpoint_and_model_specific_options(self):
        adapter = MagicMock()
        module = types.ModuleType("llama_index.llms.openai_like")
        module.OpenAILike = adapter
        args = self.args("gpt_oss_20b", "--api-base", "http://localhost:9000/v1")
        tokenizer = object()
        with patch.dict(sys.modules, {"llama_index.llms.openai_like": module}):
            factory = build_llm_factory(args, tokenizer)
            factory(model=args.model, system_prompt="Evidence only", temperature=0.3,
                    max_output_tokens=512, reasoning_effort="high", verbose=False)
        kwargs = adapter.call_args.kwargs
        self.assertEqual(kwargs["api_base"], args.api_base)
        self.assertTrue(kwargs["is_chat_model"])
        self.assertFalse(kwargs["is_function_calling_model"])
        self.assertIs(kwargs["tokenizer"], tokenizer)
        self.assertEqual(kwargs["additional_kwargs"], {"reasoning_effort": "high"})
        self.assertEqual(kwargs["system_prompt"], "Evidence only")

    def test_run_reuses_workflow_and_redacts_credentials(self):
        tokenizer = MagicMock()
        tokenizer.init_kwargs = {"_commit_hash": "test-revision"}
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = MagicMock()
        transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
        workflow = types.ModuleType("run_openai_simulation")
        workflow.validate_inputs = MagicMock()
        workflow.main_async = AsyncMock()
        with tempfile.TemporaryDirectory() as folder:
            args = self.args("qwen35_9b", "--output-dir", folder, "--api-key", "test-secret",
                             "--model", "served-alias", "--chat-template-kwargs",
                             '{"enable_thinking": false}')
            with patch.dict(sys.modules, {"transformers": transformers, "run_openai_simulation": workflow}), patch(
                "run_open_weight_simulation.build_llm_factory", return_value="factory"
            ):
                asyncio.run(run(args))
                config_text = (Path(folder) / "RUN_CONFIG.json").read_text()
                config = json.loads(config_text)
                self.assertNotIn("test-secret", config_text)
                self.assertEqual(config["model"], "served-alias")
                self.assertEqual(config["tokenizer"], MODELS["qwen35_9b"])
                self.assertEqual(config["chat_template_kwargs"], {"enable_thinking": False})
                self.assertEqual(config["tokenizer_resolved_revision"], "test-revision")
                kwargs = workflow.main_async.call_args.kwargs
                self.assertFalse(kwargs["require_openai_key"])
                self.assertEqual(kwargs["llm_factory"], "factory")
                workflow.main_async.assert_awaited_once()
                with self.assertRaises(FileExistsError):
                    asyncio.run(run(args))


if __name__ == "__main__":
    unittest.main()
