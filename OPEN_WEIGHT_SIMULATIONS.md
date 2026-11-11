# Open-weight hiring DSS simulations

These runners share the prompts, retrieval, agent workflow and result schema in
`run_openai_simulation.py`. Mistral is excluded.

| Entry point | Default model |
|---|---|
| `run_open_weight_simulation.py --profile llama31_8b` | `meta-llama/Llama-3.1-8B-Instruct` |
| `run_qwen35_9b_simulation.py` | `Qwen/Qwen3.5-9B` |
| `run_gemma4_12b_simulation.py` | `google/gemma-4-12B-it` |
| `run_qwen38_27b_simulation.py` | `Qwen/Qwen3.8-27B` |
| `run_gpt_oss_20b_simulation.py` | `openai/gpt-oss-20b` |

## Setup

In your project Python environment, install:

```bash
pip install -r requirements_open_weight.txt
```

Start an OpenAI-compatible inference server with the selected model. The runners
are clients; they do not start a server or download the generation model weights.
Use a recent server release supporting the exact architecture. Consult the
official model cards for installation, accelerator and memory requirements:

- [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Gemma 4 12B](https://huggingface.co/google/gemma-4-12B-it)
- [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)

For a supported vLLM installation, the command shape is:

```bash
vllm serve Qwen/Qwen3.5-9B --port 8000 --max-model-len 32768
```

Replace the model ID with the model being tested. Hardware, quantization and
architecture-specific flags must follow its model card. The server must apply
the model's native chat template and return the final answer in
`choices[].message.content`. In particular, gpt-oss requires Harmony formatting
and server-side reasoning parsing. Do not force a Llama template on these models.
Hidden reasoning should be separated from final answer content.

Simulation JSON and referenced PDF/DOCX assets must be available under
`data/simulation` (or `--data-dir`). Tokenizers and BGE-M3 embeddings may download
on first use; Hugging Face authentication may be needed for restricted assets.

## Run

With the corresponding model served at port 8000:

```bash
python run_qwen35_9b_simulation.py --test-case TC1 --question-limit 1
python run_gemma4_12b_simulation.py --test-case TC1 --question-limit 1
python run_qwen38_27b_simulation.py --test-case TC1 --question-limit 1
python run_gpt_oss_20b_simulation.py --test-case TC1 --question-limit 1 --reasoning-effort medium
```

Run each command against its matching server, rather than one unchanged model
server for all four. Omit `--test-case` and `--question-limit` for the full study.

Use `--api-base http://localhost:9000/v1` for another endpoint. For authenticated
servers set `OPEN_WEIGHT_API_KEY`; no OpenAI API key is required. If the server
uses an alias, pass `--model served-alias`. The tokenizer still defaults to the
profile's original HF tokenizer; override with `--tokenizer` for another checkpoint.

For supported servers and models, template options can be supplied explicitly:

```bash
python run_qwen35_9b_simulation.py --chat-template-kwargs '{"enable_thinking": false}'
```

The runner does not assume that template options work across model families.
gpt-oss defaults to medium reasoning effort; other profiles send no reasoning
effort parameter. Match the output budget to the reasoning experiment: 2048
tokens may truncate reasoning-heavy generations. Set `--context-window` to the
context configured on the server; it defaults to 32768.

## Results and evaluation

Every run gets a fresh directory:

```text
data/simulation/output/open_weight/workflow_<model>/<UTC-run-id>/
  RUN_CONFIG.json
  TC_OUTPUT_TC1.json
  ...
```

An explicit `--output-dir` must be empty. This prevents mixing checkpoints or
repeated runs. `RUN_CONFIG.json` records client settings without API keys.
Use `--checkpoint-label` to record the actual server checkpoint revision,
quantization, server version, hardware and reasoning/parser configuration:
these cannot be inferred reliably from a served model alias.

Token counts are client tokenizer estimates, not authoritative counts of server
chat-template tokens or hidden reasoning. Latency includes serving and network
effects. Keep embeddings, retrieval, data and prompts fixed across comparisons.

The existing evaluator accepts these result files:

```bash
python evaluate_simulation_results.py --results-dir /absolute/path/to/run
```

Evaluation uses the existing OpenAI judge and requires its API credentials.
For the ten-configuration GGUF/Terra study and its new analysis notebook, follow
[the comparison README](README.md). It uses an explicit run manifest.

## Offline validation

```bash
python -m unittest discover -s tests -p 'test_open_weight_simulation.py' -v
```

These tests cover profiles, invalid settings, adapter arguments, manifest
redaction and shared-workflow dispatch with mocks. They do not measure model
quality or verify a live server's template/architecture support.
