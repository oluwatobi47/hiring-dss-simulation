# Local Simulation Runners

This folder contains standalone runners for the local hiring DSS simulation.

It includes both Llama.cpp-backed and OpenAI-backed execution paths.

The runner loads the simulation JSON files, builds LlamaIndex vector stores over the referenced resume and job-description files, runs the agent workflow for each configured test case, and writes result files named `TC_OUTPUT_<test-case>.json`.

## Install

From the repository root:

```bash
cd notebooks/local_simulation/scripts
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For the OpenAI runner, install the OpenAI-specific dependencies instead:

```bash
cd notebooks/local_simulation/scripts
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_openai.txt
```

For CUDA-backed `llama-cpp-python`, install the llama.cpp package with the CMake flags that match your CUDA install before running the requirements install. Example:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
pip install -r requirements.txt
```

## Expected Files

Both simulation runners expect the simulation metadata and source documents:

```text
data/simulation/
  application_score_data.json
  evaluation_data_pool.json
  test_case_config.json
  truthful_qa_questions.json
  dataset/
    tc1/
      resume/*.pdf
      job_description/*.docx
```

The Llama.cpp runner also expects GGUF model files unless you pass `--model-path`:

```text
model/
  llama31_8b_hiring_fp16.gguf
  llama31_8b_hiring_Q4_K_M.gguf
  llama31_8b_hiring_Q8_0.gguf
  embedding/
```

The checked-in repository currently has the JSON metadata but not the referenced PDF/DOCX dataset files or GGUF model files. Put those assets in the paths above, or pass explicit paths with the arguments below.

## Download Llama Models

The model downloader pulls from `oluwatobi-alao/gguf_llama_models` into the repository `model/` folder by default:

```bash
python download_llama_models.py
```

Download only the default quantized model used in the examples:

```bash
python download_llama_models.py --model 8b_Q4
```

Download the fp16 model:

```bash
python download_llama_models.py --model 8b
```

If the Hugging Face repo requires authentication, either run `huggingface-cli login` first or pass a token:

```bash
python download_llama_models.py --token "$HF_TOKEN"
```

## Run

### Llama.cpp

Run the default quantized model over every test case and every question:

```bash
python run_llm_cpp_simulation.py --simulation-model 8b_Q4
```

Run a quick smoke test with one test case and one question:

```bash
python run_llm_cpp_simulation.py \
  --simulation-model 8b_Q4 \
  --test-case TC1 \
  --question-limit 1
```

Use a direct model path and a custom data folder:

```bash
python run_llm_cpp_simulation.py \
  --model-path /path/to/model.gguf \
  --data-dir /path/to/data/simulation \
  --output-dir /path/to/results
```

For CPU-only execution:

```bash
python run_llm_cpp_simulation.py --simulation-model 8b_Q4 --n-gpu-layers -1
```

By default, `run_llm_cpp_simulation.py` uses `--n-gpu-layers -1` (all layers).

### OpenAI

Set your API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

Run the OpenAI simulation with the default high-volume model:

```bash
python run_openai_simulation.py
```

Run a quick smoke test:

```bash
python run_openai_simulation.py \
  --test-case TC1 \
  --question-limit 1
```

Use a different OpenAI model:

```bash
python run_openai_simulation.py --model gpt-5.6-terra
```

The OpenAI docs currently list `gpt-6-astra` for highest capability, `gpt-6-sol` for balancing intelligence and cost, and `gpt-6-luna` for high-volume workloads. This script defaults to `gpt-5.6-terra` because the simulation can make many calls. If you use reasoning effort above `none` and the API rejects `temperature`, rerun with `--disable-temperature`.
The for the current version of api packages used in this project, the gpt-5 series are currently the highest capable models supported.

### Evaluate Simulation Results

After generating `TC_OUTPUT_*.json` files, run the evaluator to score correctness and relevancy:

```bash
python evaluate_simulation_results.py
```

Evaluate a specific results folder:

```bash
python evaluate_simulation_results.py \
  --results-dir data/simulation/output/openai/workflow_gpt-5.6-terra
```

Use a different judge model:

```bash
python evaluate_simulation_results.py --model gpt-5.6-terra
```

The evaluator writes `TC_EVAL_*.json` and `EVAL_SUMMARY.json` to `<results-dir>/evaluation` by default. The judge model default is `gpt-5.6-terra`.

## Main Arguments

Shared arguments:

- `--data-dir`: folder containing the simulation JSON files and referenced `dataset/` files.
- `--embedding-model-name`: Hugging Face embedding model name or local embedding model path. Defaults to `BAAI/bge-m3`.
- `--embedding-cache-dir`: cache folder for embedding downloads. Defaults to `model/embedding`.
- `--test-case`: run a specific test case, e.g. `TC1`. Can be repeated.
- `--question-limit`: run only the first N questions.

Llama.cpp-specific arguments:

- `--output-dir`: folder for result JSON files. Defaults to `data/simulation/output/gguf/workflow_<simulation-model>`.
- `--simulation-model`: one of `8b`, `8b_Q4`, or `8b_Q8`.
- `--model-path`: direct path to a `.gguf` file. Overrides `--simulation-model`.
- `--model-base-path`: folder containing the default model filenames.
- `--n-gpu-layers`: llama.cpp GPU layers. Defaults to `-1` (all layers). Use `0` for CPU-only.

OpenAI-specific arguments:

- `--output-dir`: folder for result JSON files. Defaults to `data/simulation/output/openai/workflow_<model>`.
- `--model`: OpenAI model ID. Defaults to `gpt-5.6-terra`.
- `--openai-api-key`: API key override. Defaults to `OPENAI_API_KEY`.
- `--reasoning-effort`: one of `none`, `low`, `medium`, `high`, `xhigh`, or `max`.
- `--disable-temperature`: omit temperature for models or reasoning modes that reject it.
- `--max-output-tokens`: maximum generated tokens per OpenAI call.

Results are appended to the relevant `TC_OUTPUT_<test-case>.json` file if it already exists.
