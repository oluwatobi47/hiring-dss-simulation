# Run the hiring DSS experiments

Run these steps from the repository root on Linux using Bash. Run one model
family at a time: **Gemma → Llama → Qwen → GPT Terra**. Within each GGUF
family, run **BF16 → Q8_0 → Q4_0**. Each run finishes and its server exits
before the next precision starts.

## 1. Install dependencies

The CUDA build below requires an NVIDIA GPU, its driver, and the CUDA toolkit
with `nvcc` available. Install the driver/toolkit appropriate for your Linux
system first. Check the GPU and toolkit:

```bash
nvidia-smi
nvcc --version
sudo apt-get update
sudo apt-get install -y build-essential cmake git curl python3-venv python3-pip

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_open_weight.txt -r requirements_analysis.txt

# Build llama-server outside the project directory.
git clone https://github.com/ggml-org/llama.cpp.git ../llama.cpp
cmake -S ../llama.cpp -B ../llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build ../llama.cpp/build --config Release -j "$(nproc)" --target llama-server
export LLAMA_SERVER="$(realpath ../llama.cpp/build/bin/llama-server)"
"$LLAMA_SERVER" --version
```

If llama.cpp is already installed, set `LLAMA_SERVER` to its executable instead.
Use a build supporting your exact GGUF architectures and embedded chat templates.
The [official build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
and [server reference](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
describe build requirements and GPU fitting options. Use the same build for all
nine GGUF runs and retain its version in the experiment logs.

## 2. Prepare inputs and experiment paths

Place the simulation metadata and referenced documents under:

```text
data/simulation/
  application_score_data.json
  evaluation_data_pool.json
  test_case_config.json
  truthful_qa_questions.json
  dataset/
    ...referenced PDF resumes and DOCX job descriptions...
```

Set the model directory to the folder containing these exact filenames:

| Configuration | GGUF filename |
|---|---|
| `gemma4_12b_BF16` | `gemma-4-12B-it-BF16.gguf` |
| `gemma4_12b_Q8_0` | `gemma-4-12B-it-Q8_0.gguf` |
| `gemma4_12b_Q4_0` | `gemma-4-12B-it-Q4_0.gguf` |
| `llama31_8b_BF16` | `Llama-3.1-8B-Instruct-BF16.gguf` |
| `llama31_8b_Q8_0` | `Meta-Llama-3.1-8B-Instruct-Q8_0.gguf` |
| `llama31_8b_Q4_0` | `Llama-3.1-8B-Instruct-Q4_0.gguf` |
| `qwen35_9b_BF16` | `Qwen3.5-9B-BF16.gguf` |
| `qwen35_9b_Q8_0` | `Qwen3.5-9B-Q8_0.gguf` |
| `qwen35_9b_Q4_0` | `Qwen3.5-9B-Q4_0.gguf` |

GPT Terra uses the API model `gpt-5.6-terra`, with configuration ID `terra`.
The filenames and configuration IDs are recorded in
`model_comparison_registry.py`, shared by the analysis manifest and evaluator.

```bash
export MODEL_DIR='/absolute/path/to/gguf/files'
export EXPERIMENT='study_v1'
export STUDY_ROOT="$PWD/data/simulation/output/comparison/$EXPERIMENT"
export OPENAI_API_KEY='your-key'
mkdir -p "$STUDY_ROOT/logs"

"$LLAMA_SERVER" --version > "$STUDY_ROOT/logs/llama-server-version.txt" 2>&1
git -C ../llama.cpp rev-parse HEAD > "$STUDY_ROOT/logs/llama-cpp-commit.txt"
nvidia-smi > "$STUDY_ROOT/logs/gpu-before-runs.txt"
```

If using a preinstalled server, omit the `git -C` command and retain the server
version output. Use a fresh experiment ID for a repeated experiment. Keep the
same data, embeddings, prompts, retrieval settings, and output budget across
configurations. Set the variables and helper below in the same Bash session.

The runner defaults to the family tokenizers `google/gemma-4-12B-it`,
`meta-llama/Llama-3.1-8B-Instruct`, and `Qwen/Qwen3.5-9B`, respectively.
Tokenizers and the `BAAI/bge-m3` embedding model may download on first use.
Authenticate with Hugging Face if your tokenizer requires access. Use a matching
local tokenizer via `--tokenizer` in the helper if your checkpoint requires it.

## 3. Define the sequential run helper

This function starts exactly one server on port 8000, waits for it to load,
runs every configured test case and question, then stops and waits for the server.
Keep port 8000 free before starting. Server startup failures and simulation
failures return a nonzero status.

GPU offload uses **`--gpu-layers auto --fit on`**, rather than a guessed layer
count. The context window remains fixed at **32,768 tokens**, also passed to the
client. The server fits offloading to available device memory; the actual layer
count can differ by model and precision. GPU fitting also depends on other GPU
processes, so keep the machine's workload consistent across runs. The embedding
model also requires memory when the simulation runs.

After the server becomes ready, the helper prints the **actual offloaded/total
layer counts and GPU-buffer sizes from that run's startup log**. It also records
GPU memory usage through `nvidia-smi`. The reported GPU memory includes other
processes; use the server's buffer lines for its own allocations. No layer counts
or memory measurements are supplied in advance because they must come from the
machine running the experiment.

```bash
run_gguf() (
  set -euo pipefail
  profile="$1"
  precision="$2"
  filename="$3"
  name="${profile}_${precision}"
  destination="$STUDY_ROOT/$name"
  log="$STUDY_ROOT/logs/${name}.log"
  extra=()
  if [[ "${4:-full}" == smoke ]]; then
    destination="$STUDY_ROOT/smoke/$name"
    log="$STUDY_ROOT/logs/${name}_smoke.log"
    extra=(--test-case TC1 --question-limit 1)
  fi
  [[ -f "$MODEL_DIR/$filename" ]] || { echo "Missing model: $MODEL_DIR/$filename"; exit 1; }
  [[ ! -e "$destination" ]] || { echo "Choose a fresh output directory: $destination"; exit 1; }

  "$LLAMA_SERVER" -m "$MODEL_DIR/$filename" --alias "$name" \
    --host 127.0.0.1 --port 8000 -c 32768 -np 1 \
    --gpu-layers auto --fit on --jinja > "$log" 2>&1 &
  server_pid=$!
  cleanup() {
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  }
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  ready=false
  for attempt in {1..600}; do
    kill -0 "$server_pid" 2>/dev/null || { echo "Server exited. See $log"; exit 1; }
    if curl --max-time 2 -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
      ready=true
      break
    fi
    sleep 2
  done
  [[ "$ready" == true ]] || { echo "Startup timed out. See $log"; exit 1; }

  echo "=== $name: measured startup offload and buffers ==="
  grep -Ei 'offload.*layer|CUDA.*buffer|Vulkan.*buffer|GPU.*buffer' "$log" \
    || echo "Summary fields unavailable in this build; inspect the full log: $log"
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv \
    | tee "${log%.log}_gpu.csv"

  python run_open_weight_simulation.py --profile "$profile" \
    --model "$name" --api-base http://127.0.0.1:8000/v1 \
    --context-window 32768 --max-output-tokens 2048 \
    --temperature 0.6 --retrieval-temperature 0.3 \
    --similarity-top-k 2 --max-tool-use 1 \
    --output-dir "$destination" \
    --checkpoint-label "$filename; gpu_layers=auto; fit=on; startup_log=$log" \
    "${extra[@]}"
  # The EXIT trap stops the server and waits before this function returns.
)
```

Optional one-question trial before full runs:

```bash
run_gguf gemma4_12b Q4_0 gemma-4-12B-it-Q4_0.gguf smoke
```

Its outputs go under `smoke/` and are excluded from the full-study notebook
manifest. Inspect the answer and server log before beginning full experiments.
The server must use the checkpoint's native chat template and return the final
answer in the chat response content. Preserve any model-specific template or
thinking settings consistently across a family's precisions.

## 4. Run Gemma, then Llama, then Qwen

Execute each block once, in order. The `&&` chaining stops the family block
if any precision fails. Do not start the next family until the preceding block
has finished successfully. Only the server process runs in the background;
the simulation call blocks until that configuration is finished.

### Gemma: all three precisions

```bash
run_gguf gemma4_12b BF16 gemma-4-12B-it-BF16.gguf &&
run_gguf gemma4_12b Q8_0 gemma-4-12B-it-Q8_0.gguf &&
run_gguf gemma4_12b Q4_0 gemma-4-12B-it-Q4_0.gguf
```

### Llama: all three precisions

```bash
run_gguf llama31_8b BF16 Llama-3.1-8B-Instruct-BF16.gguf &&
run_gguf llama31_8b Q8_0 Meta-Llama-3.1-8B-Instruct-Q8_0.gguf &&
run_gguf llama31_8b Q4_0 Llama-3.1-8B-Instruct-Q4_0.gguf
```

### Qwen: all three precisions

```bash
run_gguf qwen35_9b BF16 Qwen3.5-9B-BF16.gguf &&
run_gguf qwen35_9b Q8_0 Qwen3.5-9B-Q8_0.gguf &&
run_gguf qwen35_9b Q4_0 Qwen3.5-9B-Q4_0.gguf
```

## 5. Run GPT Terra

After the nine GGUF runs complete:

```bash
python run_openai_simulation.py --model gpt-5.6-terra \
  --reasoning-effort none --max-output-tokens 2048 \
  --temperature 0.6 --retrieval-temperature 0.3 \
  --similarity-top-k 2 --max-tool-use 1 --output-dir "$STUDY_ROOT/terra"
```

This command requires API access to that model ID. If your API rejects
`temperature`, add `--disable-temperature` and retain that difference in the
run metadata. Do not rerun into a nonempty simulation directory.

## 6. Generate the notebook's evaluation inputs

After all ten simulations finish, evaluate them sequentially with the same judge:

```bash
python evaluate_model_comparison.py --experiment "$EXPERIMENT" \
  --model gpt-5.6-terra --temperature 0 --max-output-tokens 600
```

This uses `OPENAI_API_KEY` and makes paid judge calls. Each configuration receives
its own evaluation files. The script stops on an evaluation error and preserves
completed results. Existing evaluation folders require explicit `--overwrite`;
use repeatable `--configuration` options to select unfinished configurations on
restart. A batch report is saved in the experiment directory.

The completed experiment has this layout:

```text
data/simulation/output/comparison/study_v1/
  logs/
  gemma4_12b_BF16/
    RUN_CONFIG.json
    TC_OUTPUT_TC1.json
    ...
    evaluation/
      TC_EVAL_TC1.json
      ...
      EVAL_SUMMARY.json
  gemma4_12b_Q8_0/
  gemma4_12b_Q4_0/
  llama31_8b_BF16/
  llama31_8b_Q8_0/
  llama31_8b_Q4_0/
  qwen35_9b_BF16/
  qwen35_9b_Q8_0/
  qwen35_9b_Q4_0/
  terra/
  BATCH_EVAL_<UTC timestamp>.json
```

Each configuration directory has the same result/evaluation structure.

## 7. Process the experiment in the notebook

```bash
python -m jupyterlab notebooks/inference_evaluation_updated_models.ipynb
```

Set `EXPERIMENT = 'study_v1'` in the notebook to match the shell's `EXPERIMENT`,
then run all cells. The notebook reads the ten configuration directories and
exports tables and PNG/SVG charts to:

```text
data/simulation/analysis/study_v1/
```

Review its coverage audit before comparing the charts. Latency includes the
workflow and serving effects; token counts are client estimates. Terra is both
a compared model and the judge, which may affect interpretation of quality rates.
