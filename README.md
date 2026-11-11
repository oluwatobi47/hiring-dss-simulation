# Hiring DSS simulation

For setup, all simulation commands, model mappings, and evaluation instructions,
see the dedicated [simulation runbook](SIMULATION_README.md).

The comparison includes Gemma 4 12B IT, Llama 3.1 8B Instruct, and Qwen3.5 9B,
each at **BF16, Q8_0, and Q4_0**, plus **gpt-5.6-terra**.

- [Model registry](model_comparison_registry.py): screenshot filenames, family tokenizers, and configuration IDs.
- [Batch evaluator](evaluate_model_comparison.py): generate isolated evaluation results for all or selected configurations.
- [Comparison notebook](notebooks/inference_evaluation_updated_models.ipynb): quality, latency, quantization, radar charts, and exports.
- [General open-weight guide](OPEN_WEIGHT_SIMULATIONS.md): other server-client profiles.

The comparison runners use an OpenAI-compatible local server for GGUF inference.
Terra inference and evaluation judging use the OpenAI API. Historical Llama-only
runner/downloader defaults refer to different checkpoint filenames; consult the
runbook before using those scripts for this model matrix.
