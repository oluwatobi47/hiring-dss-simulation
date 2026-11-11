"""Screenshot-based study mapping; filenames do not verify checkpoint contents."""
from pathlib import Path

TERRA_MODEL = "gpt-5.6-terra"
FAMILIES = {"gemma4_12b": "Gemma 4 12B", "llama31_8b": "Llama 3.1 8B", "qwen35_9b": "Qwen3.5 9B"}
TOKENIZERS = {
    "gemma4_12b": "google/gemma-4-12B-it",
    "llama31_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen35_9b": "Qwen/Qwen3.5-9B",
}
GGUF_FILES = {
    "gemma4_12b": {
        "BF16": "gemma-4-12B-it-BF16.gguf",
        "Q8_0": "gemma-4-12B-it-Q8_0.gguf",
        "Q4_0": "gemma-4-12B-it-Q4_0.gguf",
    },
    "llama31_8b": {
        "BF16": "Llama-3.1-8B-Instruct-BF16.gguf",
        "Q8_0": "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
        "Q4_0": "Llama-3.1-8B-Instruct-Q4_0.gguf",
    },
    "qwen35_9b": {
        "BF16": "Qwen3.5-9B-BF16.gguf",
        "Q8_0": "Qwen3.5-9B-Q8_0.gguf",
        "Q4_0": "Qwen3.5-9B-Q4_0.gguf",
    },
}
CONFIGURATIONS = {
    f"{profile}_{precision}": dict(configuration=f"{profile}_{precision}", profile=profile,
        family=FAMILIES[profile], precision=precision, filename=filename, tokenizer=TOKENIZERS[profile])
    for profile, variants in GGUF_FILES.items() for precision, filename in variants.items()
}
CONFIGURATIONS["terra"] = dict(configuration="terra", profile=None, family="GPT Terra",
                                precision="API", filename=None, tokenizer=None, model=TERRA_MODEL)


def comparison_manifest(output_root, experiment):
    """Return fresh entries, preserving the notebook's configuration IDs and paths."""
    if not experiment or experiment in {".", ".."} or any(c in experiment for c in "/\\"):
        raise ValueError("Experiment must be a single directory name.")
    base = Path(output_root) / experiment
    return [dict(spec, run=experiment, results=base / name, evaluation=base / name / "evaluation")
            for name, spec in CONFIGURATIONS.items()]
