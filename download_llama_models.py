#!/usr/bin/env python3
"""Download the GGUF Llama models used by the local simulation runner."""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = REPO_ROOT / "model"
DEFAULT_REPO_ID = "oluwatobi-alao/gguf_llama_models"

MODEL_FILES = {
    "8b": "llama31_8b_hiring_fp16.gguf",
    "8b_Q4": "llama31_8b_hiring_Q4_K_M.gguf",
    "8b_Q8": "llama31_8b_hiring_Q8_0.gguf",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download GGUF Llama models for the local hiring DSS simulation.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repo ID containing the GGUF files.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Destination folder. Defaults to <repo>/model.")
    parser.add_argument(
        "--model",
        action="append",
        choices=sorted(MODEL_FILES),
        help="Model key to download. Repeat to download multiple. Defaults to all known simulation models.",
    )
    parser.add_argument(
        "--filename",
        action="append",
        help="Explicit filename to download from the repo. Can be repeated. Skips --model defaults when provided.",
    )
    parser.add_argument("--revision", default=None, help="Optional Hugging Face repo revision, branch, or commit.")
    parser.add_argument("--token", default=None, help="Hugging Face token. Defaults to cached login or HF_TOKEN.")
    parser.add_argument("--login", action="store_true", help="Persist --token with huggingface_hub.login before downloading.")
    parser.add_argument("--force-download", action="store_true", help="Force re-download even if cached locally.")
    return parser


def resolve_filenames(args: argparse.Namespace) -> list[str]:
    if args.filename:
        return args.filename
    selected_models = args.model or sorted(MODEL_FILES)
    return [MODEL_FILES[model] for model in selected_models]


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download, login
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install with "
            "`pip install -r notebooks/local_simulation/scripts/requirements.txt`."
        ) from exc

    model_dir = args.model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.login:
        if not args.token:
            raise ValueError("--login requires --token")
        login(token=args.token)

    downloaded_paths = []
    for filename in resolve_filenames(args):
        print(f"Downloading {filename} from {args.repo_id} to {model_dir}")
        path = hf_hub_download(
            repo_id=args.repo_id,
            filename=filename,
            local_dir=model_dir,
            revision=args.revision,
            token=args.token,
            force_download=args.force_download,
        )
        downloaded_paths.append(path)
        print(f"Downloaded: {path}")

    print("Model download complete.")
    for path in downloaded_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
