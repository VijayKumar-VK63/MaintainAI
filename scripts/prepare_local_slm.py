"""Prepare CPU-friendly GGUF assets for local SLM evaluation.

Downloads a quantized Qwen2.5-3B base model and converts the PEFT LoRA
adapter to GGUF, so base vs fine-tuned evaluation can run on a CPU-only
machine through llama.cpp.

Prerequisites (one-time):
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
    pip install gguf sentencepiece
    git

Usage:
    python scripts/prepare_local_slm.py
    python scripts/prepare_local_slm.py --force
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = ROOT / "models" / "slm"
DEFAULT_BASE_REPO = "bartowski/Qwen2.5-3B-Instruct-GGUF"
DEFAULT_BASE_FILE = "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
DEFAULT_ADAPTER = ROOT / "slm" / "checkpoints" / "final"
DEFAULT_LORA_FILE = "maintainai-qwen2.5-3b-lora.gguf"


def _require(module: str, hint: str) -> None:
    if importlib.util.find_spec(module) is None:
        raise SystemExit(f"missing '{module}'. {hint}")


def ensure_llama_cpp(dest: Path) -> None:
    if (dest / "convert_lora_to_gguf.py").exists():
        return
    if shutil.which("git") is None:
        raise SystemExit("git is required to fetch the llama.cpp converter")
    print(f"Cloning llama.cpp -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/ggerganov/llama.cpp.git",
            str(dest),
        ],
        check=True,
    )


def download_base(repo: str, filename: str, outdir: Path, force: bool = False) -> Path:
    target = outdir / filename
    if target.exists() and not force and target.stat().st_size > 0:
        print(f"Using existing {target}")
        return target
    from huggingface_hub import hf_hub_download

    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            path = hf_hub_download(
                repo_id=repo, filename=filename, local_dir=str(outdir)
            )
            print(f"Downloaded {path}")
            return Path(path)
        except Exception as exc:  # noqa: BLE001 - network flake, retry
            last_error = exc
            print(
                f"download attempt {attempt} failed: "
                f"{type(exc).__name__}; retrying"
            )
            time.sleep(3)
    raise SystemExit(f"download failed: {last_error}")


def convert_lora(
    llama_dir: Path, adapter: Path, outfile: Path, force: bool = False
) -> Path:
    if outfile.exists() and not force:
        print(f"Using existing {outfile}")
        return outfile
    cmd = [
        sys.executable,
        str(llama_dir / "convert_lora_to_gguf.py"),
        str(adapter),
        "--outfile",
        str(outfile),
        "--outtype",
        "f16",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return outfile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--base-repo", default=DEFAULT_BASE_REPO)
    ap.add_argument("--base-file", default=DEFAULT_BASE_FILE)
    ap.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    ap.add_argument("--lora-file", default=DEFAULT_LORA_FILE)
    ap.add_argument("--llama-cpp-dir", default=str(DEFAULT_OUTDIR / "llama.cpp"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-convert", action="store_true")
    args = ap.parse_args()

    _require(
        "llama_cpp",
        "pip install llama-cpp-python --extra-index-url "
        "https://abetlen.github.io/llama-cpp-python/whl/cpu",
    )
    _require("gguf", "pip install gguf")
    _require("sentencepiece", "pip install sentencepiece")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    adapter = Path(args.adapter)
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"no PEFT adapter found at {adapter}")

    base = outdir / args.base_file
    if not args.skip_download:
        base = download_base(args.base_repo, args.base_file, outdir, args.force)

    lora = outdir / args.lora_file
    if not args.skip_convert:
        llama_dir = Path(args.llama_cpp_dir)
        ensure_llama_cpp(llama_dir)
        convert_lora(llama_dir, adapter, lora, args.force)

    print("\nLocal assets ready:")
    print(f"  base GGUF: {base}")
    print(f"  LoRA GGUF: {lora}")
    print("\nRun base evaluation:")
    print(
        f'  python -m src.slm_eval --backend llamacpp --gguf "{base}" '
        "--constrained --out reports/metrics_base.json"
    )
    print("\nRun fine-tuned evaluation:")
    print(
        f'  python -m src.slm_eval --backend llamacpp --gguf "{base}" '
        f'--adapter "{lora}" --constrained --out reports/metrics_finetuned.json'
    )
    print("\nThen: python scripts/compare_slm.py")


if __name__ == "__main__":
    main()
