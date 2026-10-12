#!/usr/bin/env python
"""Build the offline Kaggle upload for Tree-CoT-RT on your local machine.

Kaggle runs with no Internet, so the notebook needs one uploaded archive with
everything in it:

  1. validate and copy the JSONL splits,
  2. download the Linux/py3.11 wheels Kaggle is missing,
  3. zip project source + data + wheelhouse.

Tree-CoT-RT has no dependency parser, so unlike S2IT there is no Stanza step
and no language tagging; `langdetect` is not a dependency of this project.
The Qwen model is NOT packaged: attach it separately as a Kaggle Model input.

    python scripts/prepare_offline_bundle.py --data-dir asqp_data

Produces `kaggle_offline_upload.zip`; upload it as a Kaggle Dataset.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tree_cot_rt.data import read_jsonl  # noqa: E402
from tree_cot_rt.reasoning import typed_instances  # noqa: E402

SPLITS = ("train", "dev", "test")

WHEELS = (
    "transformers==4.48.3",
    "tokenizers==0.21.0",
    "accelerate==1.15.0",
    "peft==0.14.0",
    "huggingface-hub==0.28.1",
    "safetensors==0.6.2",
)
KAGGLE_PYTHON = "3.11"
KAGGLE_PLATFORM = "manylinux2014_x86_64"

PROJECT_FILES = ("pyproject.toml", "README.md")


#: Directory names checked for train/dev/test.jsonl when --data-dir is omitted.
DATA_DIR_CANDIDATES = ("asqp_data", "converted_apc")


def default_data_dir() -> Path:
    for name in DATA_DIR_CANDIDATES:
        candidate = REPO_ROOT / name
        if all((candidate / f"{split}.jsonl").is_file() for split in SPLITS):
            return candidate
    return REPO_ROOT / DATA_DIR_CANDIDATES[0]


def log(message: str) -> None:
    print(f"==> {message}", flush=True)


def copy_data(data_dir: Path, bundle: Path) -> dict[str, int]:
    log("Step 1/3: validating and copying the JSONL splits")
    typed_counts: Counter[str] = Counter()
    sizes: dict[str, int] = {}
    for split in SPLITS:
        source = data_dir / f"{split}.jsonl"
        if not source.is_file():
            raise FileNotFoundError(f"Missing split: {source}")
        records = read_jsonl(source)  # raises on malformed records
        if not records:
            raise ValueError(f"{source}: no records")
        sizes[split] = len(records)
        if split == "train":
            for record in records:
                for typed in typed_instances(record):
                    typed_counts[typed.quad_type] += 1
        shutil.copy2(source, bundle / f"{split}.jsonl")
        print(f"    {split}: {len(records)} records")
    print(f"    train ASQP types: {dict(sorted(typed_counts.items()))}")
    (bundle / "dataset_stats.json").write_text(
        json.dumps({"splits": sizes, "train_quad_types": dict(typed_counts)}, indent=2) + "\n",
        encoding="utf-8",
    )
    return sizes


def download_wheels(wheelhouse: Path) -> None:
    log(f"Step 2/3: downloading Linux py{KAGGLE_PYTHON} wheels")
    wheelhouse.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            sys.executable, "-m", "pip", "download",
            "--dest", str(wheelhouse),
            "--only-binary=:all:",
            "--no-deps",
            "--python-version", KAGGLE_PYTHON,
            "--platform", KAGGLE_PLATFORM,
            *WHEELS,
        ]
    )
    print(f"    {len(list(wheelhouse.glob('*.whl')))} wheels in {wheelhouse}")


def copy_project(staging: Path) -> None:
    project = staging / "project"
    if project.exists():
        shutil.rmtree(project)
    (project / "src").mkdir(parents=True)
    shutil.copytree(
        REPO_ROOT / "src" / "tree_cot_rt",
        project / "src" / "tree_cot_rt",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in PROJECT_FILES:
        if (REPO_ROOT / name).is_file():
            shutil.copy2(REPO_ROOT / name, project / name)


def write_upload_readme(staging: Path, sizes: dict[str, int]) -> None:
    (staging / "UPLOAD_README.txt").write_text(
        "Tree-CoT-RT offline Kaggle bundle\n"
        "=================================\n\n"
        "Contents\n"
        "  project/          Tree-CoT-RT source; the notebook imports from project/src\n"
        "  offline_bundle/   train/dev/test JSONL and the pip wheelhouse\n\n"
        f"Records: {', '.join(f'{k}={v}' for k, v in sizes.items())}\n\n"
        "How to use on Kaggle\n"
        "  1. Upload kaggle_offline_upload.zip as a Kaggle Dataset.\n"
        "  2. Attach that Dataset to the notebook.\n"
        "  3. Attach Qwen2.5-7B-Instruct under Models (NOT in this archive).\n"
        "  4. Accelerator: GPU. Internet: OFF.\n"
        "  5. Run all cells; the config cell finds everything under /kaggle/input.\n\n"
        "The archive deliberately excludes model weights and training outputs.\n"
        "S1 chain generation is the expensive phase: the notebook's session\n"
        "budget decides how much of the training split it can cover.\n",
        encoding="utf-8",
    )


def build_zip(staging: Path, archive_path: Path, notebook: Path | None) -> None:
    log("Step 3/3: writing the upload archive")
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(staging.rglob("*")):
            if path.is_dir() or "__pycache__" in path.parts:
                continue
            # Forward slashes so the archive unpacks correctly on Kaggle/Linux.
            archive.write(path, path.relative_to(staging).as_posix())
        if notebook is not None and notebook.is_file():
            archive.write(notebook, notebook.name)
    size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"    {archive_path} ({size_mb:.1f} MB)")


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help=f"train/dev/test.jsonl location (auto: {' or '.join(DATA_DIR_CANDIDATES)})",
    )
    cli.add_argument("--staging-dir", type=Path, default=REPO_ROOT / "kaggle_offline_upload")
    cli.add_argument("--archive", type=Path, default=REPO_ROOT / "kaggle_offline_upload.zip")
    cli.add_argument("--skip-wheels", action="store_true", help="reuse an existing wheelhouse")
    args = cli.parse_args()

    staging = args.staging_dir
    bundle = staging / "offline_bundle"
    bundle.mkdir(parents=True, exist_ok=True)

    sizes = copy_data(args.data_dir, bundle)
    if not args.skip_wheels:
        download_wheels(bundle / "wheelhouse")
    copy_project(staging)
    write_upload_readme(staging, sizes)
    build_zip(staging, args.archive, REPO_ROOT / "tree_cot_rt_kaggle.ipynb")
    print("\nDone. Upload the archive as a Kaggle Dataset and attach Qwen2.5-7B-Instruct separately.")


if __name__ == "__main__":
    main()
