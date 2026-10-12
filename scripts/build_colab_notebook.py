"""Generate tree_cot_rt_colab.ipynb.

Colab differs from Kaggle in four ways that matter here, and the notebook is
shaped around them rather than being a port of the Kaggle one:

  * there is Internet, so the backbone is pulled from Hugging Face instead of
    being attached, and the offline wheelhouse becomes optional;
  * there is no persistent input mount, so the bundle comes from Drive;
  * /content is wiped between sessions, so outputs live on Drive and S1-a
    resumes by itself rather than through a manual RESUME_FROM;
  * a session can be reclaimed at any time, so the wall-clock budget matters
    more, not less.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


md(r"""
# Tree-CoT-RT — ASQP on Colab

Runs the paper's four steps (S1-a chain generation → S1-b tSFT → S2 GRPO and
selection → S3 fine-tuning → S4 self-consistent inference).

## Setup, once

1. Build the bundle locally and upload **`kaggle_offline_upload.zip`** to your
   Drive, anywhere under `MyDrive`:

   ```
   python scripts/prepare_offline_bundle.py \
       --subset-ids ../s2it/kaggle_offline_upload/offline_bundle/subset_ids.json
   ```

   (The name is historical — the same archive serves Colab.)

2. **Runtime → Change runtime type → GPU.** Which GPU decides what runs:

   | GPU | VRAM | LoRA rank 32 (the paper's) |
   |---|---|---|
   | T4 (free tier) | 15 GB | **does not fit** — a 7B in fp16 is already 14.2 GB |
   | L4 | 22 GB | fits |
   | A100 | 40 GB | fits comfortably |

   On a T4, `train()` stops before loading any weights and says so. The
   workaround is `LORA_RANK_OVERRIDE = 8` below, which departs from the paper
   and has to be recorded alongside the results.

3. Run all cells.

## Closing the laptop

Colab runs on Google's machine, not yours, so the network link is what is at
risk, not the computation. What survives depends on the tier:

| | closing the tab | idle timeout | hard limit |
|---|---|---|---|
| Free | runtime continues a while, then is reclaimed | ~90 min | ~12 h |
| Pro | same | longer | longer |
| Pro+ | **background execution**: officially supported | - | longer |

Only Pro+ advertises background execution. On the other tiers, closing the
browser means the session is on borrowed time, and the keep-alive scripts that
circulate are both unreliable and against Colab's terms.

**This notebook is built for the session to die.** Outputs are on Drive and
the cache is replaced atomically, so an interrupted run costs the phase in
progress, never the chains already generated:

- S1-a chains land in `stage1_candidates.jsonl` on Drive every 25 typed
  instances, written beside the target and renamed, so the file on disk is
  always complete.
- Re-running the notebook reuses them with nothing to configure. The
  generation stage prints how much of the cache it can reuse before it
  generates anything.
- Lower `SESSION_BUDGET_HOURS` to match your tier so phases stop on their own
  terms rather than being cut off.

So the practical answer is to let it be interrupted and simply run it again,
rather than trying to keep the browser alive.
""")

code(r'''
# ================================ CONFIG ==================================
import sys
import time

# Colab streams cell output, but third-party libraries writing to a redirected
# stream still block-buffer. Line buffering keeps long phases visible.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

SESSION_START = time.perf_counter()

# --- Wall-clock budget -----------------------------------------------------
# Colab reclaims sessions without warning, so every phase stops at a deadline
# and whatever it produced is already on Drive.
SESSION_BUDGET_HOURS      = 11.0   # free tier is usually shorter; lower it
INFERENCE_RESERVE_MINUTES = 90
SAFETY_MARGIN_MINUTES     = 15

# --- Paths -----------------------------------------------------------------
DRIVE_ROOT  = "/content/drive/MyDrive"
BUNDLE_ZIP  = None   # auto-discovered under DRIVE_ROOT; set to override
# Outputs live on Drive so a lost session does not lose the S1-a chains.
OUTPUT_DIR  = f"{DRIVE_ROOT}/tree_cot_rt_outputs"
WORK_DIR    = "/content/tree_bundle"

# --- Model -----------------------------------------------------------------
# Colab has Internet, so the backbone comes from the Hub.
MODEL_REPO  = "Qwen/Qwen2.5-7B-Instruct"
# Where the ~15 GB of weights land. /content is wiped between sessions but is
# far faster than Drive both to write and to read, and the download runs at
# roughly 150 MB/s, so re-fetching beats keeping a copy on Drive. Point this
# at a Drive path only if your connection makes the download painful.
MODEL_DIR   = "/content/qwen2.5-7b-instruct"
# Set to a directory you already have to skip the download entirely.
LOCAL_MODEL_DIR = None

# --- Scale (paper values; see the table above before lowering) -------------
MAX_TRAIN_RECORDS = None   # use what the bundle contains
MAX_TEST_RECORDS  = None
CHAINS_PER_TYPE   = 10     # = paper K
GROUP_SIZE        = 10     # = paper G
GRPO_STEPS        = 10     # = paper
INFERENCE_SAMPLES = 5      # = paper M
MAX_NEW_TOKENS    = 384    # a cap, not a target: real chains stop near 200

# --- Fine-tuning -----------------------------------------------------------
SELECTED_CHAINS_PER_TYPE = 2
FINAL_STEPS              = 10    # = paper
BATCH_SIZE               = 1     # paper 4, but 1 x 8 == 4 x 2 effective
GRADIENT_ACCUMULATION    = 8     # effective batch 8 = paper
GRPO_BATCH_SIZE          = 2     # = paper
GRPO_GRADIENT_ACCUM      = 2     # = paper, 2 x 2 = 4 prompts per GRPO step
CLIP_EPSILON             = 0.2   # = paper
KL_COEFFICIENT           = 0.01  # = paper
LEARNING_RATE            = 5e-5  # = paper
MAX_LENGTH               = 2048  # = paper
SEED                     = 444   # paper seeds: 444, 555, 666, 777

# Set to 8 only if the GPU cannot hold the paper's rank 32, and record it.
LORA_RANK_OVERRIDE = None

# Install the bundled transformers/peft set rather than whatever Colab ships.
# Kaggle's newer stack broke two runs at the Trainer API; this is the
# combination the code was written against.
PIN_BUNDLED_VERSIONS = True

RUN_TRAINING  = True
RUN_INFERENCE = True

SESSION_BUDGET_SECONDS = SESSION_BUDGET_HOURS * 3600
INFERENCE_RESERVE_SECONDS = INFERENCE_RESERVE_MINUTES * 60
SAFETY_MARGIN_SECONDS = SAFETY_MARGIN_MINUTES * 60


def session_remaining():
    """Seconds left before the session budget runs out."""
    return SESSION_BUDGET_SECONDS - SAFETY_MARGIN_SECONDS - (time.perf_counter() - SESSION_START)


print(f"Session budget: {SESSION_BUDGET_HOURS:.1f} h "
      f"(reserving {INFERENCE_RESERVE_MINUTES} min for S4, "
      f"{SAFETY_MARGIN_MINUTES} min safety margin)")
''')

code(r'''
# ========================= DRIVE, BUNDLE, GPU ==============================
import os
import subprocess
from pathlib import Path
from zipfile import ZipFile

import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU. Runtime -> Change runtime type -> Hardware accelerator -> GPU."
    )
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    print(f"GPU {index}: {properties.name}, {properties.total_memory / 2**30:.1f} GiB")

from google.colab import drive

drive.mount("/content/drive")

drive_root = Path(DRIVE_ROOT)
if not drive_root.is_dir():
    raise FileNotFoundError(f"{drive_root} not found after mounting Drive.")

archive = Path(BUNDLE_ZIP) if BUNDLE_ZIP else None
if archive is None:
    matches = sorted(drive_root.rglob("kaggle_offline_upload.zip"))
    if not matches:
        raise FileNotFoundError(
            f"No kaggle_offline_upload.zip under {drive_root}. Upload the archive "
            "built by scripts/prepare_offline_bundle.py, or set BUNDLE_ZIP."
        )
    if len(matches) > 1:
        print("Several archives found; using the newest:")
        for m in matches:
            print("   ", m)
        matches.sort(key=lambda p: p.stat().st_mtime)
    archive = matches[-1]

# Unpack to local disk: /content is far faster than Drive for repeated reads.
work = Path(WORK_DIR)
if not (work / "project").is_dir():
    work.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as zf:
        zf.extractall(work)
print("Bundle   :", archive)

PROJECT_DIR = str(work / "project")
DATA_DIR = str(work / "offline_bundle")
WHEELHOUSE = str(Path(DATA_DIR) / "wheelhouse")
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

for required in (
    Path(PROJECT_DIR) / "src" / "tree_cot_rt" / "training.py",
    Path(DATA_DIR) / "train.jsonl",
    Path(DATA_DIR) / "dev.jsonl",
    Path(DATA_DIR) / "test.jsonl",
):
    if not required.exists():
        raise FileNotFoundError(f"Missing from the bundle: {required}")

sys.path.insert(0, str(Path(PROJECT_DIR) / "src"))
print("Project  :", PROJECT_DIR)
print("Data     :", DATA_DIR)
print("Output   :", OUTPUT_DIR, "(on Drive, so S1-a survives a lost session)")
''')

code(r'''
# ====================== DEPENDENCIES AND IMPORTS ===========================
import importlib

# peft >= 0.19 raises instead of returning False when torchao is installed but
# older than it wants. Nothing here uses torchao quantisation.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"],
               check=False)

PINS = ("transformers==4.48.3", "tokenizers==0.21.0", "accelerate==1.15.0",
        "peft==0.14.0", "huggingface-hub==0.28.1", "safetensors==0.6.2")

if PIN_BUNDLED_VERSIONS:
    bundled = sorted(Path(WHEELHOUSE).glob("*.whl"))
    if bundled:
        print("Pinning from the bundled wheelhouse")
        command = [sys.executable, "-m", "pip", "install", "-q",
                   "--no-index", "--no-deps", "--force-reinstall",
                   *[str(w) for w in bundled]]
    else:
        # Colab has Internet, so the same pins can come from PyPI.
        print("No wheelhouse in the bundle; installing the same pins from PyPI")
        command = [sys.executable, "-m", "pip", "install", "-q", *PINS]
    subprocess.check_call(command)
    importlib.invalidate_caches()

for name in ("transformers", "tokenizers", "accelerate", "peft",
             "huggingface_hub", "safetensors"):
    importlib.import_module(name)

# Belt and braces, if torchao could not be removed.
try:
    import peft.tuners.lora.torchao as _lora_torchao
    _lora_torchao.is_torchao_available = lambda: False
except Exception:
    pass

import json
import platform

import pandas as pd
from IPython.display import display

from tree_cot_rt.cli import exact_f1
from tree_cot_rt.data import read_jsonl, write_jsonl
from tree_cot_rt.reasoning import typed_instances
from tree_cot_rt.timing import format_duration
from tree_cot_rt.training import check_gpu_memory, predict, train

import transformers, peft, accelerate
print("python      :", platform.python_version())
print("torch       :", torch.__version__, "| CUDA:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("peft        :", peft.__version__, "| accelerate:", accelerate.__version__)
''')

code(r'''
# ========================= DOWNLOAD THE BACKBONE ===========================
import shutil

from huggingface_hub import snapshot_download

if LOCAL_MODEL_DIR:
    BASE_MODEL = str(LOCAL_MODEL_DIR)
    print("Using the model already at", BASE_MODEL)
else:
    target = Path(MODEL_DIR)
    weights = sorted(target.glob("*.safetensors"))
    if (target / "config.json").is_file() and weights:
        have = sum(f.stat().st_size for f in target.iterdir() if f.is_file()) / 2**30
        print(f"Already downloaded: {target} ({have:.1f} GiB, {len(weights)} shards)")
    else:
        free = shutil.disk_usage(target.parent if target.parent.exists() else "/content").free / 2**30
        print(f"Free disk: {free:.1f} GiB")
        if free < 20:
            raise RuntimeError(
                f"Only {free:.1f} GiB free; the weights need about 16 GiB plus room "
                "to unpack. Point MODEL_DIR at a larger volume."
            )
        print(f"Downloading {MODEL_REPO} (~15 GiB). This is the longest step of the "
              "setup; snapshot_download resumes if the session drops.")
        started = time.perf_counter()
        snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=str(target),
            # Weights, config and tokenizer only. The repo also carries GGUF and
            # other formats that would double the download for nothing.
            allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*"],
            ignore_patterns=["original/*", "*.gguf", "*.bin"],
            max_workers=8,
        )
        size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 2**30
        print(f"Downloaded {size:.1f} GiB in {format_duration(time.perf_counter() - started)}")
    BASE_MODEL = str(target)

for required in ("config.json", "tokenizer_config.json"):
    if not (Path(BASE_MODEL) / required).is_file():
        raise FileNotFoundError(f"{required} missing from {BASE_MODEL}")
print("Model    :", BASE_MODEL)

# Say now whether the paper's LoRA rank fits this GPU, before any weights are
# read, rather than minutes later inside Trainer with a CUDA OOM traceback.
check_gpu_memory(BASE_MODEL, LORA_RANK_OVERRIDE or 32)
''')

md(r"""
## 1. Data and ASQP type coverage

S1-a samples `CHAINS_PER_TYPE` chains for each **typed instance**, so that
count, not the sentence count, is what the cost scales with.
""")

code(r'''
train_records = read_jsonl(Path(DATA_DIR) / "train.jsonl")
dev_records = read_jsonl(Path(DATA_DIR) / "dev.jsonl")
test_records = read_jsonl(Path(DATA_DIR) / "test.jsonl")
if MAX_TRAIN_RECORDS is not None:
    train_records = train_records[:MAX_TRAIN_RECORDS]
if MAX_TEST_RECORDS is not None:
    test_records = test_records[:MAX_TEST_RECORDS]

typed_rows = [t for r in train_records for t in typed_instances(r)]
display(pd.DataFrame([
    {"Split": "train", "Sentences": len(train_records), "Typed instances": len(typed_rows)},
    {"Split": "dev", "Sentences": len(dev_records), "Typed instances": ""},
    {"Split": "test", "Sentences": len(test_records), "Typed instances": ""},
]))
display(pd.DataFrame([
    {"ASQP type": name,
     "Typed instances": sum(1 for t in typed_rows if t.quad_type == name)}
    for name in ("EAEO", "EAIO", "IAEO", "IAIO")
]))
''')

md(r"""
## 2. S1 → S2 → S3

Each phase gets a slice of the remaining budget, sized by how much work it has,
and stops at its deadline. S1-a degrades gracefully: the chains it produced
become the candidate pool and everything downstream still runs.

Because `OUTPUT_DIR` is on Drive, re-running this cell in a later session
reuses the cached chains. Watch the `Cache:` line — it reports how many typed
instances the cache already covers before anything is generated.
""")

code(r'''
if RUN_TRAINING:
    train_budget = session_remaining() - INFERENCE_RESERVE_SECONDS
    if train_budget <= 0:
        raise RuntimeError("No time left for training; raise SESSION_BUDGET_HOURS.")
    print(f"Training budget: {format_duration(train_budget)}")

    training_metrics = train(
        train_records,
        dev_records,
        OUTPUT_DIR,
        model_name=BASE_MODEL,
        final_model_name=BASE_MODEL,
        chains_per_type=CHAINS_PER_TYPE,
        grpo_steps=GRPO_STEPS,
        group_size=GROUP_SIZE,
        grpo_batch_size=GRPO_BATCH_SIZE,
        grpo_gradient_accumulation=GRPO_GRADIENT_ACCUM,
        clip_epsilon=CLIP_EPSILON,
        kl_coefficient=KL_COEFFICIENT,
        selected_chains_per_type=SELECTED_CHAINS_PER_TYPE,
        final_steps=FINAL_STEPS,
        batch_size=BATCH_SIZE,
        gradient_accumulation=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        max_length=MAX_LENGTH,
        max_new_tokens=MAX_NEW_TOKENS,
        seed=SEED,
        time_budget_seconds=train_budget,
    )
else:
    metrics_path = Path(OUTPUT_DIR) / "training_metrics.json"
    training_metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else None
    print("Training skipped; using the adapter already in", Path(OUTPUT_DIR) / "final")

if training_metrics:
    stage_seconds = training_metrics.get("stage_seconds", {})
    timing_table = pd.DataFrame([
        {"Phase": "S1-a multi-path chain generation", "Seconds": stage_seconds.get("S1a_generation", 0.0)},
        {"Phase": "S1-b tree-supervised fine-tuning", "Seconds": stage_seconds.get("S1b_tsft", 0.0)},
        {"Phase": "S2 GRPO and chain selection", "Seconds": stage_seconds.get("S2_grpo_and_selection", 0.0)},
        {"Phase": "S3 reasoning-aware fine-tuning", "Seconds": stage_seconds.get("S3_final_sft", 0.0)},
        {"Phase": "Total", "Seconds": training_metrics.get("training_seconds", 0.0)},
    ])
    timing_table["Time"] = timing_table["Seconds"].map(format_duration)
    timing_table["Seconds"] = timing_table["Seconds"].round(2)
    display(timing_table)
    timing_table.to_csv(Path(OUTPUT_DIR) / "training_time.csv", index=False)

    print("Generated chains:", training_metrics.get("generated_chains"))
    print("Selected chains :", training_metrics.get("selected_chains"))
    truncated = training_metrics.get("budget", {}).get("truncated_phases", [])
    if truncated:
        print("Stopped early to stay inside the budget:", ", ".join(truncated))
        print("Re-run this cell in a new session; the cached chains are on Drive.")
    print("Session time left:", format_duration(max(0.0, session_remaining())))
''')

md(r"""
## 3. S4 self-consistent prediction

The ASQP type is unknown at test time, so each sentence gets one type-agnostic
prompt, `INFERENCE_SAMPLES` chains by stochastic decoding, and a majority vote
over the quadruples (Eq. 9). Sentences left unscored at the deadline are
reported with an empty quadruple list, so F1 is measured over the whole split.
""")

code(r'''
if RUN_INFERENCE:
    inference_budget = max(60.0, session_remaining())
    print(f"Inference budget: {format_duration(inference_budget)} "
          f"for {len(test_records)} sentences")

    inference_started = time.perf_counter()
    predictions = predict(
        test_records,
        Path(OUTPUT_DIR) / "final",
        base_model=BASE_MODEL,
        samples=INFERENCE_SAMPLES,
        max_new_tokens=MAX_NEW_TOKENS,
        time_budget_seconds=inference_budget,
    )
    inference_seconds = time.perf_counter() - inference_started

    prediction_path = Path(OUTPUT_DIR) / "test_predictions.jsonl"
    write_jsonl(prediction_path, predictions)
    evaluation = exact_f1(
        {r.instance_id: [q.as_dict() for q in r.quads] for r in test_records},
        {r["instance_id"]: r["quadruples"] for r in predictions},
    )
    (Path(OUTPUT_DIR) / "evaluation_metrics.json").write_text(
        json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")

    unscored = sum(1 for r in predictions if not r["reasoning_samples"])
    results_table = pd.DataFrame([
        {"Metric": "Precision", "Value": round(evaluation["precision"], 4)},
        {"Metric": "Recall", "Value": round(evaluation["recall"], 4)},
        {"Metric": "Exact quadruple micro-F1", "Value": round(evaluation["f1"], 4)},
        {"Metric": "True positive quadruples", "Value": evaluation["true_positives"]},
        {"Metric": "Predicted quadruples", "Value": evaluation["predicted"]},
        {"Metric": "Gold quadruples", "Value": evaluation["gold"]},
        {"Metric": "Sentences evaluated", "Value": len(test_records)},
        {"Metric": "Sentences left unscored", "Value": unscored},
        {"Metric": "Inference time", "Value": format_duration(inference_seconds)},
    ])
    display(results_table)
    results_table.to_csv(Path(OUTPUT_DIR) / "evaluation_metrics.csv", index=False)
    print("Session time left:", format_duration(max(0.0, session_remaining())))
else:
    print("Inference skipped. Set RUN_INFERENCE = True.")
''')

md(r"""
## Outputs

Everything lands in `MyDrive/tree_cot_rt_outputs/` and survives the session:

- `final/` — the S3 LoRA adapter, tokenizer and `categories.json`
- `stage1_candidates.jsonl` — the S1-a pool; **this is what makes resuming cheap**
- `stage2_selected_chains.jsonl` — the chains GRPO kept
- `training_metrics.json` — per-phase timing, GRPO losses, reward weights, budget report
- `test_predictions.jsonl`, `evaluation_metrics.csv`

### Resuming

Just run the notebook again. Nothing to configure: `OUTPUT_DIR` already points
at Drive, so the cached chains are found and only the uncovered typed
instances are generated. To skip training entirely and only evaluate, set
`RUN_TRAINING = False`.

### A caveat about this dataset

The training split contains no IAIO instances at all (and few implicit cases
of any kind). Tree-CoT-RT is a paper about implicit sentiment, and IAIO is
where it reports its largest gains, so that contribution cannot show up here.
The run is still valid as a baseline; the limitation belongs in the write-up.
""")

notebook = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

target = REPO / "tree_cot_rt_colab.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {target} ({len(CELLS)} cells)")
