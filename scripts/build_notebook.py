"""Rebuild tree_cot_rt_kaggle.ipynb with zip-based inputs and a 9h session budget."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    CELLS.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True),
        }
    )


md(r"""
# Tree-CoT-RT — Aspect Sentiment Quad Prediction (offline Kaggle)

Runs the paper's four steps with **no Internet**, inside a **9-hour** Kaggle session.

| Step | What happens |
|---|---|
| S1-a | Sample K Tree-CoT chains per typed instance (EAEO / EAIO / IAEO / IAIO) |
| S1-b | Tree-supervised LoRA fine-tuning → the GRPO initial policy |
| S2 | GRPO with the rule-based reward, then offline top-N chain selection |
| S3 | Fine-tune on the selected chains (sentence → reasoning + quadruples) |
| S4 | Self-consistent inference: M samples per sentence, majority vote |

**Attach two inputs, then Run All:**

| Input | What |
|---|---|
| Dataset | `kaggle_offline_upload.zip`, built locally by `scripts/prepare_offline_bundle.py` |
| Model | `Qwen2.5-7B-Instruct` (already on Kaggle — *not* in the archive) |

Settings: **Accelerator = GPU**, **Internet = Off**.

## Running full data across sessions

This notebook is configured for the **whole training split**, which one
9-hour session cannot finish. Generation is the reason: 8,815 sentences
expand to 9,121 typed instances, and the method samples chains over all of
them **twice** — once in S1-a to build the candidate pool, once in S2 to
select from the optimised policy (Algorithm 3).

### Accelerator: **GPU T4 x2**

Memory decides this before speed does: Qwen2.5-7B in fp16 is 14.2 GiB
against a single card's 14.6, and S3's LoRA rank 32 needs another 0.92 GiB
with its AdamW state. Two cards shard the model and keep the paper's rank.

The note below still holds for raw decoding speed, but only applies if the
model fits at all.

### On bandwidth: P100 decodes faster than T4

Generation is memory-bandwidth bound, and Kaggle's P100 has 732 GB/s against
the T4's 320 GB/s — **2.3x faster for this workload**, for free. (For S2IT,
which is dominated by fine-tuning rather than decoding, the T4's higher
compute wins instead.)

### Why 1,530 training sentences, and how many sessions

1,530 is the size of the **Restaurant-ACOS training split**, one of the two
datasets this method was published on (Laptop-ACOS has 2,934). Training at
that scale with the paper's K, G and M is a faithful reproduction; keeping
all 8,815 sentences but cutting K to 2 is not.

The same 1,530 sentences are in the S2IT bundle and both evaluate on the
same 1,831 test sentences, so the two baselines are comparable.

Worst-case hours on a **P100**, assuming every generation runs to the
384-token cap (real chains stop at EOS near ~200 tokens, so expect less):

| phase | hours |
|---|---|
| S1-a over 1,591 typed instances | ~5.9 |
| S2 re-generation for selection | ~5.9 |
| S1-b, GRPO updates, S3 fine-tuning | ~1 |
| **training total** | **~13 h = 2 sessions** |
| S4 over 1,831 test sentences | ~6.8 h = 1 session |

So **3 P100 sessions**: two training (second one with `RESUME_FROM`), then
one with `RUN_TRAINING = False` and `RESUME_FROM` set, which spends the
whole session on inference.

**How to run it:**

1. Run this notebook. S1-a stops at its deadline and reports the coverage it
   reached; everything downstream still runs, so the session produces a
   usable adapter and metrics.
2. *Save Version* so `tree_cot_rt_outputs/` becomes an output.
3. Next session: attach that output and set `RESUME_FROM` to its
   `tree_cot_rt_outputs` path. Cached chains whose prompt still matches are
   reused, so S1-a continues from where it stopped.
4. Repeat until the coverage line reports all 9,121 typed instances.

Each session reports a projection of how many more are needed.

> Lowering `CHAINS_PER_TYPE` to 1 roughly halves the cost but leaves S1-b and
> the GRPO selection with a single candidate per typed instance, so invalid
> chains cannot be filtered out and nothing is actually being selected.
""")

code(r'''
# ============================== SESSION CONFIG ==============================
import sys
import time

# Kaggle runs this notebook under papermill, which redirects stdout to a file.
# Python then block-buffers it, so nothing reaches the log until 8 KB has
# accumulated - a run can look dead for a quarter of an hour while it is
# unpacking, installing and loading a 7B model. Line buffering makes progress
# visible as it happens.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

SESSION_START = time.perf_counter()

# --- Wall-clock budget -------------------------------------------------------
# Kaggle kills the session at the limit, which would lose the LoRA adapter.
# Every phase stops at its own deadline instead.
SESSION_BUDGET_HOURS      = 9.0   # Kaggle GPU session limit
SETUP_RESERVE_MINUTES     = 20    # wheels + model load
INFERENCE_RESERVE_MINUTES = 90    # partial S4; the full split needs its own
                                  # session (RUN_TRAINING = False + RESUME_FROM)
SAFETY_MARGIN_MINUTES     = 15

# --- Scale -------------------------------------------------------------------
# Paper values are in the right-hand column.
# Full data. S1-a cannot cover it in one session, so the run is resumable:
# see "Running full data across sessions" below. K and the chain length are
# cut hard to buy coverage, which is the trade this scale forces.
# Decoding is memory-bandwidth bound: every step reads all 14 GB of weights
# whatever K or M is, and num_return_sequences batches the K chains into that
# one read. So K and M are nearly FREE, and only the sentence count and
# MAX_NEW_TOKENS actually buy wall-clock time. All three are therefore kept at
# the paper's values.
# The bundle already carries the agreed 1,530-sentence training subset and
# the 1,831-sentence test split, shared byte-for-byte with the S2IT bundle.
MAX_TRAIN_RECORDS = None   # use what the bundle contains
MAX_TEST_RECORDS  = None   # use what the bundle contains
CHAINS_PER_TYPE   = 10     # = paper K
GROUP_SIZE        = 10     # = paper G
GRPO_STEPS        = 10     # = paper
INFERENCE_SAMPLES = 5      # = paper M
# The paper's own IAIO example chain (Table 14) measures ~201 tokens, so 384
# leaves ~90% headroom. 288 still fits it and cuts generation time by 25%.
MAX_NEW_TOKENS    = 384    # paper: 2048, which that chain never needs

# --- Fine-tuning -------------------------------------------------------------
# Table 9 (S3) and Table 8 (GRPO), reproduced. The paper's "batch 4, accum 2"
# is an effective batch of 8, and 1 x 8 is the same optimiser trajectory on a
# 16 GB card. GRPO keeps its own accumulation so it lands on the paper's
# effective 4 (2 x 2) independently of S3.
# S3's LoRA rank. The paper uses 32, which does not fit a single 16 GB card
# (a 7B in fp16 is already 14.2 GiB). Lower it to 8 only if 'GPU T4 x2' is
# unavailable, and record the deviation.
FINAL_LORA_RANK          = 32    # = paper
SELECTED_CHAINS_PER_TYPE = 2
FINAL_STEPS              = 10    # = paper
BATCH_SIZE               = 1     # paper 4, but 1 x 8 == 4 x 2 effective
GRADIENT_ACCUMULATION    = 8     # effective batch 8 = paper
GRPO_BATCH_SIZE          = 2     # = paper
GRPO_GRADIENT_ACCUM      = 2     # = paper, so 2 x 2 = 4 prompts per GRPO step
CLIP_EPSILON             = 0.2   # = paper
KL_COEFFICIENT           = 0.01  # = paper
LEARNING_RATE            = 5e-5  # = paper
MAX_LENGTH               = 2048  # = paper (costs nothing: sequences are ~525 tokens)
SEED                     = 444   # paper seeds: 444, 555, 666, 777

# Share of the training budget per phase: S1-a, S1-b, S2, S3.
STAGE_FRACTIONS = (0.40, 0.10, 0.30, 0.20)

# Install the bundled transformers/peft set instead of Kaggle's newer one.
# Kaggle's stack broke two runs at the Trainer API; the bundle is the
# combination this code was written against. False uses Kaggle's.
PIN_BUNDLED_VERSIONS = True

RUN_TRAINING  = True
RUN_INFERENCE = True

# --- Paths -------------------------------------------------------------------
OUTPUT_DIR  = "/kaggle/working/tree_cot_rt_outputs"
# Point this at an attached previous run's `tree_cot_rt_outputs` to continue
# S1-a where the last session stopped. Cached chains whose prompt still
# matches are reused, so only uncovered typed instances are generated.
RESUME_FROM = None
BASE_MODEL  = None   # auto-discovered; set to override
PROJECT_DIR = None
DATA_DIR    = None

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
# ========================= LOCATE AND UNPACK INPUTS =========================
import os, sys
from pathlib import Path
from zipfile import ZipFile

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["WANDB_DISABLED"] = "true"

input_root = Path("/kaggle/input")
if not input_root.is_dir():
    raise FileNotFoundError("/kaggle/input not found — run this notebook on Kaggle.")

# The bundle is uploaded as a single zip; unpack it into /kaggle/working once.
archive = next(iter(sorted(input_root.rglob("kaggle_offline_upload.zip"))), None)
if archive is not None:
    unpacked = Path("/kaggle/working/tree_bundle")
    if not (unpacked / "project").is_dir():
        unpacked.mkdir(parents=True, exist_ok=True)
        with ZipFile(archive) as zf:
            zf.extractall(unpacked)
    PROJECT_DIR = str(unpacked / "project")
    DATA_DIR = str(unpacked / "offline_bundle")
    print("Unpacked archive:", archive)
else:
    # Fall back to a Dataset uploaded already-extracted.
    sources = [p.parent.parent for p in input_root.rglob("tree_cot_rt")
               if p.is_dir() and (p / "training.py").is_file()]
    data = [p.parent for p in input_root.rglob("train.jsonl")
            if all((p.parent / f"{s}.jsonl").is_file() for s in ("dev", "test"))]
    if not sources or not data:
        raise FileNotFoundError(
            "Attach the Dataset built by scripts/prepare_offline_bundle.py "
            "(kaggle_offline_upload.zip, or its extracted project/ + offline_bundle/)."
        )
    PROJECT_DIR, DATA_DIR = str(sources[0]), str(data[0])

if BASE_MODEL is None:
    configs = [p.parent for p in input_root.rglob("config.json") if "qwen" in str(p.parent).lower()]
    if not configs:
        raise FileNotFoundError("Attach Qwen2.5-7B-Instruct under Models, or set BASE_MODEL.")
    BASE_MODEL = str(sorted(configs, key=lambda p: ("7b" not in str(p).lower(), len(str(p))))[0])

WHEELHOUSE = str(Path(DATA_DIR) / "wheelhouse")
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

for required in (
    Path(PROJECT_DIR) / "src" / "tree_cot_rt" / "training.py",
    Path(DATA_DIR) / "train.jsonl",
    Path(DATA_DIR) / "dev.jsonl",
    Path(DATA_DIR) / "test.jsonl",
    Path(BASE_MODEL) / "config.json",
):
    if not required.exists():
        raise FileNotFoundError(f"Missing offline input: {required}")

# Seed the output directory from a previous session so S1-a resumes.
if RESUME_FROM is not None:
    import shutil
    resumed = []
    for name in ("stage1_candidates.jsonl", "stage2_policy_candidates.jsonl"):
        source = Path(RESUME_FROM) / name
        if source.is_file():
            shutil.copy2(source, Path(OUTPUT_DIR) / name)
            resumed.append(f"{name} ({source.stat().st_size / 1e6:.1f} MB)")
    # Carry the adapter too, so RUN_TRAINING = False can evaluate it.
    adapter_source = Path(RESUME_FROM) / "final"
    if (adapter_source / "adapter_config.json").is_file():
        shutil.copytree(adapter_source, Path(OUTPUT_DIR) / "final", dirs_exist_ok=True)
        resumed.append("final/ (LoRA adapter)")
    if not resumed:
        raise FileNotFoundError(
            f"RESUME_FROM is set but no candidate file was found under {RESUME_FROM}. "
            "Attach the previous run's output and point RESUME_FROM at its "
            "tree_cot_rt_outputs directory."
        )
    print("Resuming from:", ", ".join(resumed))

sys.path.insert(0, str(Path(PROJECT_DIR) / "src"))
print("Project :", PROJECT_DIR)
print("Data    :", DATA_DIR)
print("Model   :", BASE_MODEL)
print("Output  :", OUTPUT_DIR)
''')

code(r'''
# ===================== INSTALL WHEELS AND IMPORT PROJECT ====================
import subprocess

import importlib

# Kaggle ships a complete, CUDA-matched stack and keeps bumping it. Passing
# bare package names to pip here did nothing at all - pip answered
# "Requirement already satisfied" for every one and the bundled wheels were
# never installed. Rather than force a downgrade across a major transformers
# release onto Kaggle's much newer torch, the wheelhouse is now a fallback
# for packages that are genuinely absent, and the versions actually in use
# are printed below so a run can be reproduced.
REQUIRED = ("transformers", "tokenizers", "accelerate", "peft",
            "huggingface_hub", "safetensors")

# peft >= 0.19 *raises* instead of returning False when torchao is installed
# but older than it wants, and the Kaggle image pairs a new peft with a stale
# torchao. Nothing here uses torchao quantisation, so dropping it is the fix.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"],
               check=False)

# The bundled stack (transformers 4.48.3, peft 0.14.0, huggingface-hub 0.28.1
# and their matching tokenizers/accelerate/safetensors) is what this code was
# written against. Kaggle's own stack is newer and has already broken two runs
# at the Trainer API, so the pinned set is the default. Set this to False to
# run on whatever Kaggle ships; the code adapts, but reports what it lost.
if PIN_BUNDLED_VERSIONS:
    bundled = sorted(Path(WHEELHOUSE).glob("*.whl"))
    if bundled:
        print("Pinning the bundled stack:",
              ", ".join(w.name.split("-")[0] + " " + w.name.split("-")[1] for w in bundled))
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
            "--no-index", "--no-deps", "--force-reinstall", *[str(w) for w in bundled],
        ])
        importlib.invalidate_caches()
    else:
        print("PIN_BUNDLED_VERSIONS is set but the wheelhouse is empty.")

missing = []
for name in REQUIRED:
    try:
        importlib.import_module(name)
    except ImportError:
        missing.append(name)

if missing:
    print("Installing from the offline wheelhouse:", ", ".join(missing))
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
        "--no-index", "--no-deps", "--find-links", WHEELHOUSE, *missing,
    ])
    for name in missing:
        importlib.invalidate_caches()
        importlib.import_module(name)
else:
    print("Every required package is already present; wheelhouse not needed.")

# Belt and braces: if torchao could not be removed, neutralise peft's probe.
try:
    import peft.tuners.lora.torchao as _lora_torchao
    _lora_torchao.is_torchao_available = lambda: False
except Exception:
    pass

import json
import pandas as pd
from IPython.display import display
import torch

from tree_cot_rt.cli import exact_f1
from tree_cot_rt.data import read_jsonl, write_jsonl
from tree_cot_rt.reasoning import typed_instances
from tree_cot_rt.timing import format_duration
from tree_cot_rt.training import predict, train

import platform, transformers, peft, accelerate
print("python      :", platform.python_version())
print("torch       :", torch.__version__, "| CUDA:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("peft        :", peft.__version__, "| accelerate:", accelerate.__version__)
if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU accelerator before running this notebook.")
''')

md(r"""
## 1. Data and ASQP type coverage

Every sentence is split into one sub-instance per ASQP type it contains, and
S1/S2 work on those typed instances. The table below is the real unit of cost:
S1-a samples `CHAINS_PER_TYPE` chains for each row of it.
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
print(f"S1-a will sample up to {len(typed_rows) * CHAINS_PER_TYPE} chains "
      f"({CHAINS_PER_TYPE} per typed instance) before its deadline.")
''')

md(r"""
## 2. S1 → S2 → S3

Each phase gets a slice of the remaining budget and stops at its deadline.
S1-a degrades gracefully: whatever chains it produced become the candidate pool,
and the coverage it reached is printed and recorded.

Candidate chains are cached to `stage1_candidates.jsonl`. Save this notebook's
output and attach it to a later session to resume instead of regenerating.
""")

code(r'''
if RUN_TRAINING:
    train_budget = session_remaining() - INFERENCE_RESERVE_SECONDS
    if train_budget <= 0:
        raise RuntimeError("No time left for training; raise SESSION_BUDGET_HOURS or cut the reserves.")
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
        final_lora_rank=FINAL_LORA_RANK,
        time_budget_seconds=train_budget,
        stage_fractions=STAGE_FRACTIONS,
    )
else:
    metrics_path = Path(OUTPUT_DIR) / "training_metrics.json"
    training_metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else None
    print("Training skipped; using the existing adapter at", Path(OUTPUT_DIR) / "final")

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

    # Project how much of S1-a is left, so the next session is predictable.
    covered = len({
        (row.get("instance_id"), row.get("quad_type"))
        for row in (json.loads(line) for line in
                    (Path(OUTPUT_DIR) / "stage1_candidates.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip())
    }) if (Path(OUTPUT_DIR) / "stage1_candidates.jsonl").is_file() else 0
    total_typed = len(typed_rows)
    s1a_seconds = stage_seconds.get("S1a_generation", 0.0)
    print()
    print(f"S1-a coverage: {covered}/{total_typed} typed instances "
          f"({covered / total_typed * 100:.1f}%)")
    if covered and covered < total_typed:
        rate = s1a_seconds / covered
        remaining_hours = (total_typed - covered) * rate / 3600
        budget_hours = train_budget * STAGE_FRACTIONS[0] / 3600 if RUN_TRAINING else 1.0
        print(f"At {rate:.1f} s/typed instance, the rest needs "
              f"{remaining_hours:.0f} GPU-h ~ {remaining_hours / max(budget_hours, 0.1):.0f} more session(s).")
        print("Save Version, then set RESUME_FROM to this run's tree_cot_rt_outputs.")
    truncated = training_metrics.get("budget", {}).get("truncated_phases", [])
    if truncated:
        print("Stopped early to stay inside the budget:", ", ".join(truncated))
        print("Lower MAX_TRAIN_RECORDS or CHAINS_PER_TYPE for fuller coverage per phase.")
    print("Session time left:", format_duration(max(0.0, session_remaining())))
''')

md(r"""
## 3. S4 self-consistent prediction and exact-match F1

The ASQP type is unknown at test time, so each sentence gets **one** type-agnostic
prompt, `INFERENCE_SAMPLES` reasoning chains by stochastic decoding, and a
majority vote over the quadruples (Eq. 9).

Sentences left unscored at the deadline are reported with an empty quadruple
list, so F1 is always measured over the full evaluation split.
""")

code(r'''
if RUN_INFERENCE:
    inference_budget = max(60.0, session_remaining())
    print(f"Inference budget: {format_duration(inference_budget)} for {len(test_records)} sentences")

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

Download these from the notebook's **Output** tab:

- `tree_cot_rt_outputs/final/` — S3 LoRA adapter, tokenizer, `categories.json`
- `tree_cot_rt_outputs/stage1_candidates.jsonl` — S1-a candidate pool (**reattach this to resume a later session**)
- `tree_cot_rt_outputs/stage2_selected_chains.jsonl` — chains GRPO kept
- `tree_cot_rt_outputs/training_metrics.json` — per-phase timing, GRPO losses, reward weights, budget report
- `tree_cot_rt_outputs/test_predictions.jsonl` — voted quadruples and the sampled chains
- `tree_cot_rt_outputs/evaluation_metrics.csv` — precision, recall, exact-quadruple F1

### Spreading the work over several sessions

S1-a is the bottleneck. To cover more of the training split, run the notebook
once, save the version, attach **its own output** as an extra input to the next
run, and point `OUTPUT_DIR` at a copy of `stage1_candidates.jsonl`. Cached
chains that still match their prompt are reused, so the next session continues
generating where this one stopped.
""")

notebook = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

target = REPO / "tree_cot_rt_kaggle.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {target} ({len(CELLS)} cells)")
