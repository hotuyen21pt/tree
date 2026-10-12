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

> **S1-a dominates the cost.** Sampling K chains for every typed instance is by
> far the most expensive phase, and a 7B model on a Kaggle GPU cannot cover a
> full ACOS training split in 9 hours. `MAX_TRAIN_RECORDS` and `CHAINS_PER_TYPE`
> below are sized for one session; the budget is a backstop that keeps whatever
> was generated rather than losing the run.
>
> **Order of magnitude on this dataset:** 8,815 training sentences expand to
> roughly 10,500 typed instances. At ~30 s to sample 4 chains, a full S1-a pass
> would take about 87 GPU-hours. `MAX_TRAIN_RECORDS = 300` keeps one session's
> S1-a near 3 hours; raise it only if you are willing to run several sessions
> and resume from `stage1_candidates.jsonl`.
""")

code(r'''
# ============================== SESSION CONFIG ==============================
import time
SESSION_START = time.perf_counter()

# --- Wall-clock budget -------------------------------------------------------
# Kaggle kills the session at the limit, which would lose the LoRA adapter.
# Every phase stops at its own deadline instead.
SESSION_BUDGET_HOURS      = 9.0   # Kaggle GPU session limit
SETUP_RESERVE_MINUTES     = 20    # wheels + model load
INFERENCE_RESERVE_MINUTES = 90    # S4 sampling is 5 generations per sentence
SAFETY_MARGIN_MINUTES     = 15

# --- Scale -------------------------------------------------------------------
# Paper values are in the right-hand column; the defaults are what fits a
# single 9h Kaggle session on a 16 GB GPU.
MAX_TRAIN_RECORDS = 300    # paper: the whole split (8,815 here)
MAX_TEST_RECORDS  = 200    # paper: the whole split (1,890 here)
CHAINS_PER_TYPE   = 4      # paper: K = 10
GROUP_SIZE        = 6      # paper: G = 10
GRPO_STEPS        = 10     # paper: K = 10 GRPO steps
INFERENCE_SAMPLES = 5      # paper: M = 5
MAX_NEW_TOKENS    = 512    # paper: 2048

# --- Fine-tuning -------------------------------------------------------------
SELECTED_CHAINS_PER_TYPE = 2
FINAL_STEPS              = 10
BATCH_SIZE               = 1    # paper: 4 (needs an A100)
GRADIENT_ACCUMULATION    = 8    # paper: 2
GRPO_BATCH_SIZE          = 1    # paper: 2
CLIP_EPSILON             = 0.2
KL_COEFFICIENT           = 0.01
LEARNING_RATE            = 5e-5
MAX_LENGTH               = 1024 # paper: 2048
SEED                     = 444  # paper seeds: 444, 555, 666, 777

# Share of the training budget per phase: S1-a, S1-b, S2, S3.
STAGE_FRACTIONS = (0.40, 0.10, 0.30, 0.20)

RUN_TRAINING  = True
RUN_INFERENCE = True

# --- Paths -------------------------------------------------------------------
OUTPUT_DIR  = "/kaggle/working/tree_cot_rt_outputs"
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

sys.path.insert(0, str(Path(PROJECT_DIR) / "src"))
print("Project :", PROJECT_DIR)
print("Data    :", DATA_DIR)
print("Model   :", BASE_MODEL)
print("Output  :", OUTPUT_DIR)
''')

code(r'''
# ===================== INSTALL WHEELS AND IMPORT PROJECT ====================
import subprocess

# Only pinned local wheels; --no-deps keeps the resolver off the network.
# Kaggle already ships torch, numpy and pandas.
wheels = sorted(Path(WHEELHOUSE).glob("*.whl"))
if wheels:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
        "--no-index", "--no-deps", "--find-links", WHEELHOUSE,
        *[w.name.split("-")[0].replace("_", "-") for w in wheels],
    ])
else:
    print("No wheelhouse found; relying on the preinstalled Kaggle packages.")

import json
import pandas as pd
from IPython.display import display
import torch

from tree_cot_rt.cli import exact_f1
from tree_cot_rt.data import read_jsonl, write_jsonl
from tree_cot_rt.reasoning import typed_instances
from tree_cot_rt.timing import format_duration
from tree_cot_rt.training import predict, train

print("PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
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
