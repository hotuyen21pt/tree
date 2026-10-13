# Tree-CoT-RT — Tree-guided CoT + GRPO for ASQP

Reproduction of **Tree-CoT-RT** (Zhang et al., *Findings of ACL 2026*; the PDF
is in this repo as `2026.findings-acl.806.pdf`) for Aspect Sentiment Quad
Prediction, run on a private hotel-review dataset.

**Purpose: this is a baseline.** Its scores will be published next to another
method, so faithfulness to the paper matters more than peak F1. Several
things here look like they could be "optimised" and must not be — see
[Decisions that must not be reverted](#decisions-that-must-not-be-reverted).

This repo is coupled to a sibling repo, **s2it** (S2IT), the other baseline.
The two must train and evaluate on byte-identical splits; see
[Shared splits](#shared-splits).

---

## Current state

| | |
|---|---|
| Status | Code aligned with the paper; **never run end to end** |
| Tests | 42 pass |
| Kaggle bundle | Built and ready at `kaggle_offline_upload.zip` (not in git) |
| Expected cost | ~10 Kaggle P100 sessions, resumable — see [How many sessions](#how-many-sessions) |

---

## What the method does

Four steps (paper §3.3–3.6):

| Step | What happens |
|---|---|
| **S1-a** | Sample K Tree-CoT chains per *typed instance*, using the type-specific reasoning tree |
| **S1-b** | Tree-supervised LoRA fine-tuning (tSFT) on the valid chains → the GRPO initial policy |
| **S2** | GRPO with a rule-based reward, then re-sample and keep the top-N chains per typed instance |
| **S3** | Fine-tune on the selected chains: sentence → reasoning + quadruples |
| **S4** | Self-consistent inference: M chains per sentence, majority vote (Eq. 9) |

A *typed instance* is a sentence restricted to the quadruples of one ASQP
type, so a sentence with both an EAEO and an IAEO quad contributes two.

### The four ASQP types

`EAEO`, `EAIO`, `IAEO`, `IAIO` — explicit/implicit aspect × explicit/implicit
opinion. Each has its own reasoning-step order (paper Table 1), and **that
order is the method**:

| type | path |
|---|---|
| EAEO | identify A + O → pair → sentiment → category |
| EAIO | identify A → **infer O** → sentiment → category |
| IAEO | identify O → **infer A** → sentiment → category |
| IAIO | context → infer A and O → sentiment → category |

---

## Decisions that must not be reverted

| Decision | Why |
|---|---|
| Type names are **EAEO / EAIO / IAEO / IAIO** | The original code used `EE/EA/IE/IA`, where `"A"` meant *implicit opinion*. Every positional check (`quad_type[1] == "A"`) was unreadable and two of the four reasoning paths were wrong because of it. |
| **EAIO infers the opinion before deriving sentiment/category; IAEO infers the aspect before deriving sentiment** | Table 1. The original code assigned the category before inferring the opinion (EAIO) and derived the sentiment before inferring the aspect (IAEO), which inverts the paper's core idea. |
| **S3 and S4 are type-agnostic** | §3.6 prompts once per sentence with no gold type, because the type is unknown at test time. The original code enumerated all four types and unioned the results: 4× the cost and wrecked precision. S3 therefore trains on the full gold quad set with a type-free prompt so training and inference agree. |
| `R_cons` is **element-wise with bipartite matching** | Eq. 10. Exact-quad F1 scores three-of-four elements as zero, leaving GRPO with no gradient early on. |
| `R_valid` is the **normalised count** of satisfied path constraints | Appendix C.1, not a boolean. |
| `R_impl` requires a **sentence-grounded cue span** | Appendix C.1. Previously any occurrence of "because" earned full credit, making the component with the largest ablation drop (−5.39 F1) a no-op. |
| `R_div` measures distance from the **Step-1 template chains** | Eq. 11, not from the current sampling group. |
| GRPO KL uses the **k3 estimator** | Eq. 7: `π_ref/π − log(π_ref/π) − 1`, not `log π − log π_ref`. |
| **GRPO has its own gradient accumulation** | Table 8 sets GRPO to effective batch 4 (2×2) while Table 9 sets S3 to effective 8. Sharing one accumulation value cannot satisfy both. |
| GRPO **skips the optimiser step when every group is degenerate** | All-equal rewards give zero advantage; AdamW would still apply weight decay to a zero gradient. |
| `CHAINS_PER_TYPE = 10`, `GROUP_SIZE = 10` | **Raising K and G is nearly free.** Decoding a 7B model is memory-bandwidth bound: every step reads all 14 GB of weights whatever K is, and `num_return_sequences` batches the K chains into that one read. K=2 and K=10 measure the same wall-clock time. K=2 with `SELECTED_CHAINS_PER_TYPE=2` also made S2's selection a no-op — top-2 of at most 2 discards nothing — removing the mechanism the paper's `w/o GRPO-RL` ablation values at 3.92 F1. |
| `BATCH_SIZE=1`, `GRADIENT_ACCUMULATION=8` | Table 9's "batch 4, accumulation 2" is an **effective batch of 8**. Same optimiser trajectory, fits 16 GB. Not a deviation. |
| `MAX_LENGTH = 2048` | The paper's value, and it costs nothing: real sequences measure ~525 tokens and the collator pads to the longest in the batch, not to the cap. |

### Known deviations from the paper

| Deviation | Reason |
|---|---|
| `MAX_NEW_TOKENS = 384`, paper says 2048 | It is a **cap, not a target** — generation stops at EOS. The paper's own IAIO example chain (Table 14) measures ~201 tokens. 384 only bounds a model that fails to emit EOS. 2048 would multiply the worst case by 5. |
| **S1-a chains are generated by the local Qwen model**, not an external API | The paper uses Gemini 2.5 Pro / ChatGPT for S1-a. Kaggle has no Internet. To follow the paper, generate chains elsewhere and pre-populate `stage1_candidates.jsonl`, which the generation stage reuses when the prompt still matches. |
| `FINAL_LORA_RANK` below 32 | Only if forced. A 7B in fp16 is 14.2 GiB, so rank 32 needs more than one 16 GB card. `GPU T4 x2` keeps the paper's value; rank 8 fits one card and must be recorded. |
| Dataset is this project's hotel reviews, not Restaurant/Laptop ACOS | The paper's published numbers are not comparable to these. |
| **One seed (444), not the 4-seed mean** | §4.3.1 runs each method with seeds 444/555/666/777 and reports the average. Four runs is four times the compute. A single GRPO run has real variance, so report this number as single-run. |
| **fp16, not bf16** | Table 8/9 specify bf16; Kaggle's T4 and P100 do not support it. The GRPO log-probabilities are computed in fp32 to compensate, because `exp()` on an fp16 ratio overflows at 65504. |

---

## A caveat about this dataset

Typed-instance distribution on the 8,557-sentence training split this bundle
holds (8,859 typed instances in total):

| EAEO | IAEO | EAIO | IAIO |
|---|---|---|---|
| 8,379 | 344 | 136 | **0** |

Same shape before the language filter, on all 8,815 sentences (EAEO 8,611,
IAEO 370, EAIO 140, IAIO 0). **This data has essentially no implicit cases,
and no IAIO at all.**

Tree-CoT-RT is a paper *about* implicit sentiment; IAIO is the case it
emphasises most and where it reports its largest gains. On this dataset that
contribution cannot show up. The baseline is still valid as a baseline, but
any claim comparing implicit-sentiment handling will be uninformative here.
Worth stating explicitly in the write-up.

---

## Layout

```
src/tree_cot_rt/
  reasoning.py    TREE_PATHS, prompts, chain validity, majority voting
  rewards.py      R_cons / R_valid / R_impl / R_div and group advantages
  training.py     S1-a, S1-b, S2 (GRPO + selection), S3, and predict() for S4
  data.py         JSONL reader, Quad/Instance, quad_type()
  budget.py       Budget/Deadline: stop a phase before Kaggle kills the session
  cli.py          train | predict | evaluate
scripts/
  prepare_offline_bundle.py   build the Kaggle upload (data, wheels, zip)
  build_notebook.py           regenerate the notebook — EDIT THIS, not the .ipynb
tests/                        20 tests, no GPU needed
```

> `tree_cot_rt_kaggle.ipynb` is **generated**. Edit `scripts/build_notebook.py`
> and re-run it, or your change is lost on the next rebuild.

This project has **no dependency parser** — unlike s2it there is no Stanza
step and no language tagging, and `langdetect` is not a dependency.

---

## Data

Not in git. Expected locally as `converted_apc/{train,dev,test}.jsonl`
(8,815 / 1,895 / 1,890 sentences). `asqp_data/` is also accepted.

### Shared splits

Build the **s2it bundle first** — it applies the language filter and writes
the id list — then point this one at the ids it wrote:

```bash
python scripts/prepare_offline_bundle.py \
    --subset-ids ../s2it/kaggle_offline_upload/offline_bundle/subset_ids.json
```

Both bundles then hold identical splits (verified: 8,557 / 1,846 / 1,831,
same ids, same texts). Nothing is sampled; the alignment is still required,
because s2it's language filter drops records this repo keeps, so without it
the two baselines would be measured on different test sets.

**Any third method compared against these baselines must use the same
`subset_ids.json`.**

`offline_bundle/dataset_stats.json` records what the bundle ended up holding,
after that filtering — not the sizes of the source files.

Output: `kaggle_offline_upload.zip` (~15 MB) with `project/`,
`offline_bundle/`, `UPLOAD_README.txt` and the notebook. No model weights.

### The evaluation set

**Neither split is sampled.** Both baselines train on every sentence the
language filter keeps, and evaluate on every sentence they can score.

| | sentences |
|---|---|
| `converted_apc/test.jsonl` as delivered | 1,890 |
| removed by the language filter | 59 |
| **evaluated, by both baselines** | **1,831** |

Those 59 are spread over 17 languages with no parser in the supported set
(ro, es, af, ca, no, da, it, sv, ko, cy, et, ru, sw, tl, nl, pt, hr). S2IT
structurally cannot score them: it needs a dependency parse, and Stanza has
no model for them here. 1,831 is therefore the largest test set the two
baselines can share.

So:

- **s2it vs tree** — valid, identical 1,831 sentences, verified id by id.
- **vs a third method** — valid only if it is scored on the same 1,831.
  Use `offline_bundle/subset_ids.json`, which lists them.
- **vs the numbers printed in the papers** — not comparable at all. Those
  are Restaurant/Laptop ACOS; this is a private hotel-review dataset.

A method that needs no parser could be scored on all 1,890, but then it is
no longer comparable to S2IT. Report 1,831 for everything.

### How many sessions

The full split is 8,557 sentences, which expand to **8,859 typed instances**,
and the method generates chains over all of them twice — S1-a to build the
candidate pool, S2 to re-sample from the optimised policy. Worst-case P100
hours, assuming every generation runs to the 384-token cap:

| phase | hours |
|---|---|
| S1-a over 8,859 typed instances | ~33 |
| S2 re-generation for selection | ~33 |
| S1-b, GRPO updates, S3 | ~6 |
| **training total** | **~71 h → 9 sessions** |
| S4 over 1,831 test sentences | ~6.8 h → 1 session |

So about **10 P100 sessions**, resumed with `RESUME_FROM`. The cheap-looking
alternatives are not cheap: cutting `CHAINS_PER_TYPE` makes S2's selection a
near no-op and removes the mechanism the paper's `w/o GRPO-RL` ablation values
at 3.92 F1, and cutting the training split means the two baselines are no
longer measured on the same data. Sessions are the only budget here that can
grow without changing what is being measured.

These hours come from a memory-bandwidth model, not a measurement — expect
±40%, and let the first session's printed rate re-project the count.

---

## Running on Kaggle

Upload the zip as a Dataset, import `tree_cot_rt_kaggle.ipynb`, attach the
Dataset plus the **Qwen2.5-7B-Instruct** Kaggle Model, Run All.

- **Accelerator: P100.** This workload is dominated by decoding, which is
  memory-bandwidth bound, and the P100 has 732 GB/s against the T4's 320 —
  2.3× faster here for free. (S2IT is the opposite case and wants the T4.)
- **Internet: Off.**

### The Kaggle environment is not pinned
> **Two runs were lost to the Kaggle image's transformers.** `PIN_BUNDLED_VERSIONS`
> defaults to `True`, which force-installs the bundled transformers 4.48.3 /
> peft 0.14.0 set this code was written against. Set it to `False` to run on
> whatever Kaggle ships; `_training_arguments()` then filters `TrainingArguments`
> against the installed signature and prints what it had to drop.

The bundled wheelhouse is a **fallback, not a pin**. The notebook installs
from it only what is genuinely missing, and prints the versions actually in
use. Two findings forced this:

- Passing bare package names to `pip install --no-index --find-links` did
  nothing: pip answered *"Requirement already satisfied"* for all six and the
  bundled wheels were never installed. The run silently used Kaggle's stack.
- Kaggle moves fast. One observed image: Python 3.13, torch 2.11,
  transformers 5.16, peft 0.20. Forcing a downgrade to transformers 4.48 on
  that torch build is riskier than using what Kaggle tests together.

Consequences to be aware of:

- `peft >= 0.19` **raises** instead of returning `False` when `torchao` is
  installed but older than it wants, and the image pairs a new peft with a
  stale torchao. The notebook uninstalls `torchao` (nothing here uses torchao
  quantisation) and, as a fallback, neutralises peft's probe directly.
- `torch_dtype` was renamed `dtype` in transformers 5. `_dtype_kwargs()`
  picks the name the installed version accepts, so both majors work.
- **Record the printed versions with any result.** The environment is not
  reproducible from this repo alone.

### This needs about ten sessions

The hours are in [How many sessions](#how-many-sessions): ~71 h of training
and ~6.8 h of inference on a P100, against a 9 h session limit. The plan is
nine training sessions, each after the first with `RESUME_FROM` set, then one
with `RUN_TRAINING = False` and `RESUME_FROM` set, which spends the whole
session on inference.

### Resuming

Every phase has a deadline from `budget.py`. S1-a degrades gracefully: the
chains it produced become the candidate pool, everything downstream still
runs, and the session reports the coverage reached plus a projection of how
many sessions remain.

To continue: *Save Version*, attach that output to the next run, set
`RESUME_FROM` to its `tree_cot_rt_outputs` path. Cached chains whose prompt
still matches are reused, so S1-a continues where it stopped. `RESUME_FROM`
also copies the LoRA adapter, so a session can evaluate without retraining.

> Changing a prompt invalidates the cache — cached chains are keyed by
> `(instance_id, quad_type, prompt, generator_model)`. After editing
> `reasoning.py`, previous candidates are regenerated.

---

## Local development

```bash
pip install -e ".[dev]"
python -m pytest -q        # 42 pass, no GPU needed
```

CLI:

```bash
tree-cot-rt train    --train train.jsonl --dev dev.jsonl --output-dir runs/tree
tree-cot-rt predict  --input test.jsonl --adapter-dir runs/tree/final \
                     --base-model Qwen/Qwen2.5-7B --output preds.jsonl
tree-cot-rt evaluate --gold test.jsonl --predictions preds.jsonl
```

`train(..., time_budget_seconds=None)` disables the deadlines, which is what
you want off Kaggle.

---

## Not in git

`converted_apc/`, `kaggle_offline_upload/`, `kaggle_offline_upload.zip`,
`runs/`, caches. All reproducible from the data plus
`scripts/prepare_offline_bundle.py`. A fresh clone has no data.

---

## Open items

- **Never run end to end.** Every runtime figure here is extrapolated from a
  memory-bandwidth model, not measured. Expect ±40%; the first session prints
  the real rate and re-projects.
- S1-a uses the local model instead of the external APIs the paper used.
- `dev_instances` is loaded and counted but not otherwise used by `train()`.
- No IAIO instances exist in this dataset, so that part of the method is
  exercised by nothing.
