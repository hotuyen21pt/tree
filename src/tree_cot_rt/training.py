from __future__ import annotations

import inspect
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from .budget import Budget, Deadline
from .data import Instance
from .reasoning import (
    extract_reasoning,
    majority_vote,
    chain_is_valid,
    parse_quads,
    reasoning_prompt,
    typed_instances,
)
from .rewards import REWARD_WEIGHTS, group_advantages, score_chain
from .timing import format_duration


#: TrainingArguments fields this project cannot silently lose. Transformers
#: renames and removes arguments across majors - `evaluation_strategy` became
#: `eval_strategy` in 4.41, `warmup_ratio` is gone in 5.x - and Kaggle's image
#: moves between majors, so everything else is filtered against the installed
#: signature and reported rather than assumed.
_ESSENTIAL_TRAINING_ARGS = (
    "output_dir",
    "learning_rate",
    "num_train_epochs",
    "max_steps",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "lr_scheduler_type",
)


def _accepted_training_args() -> set[str] | None:
    """Field names the installed TrainingArguments takes, or None if unknown.

    Tried most reliable first. A signature that collapsed to bare `*args,
    **kwargs` tells us nothing and is skipped; if every source does, None is
    returned and nothing is filtered, so a genuine mistake still raises.
    """
    import dataclasses

    sources = []
    for probe in (
        lambda: set(inspect.signature(TrainingArguments).parameters),
        lambda: {field.name for field in dataclasses.fields(TrainingArguments)},
        lambda: set(inspect.signature(TrainingArguments.__init__).parameters),
    ):
        try:
            sources.append(probe())
        except (TypeError, ValueError):
            continue
    for names in sources:
        names -= {"self", "args", "kwargs"}
        if names:
            return names
    return None


def _training_arguments(**desired: Any) -> TrainingArguments:
    """Build TrainingArguments, dropping fields this transformers cannot take."""
    import transformers

    accepted = _accepted_training_args()
    if accepted is None:
        # Could not introspect; pass everything through rather than guess.
        return TrainingArguments(**desired)
    lost_essential = [
        name for name in _ESSENTIAL_TRAINING_ARGS
        if name in desired and name not in accepted
    ]
    if lost_essential:
        raise RuntimeError(
            f"transformers {transformers.__version__} does not accept "
            f"{lost_essential} on TrainingArguments. Dropping these would change "
            "the training configuration silently, so the run stops instead. Pin "
            "the bundled transformers version, or map them to the new names."
        )
    dropped = sorted(name for name in desired if name not in accepted)
    if dropped:
        print(
            f"[transformers {transformers.__version__}] TrainingArguments does not "
            f"accept {dropped}; continuing without them.",
            flush=True,
        )
        # Losing these does not fail here, it fails later as an out-of-memory
        # error that looks unrelated, so say so now.
        memory_critical = sorted(
            {"gradient_checkpointing", "fp16", "bf16"}.intersection(dropped)
        )
        if memory_critical:
            print(
                f"  WARNING: {memory_critical} control precision and activation "
                "memory. Without them this run is likely to exhaust GPU memory. "
                "Set PIN_BUNDLED_VERSIONS = True to use the bundled transformers.",
                flush=True,
            )
    return TrainingArguments(**{k: v for k, v in desired.items() if k in accepted})


def _dtype_kwargs(dtype) -> dict[str, Any]:
    """`torch_dtype` is deprecated in transformers 5, which renamed it `dtype`.

    Kaggle's image moves between major versions, so pick the name the
    installed library actually accepts instead of pinning to one of them.
    """
    import transformers

    try:
        major = int(transformers.__version__.split(".")[0])
    except (AttributeError, ValueError):
        major = 4
    return {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}


def _auto_device_map() -> str | None:
    """Shard across GPUs only when there is more than one.

    With a single device `device_map="auto"` offloads whatever does not fit to
    CPU instead of failing. That is roughly fifty times slower and presents as
    a run that simply never finishes, so one GPU loads normally and lets
    `check_gpu_memory` decide whether the configuration is possible at all.
    """
    if not torch.cuda.is_available():
        return None
    return "auto" if torch.cuda.device_count() > 1 else None


def assert_not_offloaded(model: Any) -> None:
    """Stop if accelerate placed any layer on CPU or disk.

    An offloaded layer copies its weights over PCIe on every forward and every
    backward pass. A stage that should take half an hour then reports an ETA of
    twenty-two hours, which is how this was found.
    """
    device_map = getattr(model, "hf_device_map", None)
    if not device_map:
        return
    placements: dict[str, int] = {}
    for device in device_map.values():
        placements[str(device)] = placements.get(str(device), 0) + 1
    print(f"Model placement: {placements}", flush=True)
    spilled = sorted(
        name for name, device in device_map.items() if str(device) in {"cpu", "disk"}
    )
    if spilled:
        raise RuntimeError(
            f"{len(spilled)} module(s) were offloaded to CPU or disk because they "
            f"did not fit in GPU memory, for example {spilled[:3]}.\n"
            "Training would still run, roughly fifty times slower, so it stops "
            "here instead.\n"
            "  - Switch the Kaggle accelerator to 'GPU T4 x2', or\n"
            "  - Lower LORA_RANK and MAX_LENGTH and record the deviation."
        )


def _model_parameter_count(model_name: str) -> int | None:
    """Parameter count from the config alone, without loading any weights."""
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        layers = config.num_hidden_layers
        hidden = config.hidden_size
        inter = config.intermediate_size
        heads = config.num_attention_heads
        kv_heads = getattr(config, "num_key_value_heads", heads)
        head_dim = getattr(config, "head_dim", hidden // heads)
    except Exception:
        return None
    kv = kv_heads * head_dim
    per_layer = 2 * hidden * hidden + 2 * hidden * kv + 3 * hidden * inter
    embeddings = config.vocab_size * hidden
    if not getattr(config, "tie_word_embeddings", False):
        embeddings *= 2
    return layers * per_layer + embeddings


def _lora_parameter_count(model_name: str, rank: int) -> int | None:
    """Trainable parameters LoRA adds to the seven targeted projections."""
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        hidden = config.hidden_size
        inter = config.intermediate_size
        heads = config.num_attention_heads
        kv_heads = getattr(config, "num_key_value_heads", heads)
        head_dim = getattr(config, "head_dim", hidden // heads)
    except Exception:
        return None
    kv = kv_heads * head_dim
    widths = [
        (hidden, hidden), (hidden, kv), (hidden, kv), (hidden, hidden),
        (hidden, inter), (hidden, inter), (inter, hidden),
    ]
    return config.num_hidden_layers * sum(rank * (a + b) for a, b in widths)


def check_gpu_memory(model_name: str, lora_rank: int, dtype_bytes: int = 2) -> None:
    """Fail before loading weights if this model and rank cannot possibly fit.

    The base model dominates: a 7B in fp16 is 14.2 GiB of a 16 GB card's
    14.6 GiB, so the LoRA weights and their two AdamW states decide whether a
    run is possible at all. Discovering that during `Trainer.__init__` wastes
    several minutes and produces a traceback that does not name the cause.
    """
    if not torch.cuda.is_available():
        return
    params = _model_parameter_count(model_name)
    lora = _lora_parameter_count(model_name, lora_rank)
    if params is None or lora is None:
        return
    gib = float(2**30)
    weights = params * dtype_bytes / gib
    # LoRA weights stay fp32, and AdamW keeps two states of the same size.
    adapter = lora * 4 * 3 / gib
    devices = torch.cuda.device_count()
    capacity = sum(
        torch.cuda.get_device_properties(i).total_memory for i in range(devices)
    ) / gib
    needed = weights + adapter
    print(
        f"GPU budget: {capacity:.2f} GiB across {devices} device(s) | "
        f"weights {weights:.2f} + LoRA r{lora_rank} {adapter:.2f} = {needed:.2f} GiB",
        flush=True,
    )
    headroom = capacity - needed
    if headroom <= 0:
        per_rank = lora / lora_rank
        affordable = max(
            (r for r in (64, 32, 16, 8, 4)
             if weights + per_rank * r * 4 * 3 / gib < capacity),
            default=None,
        )
        advice = (
            f"Lower LORA_RANK to {affordable}, which departs from the paper and "
            "should be recorded"
            if affordable
            else "Use a smaller backbone"
        )
        raise RuntimeError(
            f"This cannot fit: {needed:.2f} GiB of weights and optimiser state "
            f"against {capacity:.2f} GiB of GPU memory, before any activations.\n"
            f"  - Switch the Kaggle accelerator to 'GPU T4 x2' "
            f"({2 * capacity / max(devices, 1):.0f} GiB total), which shards the "
            f"model across both and keeps the paper's rank, or\n"
            f"  - {advice}."
        )
    # A 7B in fp16 leaves very little room on a 16 GB card. The run that
    # trained for eight hours had 0.15 GiB spare at rank 8 and sequence length
    # 768, so a thin margin is survivable but worth saying out loud.
    if headroom < 0.5:
        print(
            f"  WARNING: only {headroom * 1024:.0f} MiB spare for activations. "
            "Keep MAX_LENGTH and BATCH_SIZE small, or use 'GPU T4 x2'.",
            flush=True,
        )


def _dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def _load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_base(model_name: str):
    started = time.perf_counter()
    print(f"Loading {model_name}", flush=True)
    print(
        "  Reading ~15 GB of weights from the Kaggle input mount; this is the "
        "longest step with no output, often several minutes.",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=_auto_device_map(),
        trust_remote_code=True,
        **_dtype_kwargs(_dtype()),
    )
    assert_not_offloaded(model)
    print(f"  loaded in {format_duration(time.perf_counter() - started)}", flush=True)
    model.config.use_cache = False
    return model


def _prepare_for_training(model: Any, rank: int, alpha: int, dropout: float):
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    return get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        ),
    )


class _TimeBudgetCallback(TrainerCallback):
    """Stops a fine-tuning stage at its deadline so the adapter is still saved."""

    def __init__(self, deadline: Deadline) -> None:
        self.deadline = deadline

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if self.deadline.expired():
            control.should_training_stop = True
            control.should_save = False
        return control


def _chat_text(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt + "\n"


class _PromptDataset(torch.utils.data.Dataset):
    def __init__(self, examples, tokenizer, max_length: int):
        self.rows = []
        for prompt, completion in examples:
            prompt_ids = tokenizer(
                _chat_text(tokenizer, prompt),
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )["input_ids"]
            completion_ids = tokenizer(
                completion + (tokenizer.eos_token or ""),
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )["input_ids"]
            if len(completion_ids) >= max_length:
                completion_ids = completion_ids[-max_length:]
                prompt_ids = []
            else:
                prompt_ids = prompt_ids[-(max_length - len(completion_ids)):]
            self.rows.append(
                {
                    "input_ids": prompt_ids + completion_ids,
                    "labels": [-100] * len(prompt_ids) + completion_ids,
                }
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class _CompletionCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, rows):
        width = max(len(row["input_ids"]) for row in rows)
        ids, masks, labels = [], [], []
        for row in rows:
            padding = width - len(row["input_ids"])
            ids.append(row["input_ids"] + [self.tokenizer.pad_token_id] * padding)
            masks.append([1] * len(row["input_ids"]) + [0] * padding)
            labels.append(row["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _fit_lora(
    model,
    tokenizer,
    examples,
    output_dir: Path,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    steps: int,
    batch_size: int,
    gradient_accumulation: int,
    learning_rate: float,
    max_length: int,
    deadline: Deadline | None = None,
):
    if not examples:
        raise ValueError(f"No valid fine-tuning examples for {output_dir.name}")
    model = _prepare_for_training(model, rank, alpha, dropout)
    trainer = Trainer(
        model=model,
        args=_training_arguments(
            output_dir=str(output_dir / "trainer"),
            max_steps=steps,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=max(1, min(10, steps)),
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
            gradient_checkpointing=True,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            optim="adamw_torch",
        ),
        train_dataset=_PromptDataset(examples, tokenizer, max_length),
        data_collator=_CompletionCollator(tokenizer),
        callbacks=[_TimeBudgetCallback(deadline)] if deadline is not None else None,
    )
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer.model


def _generate(
    model,
    tokenizer,
    prompt: str,
    *,
    count: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    chat = _chat_text(tokenizer, prompt)
    inputs = tokenizer(
        chat,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)
    with torch.inference_mode():
        sequences = model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            num_return_sequences=count,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    prompt_length = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True).strip()
        for sequence in sequences
    ]


def _examples_for_typed(instances: list[Instance], categories: list[str]):
    for instance in instances:
        for typed in typed_instances(instance):
            yield typed, reasoning_prompt(typed.text, typed.quad_type, categories)


def _generator_identity(name: str) -> str:
    """Identify the generator by model directory, not by mount path.

    Kaggle mounts the same model under a different path between sessions, and
    a cache keyed on the full path is then silently discarded - which costs
    hours of regeneration for chains that were perfectly good.
    """
    base, separator, suffix = str(name).partition(" (")
    return Path(base).name + separator + suffix


def _write_candidates(path: Path, rows: list[dict[str, Any]]) -> None:
    """Replace the cache atomically.

    This file is rewritten every 25 typed instances and is the only thing that
    makes a lost session cheap to resume. A session stopped part way through a
    plain overwrite leaves a truncated last line, and the next run then fails
    to read any of it - which loses every chain generated so far. Writing
    beside the target and renaming means the cache on disk is always either
    the previous complete file or the new one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_candidates(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as stream:
        lines = stream.readlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            if line_number == len(lines):
                # A torn last line is what an interrupted write leaves behind.
                # Dropping one chain is far better than discarding the cache.
                print(
                    f"{path}: last line is incomplete and was skipped; "
                    f"{len(rows)} chains recovered.",
                    flush=True,
                )
                break
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _generation_stage(
    model,
    tokenizer,
    instances: list[Instance],
    categories: list[str],
    *,
    generator_name: str,
    count: int,
    max_new_tokens: int,
    output_path: Path,
    deadline: Deadline | None = None,
) -> list[dict[str, Any]]:
    cached = _read_candidates(output_path)
    cached_by_id = defaultdict(list)
    for row in cached:
        cached_by_id[(row.get("instance_id"), row.get("quad_type"))].append(row)
    rows = []
    total = sum(1 for _ in _examples_for_typed(instances, categories))
    if cached:
        reusable = sum(
            1
            for typed, prompt in _examples_for_typed(instances, categories)
            if any(
                row.get("prompt") == prompt
                and _generator_identity(row.get("generator_model", ""))
                == _generator_identity(generator_name)
                and chain_is_valid(row.get("chain", ""), typed.quad_type)
                for row in cached_by_id[(typed.instance_id, typed.quad_type)]
            )
        )
        print(
            f"Cache: {len(cached)} chains on disk, covering {reusable}/{total} "
            "typed instances for this prompt and model.",
            flush=True,
        )
        if not reusable:
            print(
                "  None are reusable - the prompt or the model changed, so every "
                "chain will be regenerated. Stop now if that is not intended.",
                flush=True,
            )
    covered = 0
    for index, (typed, prompt) in enumerate(_examples_for_typed(instances, categories), 1):
        # Generating K chains for every typed instance is by far the most
        # expensive phase. When the session budget runs out, keep what has been
        # generated so far instead of losing the run; coverage is recorded.
        if deadline is not None and deadline.expired():
            print(
                f"Stopped Tree-CoT generation at {index - 1}/{total} typed instances "
                "to stay inside the session budget.",
                flush=True,
            )
            break
        prior = [
            row
            for row in cached_by_id[(typed.instance_id, typed.quad_type)]
            if row.get("prompt") == prompt
            and _generator_identity(row.get("generator_model", ""))
            == _generator_identity(generator_name)
            and chain_is_valid(row.get("chain", ""), typed.quad_type)
        ]
        for row in prior:
            row["valid"] = True
        candidates = list(prior)
        if len(candidates) < count:
            generated = _generate(
                model,
                tokenizer,
                prompt,
                count=count - len(candidates),
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.95,
            )
            for chain in generated:
                parsed = parse_quads(chain)
                candidates.append(
                    {
                        "instance_id": typed.instance_id,
                        "text": typed.text,
                        "quad_type": typed.quad_type,
                        "generator_model": generator_name,
                        "prompt": prompt,
                        "chain": chain,
                        "valid": chain_is_valid(chain, typed.quad_type),
                        "gold": [quad.as_dict() for quad in typed.quads],
                    }
                )
        rows.extend(candidates[:count])
        covered = index
        if index % 25 == 0 or index == total:
            _write_candidates(output_path, rows + cached)
            print(f"Generated Tree-CoT chains for {index}/{total} typed instances")
    unique = {}
    for row in rows:
        unique[(row["instance_id"], row["quad_type"], row["chain"])] = row
    result = list(unique.values())
    _write_candidates(output_path, result)
    print(f"Tree-CoT chains: {len(result)} over {covered}/{total} typed instances")
    return result


def _sft_examples(candidates: list[dict[str, Any]], valid_only: bool) -> list[tuple[str, str]]:
    examples = []
    for row in candidates:
        chain = row.get("chain", "")
        parsed = parse_quads(chain)
        if valid_only and (
        not parsed or not chain_is_valid(chain, row.get("quad_type", ""))
        ):
            continue
        target = json.dumps(
            {
                "reasoning": extract_reasoning(chain),
                "quadruples": row["gold"],
            },
            ensure_ascii=False,
        )
        examples.append((row["prompt"], target))
    return examples


def _final_examples(
    selected: list[dict[str, Any]],
    instances_by_id: dict[str, Instance],
    categories: list[str],
) -> list[tuple[str, str]]:
    """S3 supervision: (sentence, selected reasoning chain, gold quadruples).

    The paper's Step 3 training instance carries no ASQP type, because Step 4
    has to predict without one. The per-type chains selected in Step 2 are
    merged back into a single example per sentence targeting all its quads.
    """
    by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_instance[str(row.get("instance_id"))].append(row)
    examples: list[tuple[str, str]] = []
    for instance_id, rows in by_instance.items():
        instance = instances_by_id.get(instance_id)
        if instance is None:
            continue
        reasoning = " ".join(
            extract_reasoning(row.get("chain", "")) for row in rows
        ).strip()
        if not reasoning:
            continue
        target = json.dumps(
            {
                "reasoning": reasoning,
                "quadruples": [quad.as_dict() for quad in instance.quads],
            },
            ensure_ascii=False,
        )
        examples.append((reasoning_prompt(instance.text, None, categories), target))
    return examples


def _completion_ids(tokenizer, prompt: str, completion: str, max_length: int):
    prompt_text = _chat_text(tokenizer, prompt)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(
        completion + (tokenizer.eos_token or ""),
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    if len(completion_ids) >= max_length:
        completion_ids = completion_ids[-max_length:]
        prompt_ids = []
    else:
        prompt_ids = prompt_ids[-(max_length - len(completion_ids)):]
    return prompt_ids + completion_ids, max(0, len(prompt_ids) - 1)


def _sequence_log_probability(model, tokenizer, prompt: str, completion: str, max_length: int):
    full_ids, completion_start = _completion_ids(tokenizer, prompt, completion, max_length)
    if len(full_ids) < 2:
        raise ValueError("Prompt/completion tokenization produced an empty sequence")
    device = next(model.parameters()).device
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits
        # fp32 regardless of the model dtype: Kaggle's T4/P100 have no bf16,
        # so these tensors would otherwise be fp16, whose 65504 ceiling the
        # GRPO ratio exp() can overflow.
        log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        target_ids = input_ids[:, 1:]
        token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        result = token_log_probs.squeeze(0)[completion_start:]
        if result.numel() == 0:
            raise ValueError("Completion has no next-token log probabilities")
        return result


def _sequence_log_probability_with_grad(
    model, tokenizer, prompt: str, completion: str, max_length: int
):
    full_ids, completion_start = _completion_ids(tokenizer, prompt, completion, max_length)
    device = next(model.parameters()).device
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    logits = model(input_ids=input_ids, use_cache=False).logits
    log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    target_ids = input_ids[:, 1:]
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    result = token_log_probs.squeeze(0)[completion_start:]
    if result.numel() == 0:
        raise ValueError("Completion has no next-token log probabilities")
    return result


def _grpo_accumulate(
    model,
    tokenizer,
    prompt: str,
    completions: list[str],
    rewards: list[float],
    *,
    clip_epsilon: float,
    kl_coefficient: float,
    max_length: int,
    scale: float = 1.0,
):
    """One GRPO group: backward only, so the caller controls accumulation."""
    old_logps = [
        _sequence_log_probability(model, tokenizer, prompt, text, max_length).detach()
        for text in completions
    ]
    with torch.no_grad():
        set_adapter = getattr(model, "set_adapter", None)
        if set_adapter is None:
            raise RuntimeError("LoRA model does not expose set_adapter for the GRPO reference policy")
        policy_adapter = model.active_adapter
        set_adapter("reference")
        try:
            ref_logps = [
                _sequence_log_probability(model, tokenizer, prompt, text, max_length)
                for text in completions
            ]
        finally:
            set_adapter(policy_adapter)
    advantages = torch.tensor(
        group_advantages(rewards),
        dtype=old_logps[0].dtype,
        device=old_logps[0].device,
    )
    if torch.count_nonzero(advantages) == 0:
        return 0.0
    losses = []
    for index, text in enumerate(completions):
        new_logp = _sequence_log_probability_with_grad(
            model,
            tokenizer,
            prompt,
            text,
            max_length,
        )
        # exp(20) is 4.9e8, far past fp16's 65504; the log-probs above are
        # fp32 so this is safe, and 10 still allows a 22,000x ratio, which a
        # single clipped GRPO step can never legitimately reach.
        ratio = torch.exp(
            torch.clamp(new_logp - old_logps[index], min=-10, max=10)
        )
        unclipped = ratio * advantages[index]
        clipped = torch.clamp(
            ratio,
            1.0 - clip_epsilon,
            1.0 + clip_epsilon,
        ) * advantages[index]
        # Eq. 7: the k3 forward-KL estimator, pi_ref/pi - log(pi_ref/pi) - 1.
        log_ratio_reference = ref_logps[index] - new_logp
        kl = torch.exp(log_ratio_reference) - log_ratio_reference - 1.0
        loss = (
            -torch.minimum(unclipped, clipped) + kl_coefficient * kl
        ) / len(completions)
        # Log-probabilities are token vectors; reduce each completion's
        # token-level objective to a scalar before backward().
        loss = loss.mean() * scale
        loss.backward()
        losses.append(float(loss.detach().cpu()))
    return sum(losses)


def train(
    train_instances: list[Instance],
    dev_instances: list[Instance],
    output_dir: str | Path,
    *,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    final_model_name: str = "Qwen/Qwen2.5-7B",
    chains_per_type: int = 10,
    grpo_steps: int = 10,
    group_size: int = 10,
    grpo_batch_size: int = 2,
    grpo_gradient_accumulation: int = 2,
    clip_epsilon: float = 0.2,
    kl_coefficient: float = 0.01,
    selected_chains_per_type: int = 2,
    final_steps: int = 10,
    batch_size: int = 4,
    gradient_accumulation: int = 2,
    learning_rate: float = 5e-5,
    max_length: int = 2048,
    max_new_tokens: int = 1024,
    seed: int = 444,
    final_lora_rank: int = 32,
    time_budget_seconds: float | None = None,
    stage_fractions: tuple[float, float, float, float] = (0.40, 0.10, 0.30, 0.20),
) -> dict[str, Any]:
    """Run S1-a, S1-b, S2 and S3 inside an optional wall-clock budget.

    `stage_fractions` splits `time_budget_seconds` across the four phases.
    S1-a gets the largest share because sampling K chains per typed instance
    dominates the cost, and it degrades gracefully: whatever is generated
    before the deadline becomes the candidate pool.
    """
    if not train_instances or not dev_instances:
        raise ValueError("Both train and development splits must contain records")
    random.seed(seed)
    torch.manual_seed(seed)
    budget = Budget(time_budget_seconds)
    if not budget.unlimited:
        print(f"Session budget: {format_duration(time_budget_seconds)}", flush=True)

    def phase_deadline(index: int, name: str) -> Deadline | None:
        if budget.unlimited:
            return None
        share = stage_fractions[index] / sum(stage_fractions[index:])
        deadline = Deadline(budget, budget.remaining() * share, name)
        print(f"[budget] {deadline.describe()}", flush=True)
        return deadline
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    categories = sorted(
        {quad.aspect_category for row in train_instances for quad in row.quads}
    )
    if not categories:
        raise ValueError("Training data contains no aspect categories")
    times: dict[str, float] = {}
    run_started = time.perf_counter()

    check_gpu_memory(model_name, final_lora_rank)   # S3 uses the largest rank of the run
    print("S1-a: generating local multi-path Tree-CoT candidates (K =", chains_per_type, ")")
    started = time.perf_counter()
    tokenizer = _load_tokenizer(model_name)
    generator = _load_base(model_name)
    initial_candidates = _generation_stage(
        generator,
        tokenizer,
        train_instances,
        categories,
        generator_name=model_name,
        count=chains_per_type,
        max_new_tokens=max_new_tokens,
        output_path=output / "stage1_candidates.jsonl",
        deadline=phase_deadline(0, "S1a_generation"),
    )
    times["S1a_generation"] = time.perf_counter() - started

    print("S1-b: tree-supervised LoRA initialization")
    started = time.perf_counter()
    s1_examples = _sft_examples(initial_candidates, valid_only=True)
    if not s1_examples:
        examples = [
            f"- type={row.get('quad_type')}, valid={row.get('valid')}, chain={row.get('chain', '')[:1200]!r}"
            for row in initial_candidates[:3]
        ]
        raise ValueError(
            "No valid Tree-CoT chains were generated for S1-b. Inspect "
            f"{output / 'stage1_candidates.jsonl'}; first candidates:\n" + "\n".join(examples)
            + "\nThe notebook will regenerate invalid cached candidates on the next run. "
            "If outputs are truncated, increase MAX_NEW_TOKENS."
        )
    tSFT_dir = output / "stage1_tsft"
    tSFT_model = _fit_lora(
        generator,
        tokenizer,
        s1_examples,
        tSFT_dir,
        rank=8,
        alpha=16,
        dropout=0.05,
        steps=15,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        learning_rate=learning_rate,
        max_length=max_length,
        deadline=phase_deadline(1, "S1b_tsft"),
    )
    times["S1b_tsft"] = time.perf_counter() - started
    del tSFT_model, generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("S2: GRPO reasoning-chain optimization and offline selection")
    started = time.perf_counter()
    policy_model = PeftModel.from_pretrained(
        _load_base(model_name),
        tSFT_dir,
        is_trainable=True,
    )
    policy_model.load_adapter(
        tSFT_dir,
        adapter_name="reference",
        is_trainable=False,
    )
    policy_model.config.use_cache = True
    policy_model.eval()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy_model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    typed_training = [
        (typed, reasoning_prompt(typed.text, typed.quad_type, categories))
        for row in train_instances
        for typed in typed_instances(row)
    ]
    rng = random.Random(seed)
    # R_div (Eq. 11) measures dissimilarity against the Step-1 template chains.
    templates_by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in initial_candidates:
        templates_by_key[(str(row.get("instance_id")), str(row.get("quad_type")))].append(
            row.get("chain", "")
        )
    rl_losses = []
    # Table 8: GRPO uses batch size 2 with gradient accumulation 2, i.e. four
    # prompts per optimiser step. This is deliberately separate from the SFT
    # `gradient_accumulation`, which Table 9 sets independently.
    prompts_per_step = max(1, grpo_batch_size) * max(1, grpo_gradient_accumulation)
    s2_deadline = phase_deadline(2, "S2_grpo")
    # Half of S2 is reserved for re-sampling and scoring the selected chains.
    grpo_deadline = (
        None if s2_deadline is None else Deadline(budget, s2_deadline.remaining() * 0.5, "S2_grpo_updates")
    )
    for step in range(grpo_steps):
        if grpo_deadline is not None and grpo_deadline.expired():
            print(f"Stopped GRPO after {step}/{grpo_steps} steps to stay inside the budget.")
            break
        step_losses = []
        best_reward = 0.0
        optimizer.zero_grad(set_to_none=True)
        for _ in range(prompts_per_step):
            typed, prompt = rng.choice(typed_training)
            completions = _generate(
                policy_model,
                tokenizer,
                prompt,
                count=group_size,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.95,
            )
            templates = templates_by_key.get((typed.instance_id, typed.quad_type), [])
            rewards = [
                score_chain(text, typed, templates)["total"] for text in completions
            ]
            best_reward = max(best_reward, max(rewards))
            step_losses.append(
                _grpo_accumulate(
                    policy_model,
                    tokenizer,
                    prompt,
                    completions,
                    rewards,
                    clip_epsilon=clip_epsilon,
                    kl_coefficient=kl_coefficient,
                    max_length=max_length,
                    scale=1.0 / prompts_per_step,
                )
            )
        loss = sum(step_losses)
        # Every group can be degenerate (identical rewards give zero advantage),
        # and AdamW would still apply weight decay to a step with no gradient.
        if any(step_losses):
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in policy_model.parameters()
                    if parameter.requires_grad
                ],
                1.0,
            )
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        rl_losses.append(loss)
        print(f"GRPO step {step + 1}/{grpo_steps}: loss={loss:.4f}, best_reward={best_reward:.3f}")

    policy_candidates = _generation_stage(
        policy_model,
        tokenizer,
        train_instances,
        categories,
        generator_name=model_name + " (GRPO policy)",
        count=chains_per_type,
        max_new_tokens=max_new_tokens,
        output_path=output / "stage2_policy_candidates.jsonl",
        deadline=s2_deadline,
    )
    grouped_candidates = defaultdict(list)
    typed_lookup = {
        (typed.instance_id, typed.quad_type): typed
        for row in train_instances
        for typed in typed_instances(row)
    }
    for candidate in policy_candidates:
        typed = typed_lookup.get((candidate.get("instance_id"), candidate.get("quad_type")))
        if typed is None:
            continue
        grouped_candidates[(typed.instance_id, typed.quad_type)].append(candidate)
    selected = []
    for (instance_id, type_name), candidates in grouped_candidates.items():
        typed = typed_lookup[(instance_id, type_name)]
        templates = templates_by_key.get((instance_id, type_name), [])
        for candidate in candidates:
            candidate["_reward"] = score_chain(
                candidate.get("chain", ""),
                typed,
                templates,
            )["total"]
        valid = [row for row in candidates if row.get("valid")]
        selected.extend(
            sorted(valid, key=lambda row: row["_reward"], reverse=True)[
                :selected_chains_per_type
            ]
        )
    for candidate in selected:
        candidate.pop("_reward", None)
    _write_candidates(output / "stage2_selected_chains.jsonl", selected)
    policy_dir = output / "stage2_grpo_policy"
    policy_model.save_pretrained(policy_dir)
    times["S2_grpo_and_selection"] = time.perf_counter() - started
    del policy_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not selected:
        raise ValueError(
            "GRPO selection produced no valid reasoning chains; inspect stage2_policy_candidates.jsonl."
        )
    print("S3: fine-tuning the final model on selected Tree-CoT chains")
    started = time.perf_counter()
    final_tokenizer = _load_tokenizer(final_model_name)
    final_base = _load_base(final_model_name)
    final_examples = _final_examples(
        selected,
        {instance.instance_id: instance for instance in train_instances},
        categories,
    )
    if not final_examples:
        raise ValueError(
            "No Step 3 training examples were built from the selected chains; "
            "inspect stage2_selected_chains.jsonl."
        )
    final_dir = output / "final"
    final_model = _fit_lora(
        final_base,
        final_tokenizer,
        final_examples,
        final_dir,
        rank=final_lora_rank,
        # Table 9 pairs rank 32 with alpha 32, a LoRA scaling of 1. Keeping
        # alpha equal to the rank preserves that scaling if a smaller card
        # forces the rank down.
        alpha=final_lora_rank,
        dropout=0.1,
        steps=final_steps,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        learning_rate=learning_rate,
        max_length=max_length,
        deadline=phase_deadline(3, "S3_final_sft"),
    )
    final_model.save_pretrained(final_dir)
    final_tokenizer.save_pretrained(final_dir)
    (final_dir / "categories.json").write_text(
        json.dumps(categories, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    times["S3_final_sft"] = time.perf_counter() - started

    metrics = {
        "paper": "Tree-CoT-RT",
        "generation_backend": "local Hugging Face model (paper uses external API-generated chains)",
        "base_model": model_name,
        "final_model": final_model_name,
        "train_instances": len(train_instances),
        "dev_instances": len(dev_instances),
        "tree_types": sorted({typed.quad_type for row in train_instances for typed in typed_instances(row)}),
        "generated_chains": len(initial_candidates),
        "selected_chains": len(selected),
        "budget": budget.report(),
        "grpo_losses": rl_losses,
        "grpo": {
            "group_size": group_size,
            "steps": grpo_steps,
            "batch_size": grpo_batch_size,
            "gradient_accumulation": grpo_gradient_accumulation,
            "prompts_per_step": prompts_per_step,
            "clip_epsilon": clip_epsilon,
            "kl_coefficient": kl_coefficient,
            "reward_weights": dict(REWARD_WEIGHTS),
        },
        "s3_lora_rank": final_lora_rank,
        "stage_seconds": times,
        "training_seconds": sum(times.values()),
        "total_elapsed_seconds": time.perf_counter() - run_started,
        "note": (
            "Paper results use Restaurant/Laptop ACOS data, external API-generated chains, "
            "four A100 GPUs, and four random seeds. This run uses the supplied hotel-review "
            "dataset and local generation; its scores are not directly comparable."
        ),
    }
    (output / "training_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    del final_model, final_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def predict(
    instances: list[Instance],
    adapter_dir: str | Path,
    *,
    base_model: str,
    samples: int = 5,
    max_new_tokens: int = 1024,
    time_budget_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """S4 self-consistent prediction, optionally capped by a wall clock.

    Sentences left unprocessed when the budget runs out are returned with an
    empty quadruple list, so evaluation still covers the whole test split and
    the recall penalty is visible rather than hidden.
    """
    adapter = Path(adapter_dir)
    if not adapter.is_dir():
        raise FileNotFoundError(f"Trained adapter not found: {adapter}")
    tokenizer = _load_tokenizer(str(adapter))
    categories_path = adapter / "categories.json"
    categories = json.loads(categories_path.read_text(encoding="utf-8"))
    model = PeftModel.from_pretrained(_load_base(base_model), adapter)
    model.eval()
    # S4 (Sec. 3.6): the gold ASQP type is unknown at test time, so the model is
    # prompted once per sentence, samples M chains by stochastic decoding, and
    # the quadruples are decided by self-consistent majority voting (Eq. 9).
    rows: list[dict[str, Any]] = []
    total = len(instances)
    budget = Budget(time_budget_seconds)
    deadline = Deadline(budget, None, "S4_inference")
    skipped = 0
    for index, instance in enumerate(instances, start=1):
        if deadline.expired():
            skipped = total - index + 1
            rows.extend(
                {
                    "instance_id": row.instance_id,
                    "text": row.text,
                    "quadruples": [],
                    "reasoning_samples": [],
                }
                for row in instances[index - 1 :]
            )
            print(f"Inference budget reached; {skipped}/{total} sentences left unscored.")
            break
        outputs = _generate(
            model,
            tokenizer,
            reasoning_prompt(instance.text, None, categories),
            count=samples,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.95,
        )
        rows.append(
            {
                "instance_id": instance.instance_id,
                "text": instance.text,
                "quadruples": majority_vote(outputs),
                "reasoning_samples": outputs,
            }
        )
        if index % 25 == 0 or index == total:
            print(f"Predicted {index}/{total} sentences")
    return rows
