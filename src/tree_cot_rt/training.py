from __future__ import annotations

import json
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
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=_dtype(),
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
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
        args=TrainingArguments(
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


def _write_candidates(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_candidates(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
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
            and row.get("generator_model") == generator_name
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
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
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
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
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
        ratio = torch.exp(
            torch.clamp(new_logp - old_logps[index], min=-20, max=20)
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
    # Table 8: batch size 2 with gradient accumulation 2, i.e. four prompts per
    # optimiser step rather than one.
    prompts_per_step = max(1, grpo_batch_size) * max(1, gradient_accumulation)
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
        rank=32,
        alpha=32,
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
            "gradient_accumulation": gradient_accumulation,
            "clip_epsilon": clip_epsilon,
            "kl_coefficient": kl_coefficient,
            "reward_weights": dict(REWARD_WEIGHTS),
        },
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
