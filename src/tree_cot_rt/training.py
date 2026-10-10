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
    TrainingArguments,
)

from .data import Instance
from .reasoning import (
    extract_reasoning,
    majority_vote,
    chain_is_valid,
    parse_quads,
    reasoning_prompt,
    typed_instances,
)
from .rewards import group_advantages, score_chain


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
) -> list[dict[str, Any]]:
    cached = _read_candidates(output_path)
    cached_by_id = defaultdict(list)
    for row in cached:
        cached_by_id[(row.get("instance_id"), row.get("quad_type"))].append(row)
    rows = []
    total = sum(1 for _ in _examples_for_typed(instances, categories))
    for index, (typed, prompt) in enumerate(_examples_for_typed(instances, categories), 1):
        prior = [
            row
            for row in cached_by_id[(typed.instance_id, typed.quad_type)]
            if row.get("prompt") == prompt
            and row.get("generator_model") == generator_name
        ]
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
        if index % 25 == 0 or index == total:
            _write_candidates(output_path, rows + cached)
            print(f"Generated Tree-CoT chains for {index}/{total} typed instances")
    unique = {}
    for row in rows:
        unique[(row["instance_id"], row["quad_type"], row["chain"])] = row
    result = list(unique.values())
    _write_candidates(output_path, result)
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


def _grpo_update(
    model,
    tokenizer,
    instance: Instance,
    prompt: str,
    completions: list[str],
    rewards: list[float],
    optimizer,
    *,
    clip_epsilon: float,
    kl_coefficient: float,
    max_length: int,
):
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
        loss = (
            -torch.minimum(unclipped, clipped)
            + kl_coefficient * (new_logp - ref_logps[index])
        ) / len(completions)
        loss.backward()
        losses.append(float(loss.detach().cpu()))
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        1.0,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
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
    selected_chains_per_type: int = 2,
    final_steps: int = 10,
    batch_size: int = 4,
    gradient_accumulation: int = 2,
    learning_rate: float = 5e-5,
    max_length: int = 2048,
    max_new_tokens: int = 1024,
    seed: int = 444,
) -> dict[str, Any]:
    if not train_instances or not dev_instances:
        raise ValueError("Both train and development splits must contain records")
    random.seed(seed)
    torch.manual_seed(seed)
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
    )
    times["S1a_generation"] = time.perf_counter() - started

    print("S1-b: tree-supervised LoRA initialization")
    started = time.perf_counter()
    s1_examples = _sft_examples(initial_candidates, valid_only=True)
    if not s1_examples:
        raise ValueError(
            "No parseable Tree-CoT chains were generated. Inspect stage1_candidates.jsonl, "
            "increase max_new_tokens, or choose a JSON-instruction-tuned model."
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
    rl_losses = []
    for step in range(grpo_steps):
        typed, prompt = rng.choice(typed_training)
        completions = _generate(
            policy_model,
            tokenizer,
            prompt,
            count=10,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.95,
        )
        rewards = [
            score_chain(text, typed, completions)["total"]
            for text in completions
        ]
        loss = _grpo_update(
            policy_model,
            tokenizer,
            typed,
            prompt,
            completions,
            rewards,
            optimizer,
            clip_epsilon=0.2,
            kl_coefficient=0.01,
            max_length=max_length,
        )
        rl_losses.append(loss)
        print(f"GRPO step {step + 1}/{grpo_steps}: loss={loss:.4f}, best_reward={max(rewards):.3f}")

    policy_candidates = _generation_stage(
        policy_model,
        tokenizer,
        train_instances,
        categories,
        generator_name=model_name + " (GRPO policy)",
        count=chains_per_type,
        max_new_tokens=max_new_tokens,
        output_path=output / "stage2_policy_candidates.jsonl",
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
        group = [row.get("chain", "") for row in candidates]
        for candidate in candidates:
            candidate["_reward"] = score_chain(
                candidate.get("chain", ""),
                typed,
                group,
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
    final_examples = _sft_examples(selected, valid_only=False)
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
        "grpo_losses": rl_losses,
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
) -> list[dict[str, Any]]:
    adapter = Path(adapter_dir)
    if not adapter.is_dir():
        raise FileNotFoundError(f"Trained adapter not found: {adapter}")
    tokenizer = _load_tokenizer(str(adapter))
    categories_path = adapter / "categories.json"
    categories = json.loads(categories_path.read_text(encoding="utf-8"))
    model = PeftModel.from_pretrained(_load_base(base_model), adapter)
    model.eval()
    by_id = {instance.instance_id: instance for instance in instances}
    predictions: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for instance in instances:
        typed_rows = [
            (quad_type, reasoning_prompt(instance.text, quad_type, categories))
            for quad_type in ("EE", "EA", "IE", "IA")
        ]
        for quad_type, prompt in typed_rows:
            outputs = _generate(
                model,
                tokenizer,
                prompt,
                count=samples,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.95,
            )
            predictions[instance.instance_id][quad_type].extend(outputs)
    return [
        {
            "instance_id": instance_id,
            "text": by_id[instance_id].text,
            "quadruples": [
                quad
                for quad_type, samples_for_type in predictions[instance_id].items()
                for quad in majority_vote(
                    samples_for_type,
                    expected_type=quad_type,
                )
            ],
            "reasoning_samples": [
                sample
                for samples_for_type in predictions[instance_id].values()
                for sample in samples_for_type
            ],
        }
        for instance_id in by_id
    ]
