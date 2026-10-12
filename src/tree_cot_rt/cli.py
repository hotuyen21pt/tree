from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import unicodedata

from .data import Instance, read_jsonl, write_jsonl
from .reasoning import parse_quads
from .training import predict, train


def _gold_map(instances: list[Instance]) -> dict[str, list[dict[str, Any]]]:
    return {
        instance.instance_id: [quad.as_dict() for quad in instance.quads]
        for instance in instances
    }


def _normalize(value: object) -> object:
    if value is None:
        return None
    return " ".join(unicodedata.normalize("NFC", str(value)).casefold().split())


def _quad_key(quad: dict[str, Any]) -> tuple[object, ...]:
    return tuple(
        _normalize(quad.get(field))
        for field in ("aspect_term", "aspect_category", "opinion_term", "sentiment")
    )


def exact_f1(
    gold: dict[str, list[dict[str, Any]]],
    predicted: dict[str, list[dict[str, Any]]],
) -> dict[str, float | int]:
    if set(gold) != set(predicted):
        raise ValueError("Gold and predictions must contain the same instance IDs")
    true_positives = predicted_count = gold_count = 0
    for instance_id, gold_quads in gold.items():
        expected = {_quad_key(row) for row in gold_quads}
        actual = {_quad_key(row) for row in predicted[instance_id]}
        true_positives += len(expected & actual)
        gold_count += len(expected)
        predicted_count += len(actual)
    precision = true_positives / predicted_count if predicted_count else 0.0
    recall = true_positives / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": true_positives,
        "predicted": predicted_count,
        "gold": gold_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _train(args: argparse.Namespace) -> None:
    train_rows = read_jsonl(args.train)
    dev_rows = read_jsonl(args.dev)
    metrics = train(
        train_rows,
        dev_rows,
        args.output_dir,
        model_name=args.model,
        final_model_name=args.final_model,
        chains_per_type=args.chains_per_type,
        grpo_steps=args.grpo_steps,
        group_size=args.group_size,
        grpo_batch_size=args.grpo_batch_size,
        grpo_gradient_accumulation=args.grpo_gradient_accumulation,
        clip_epsilon=args.clip_epsilon,
        kl_coefficient=args.kl_coefficient,
        selected_chains_per_type=args.selected_chains_per_type,
        final_steps=args.final_steps,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def _predict(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    predictions = predict(
        rows,
        args.adapter_dir,
        base_model=args.base_model,
        samples=args.samples,
        max_new_tokens=args.max_new_tokens,
    )
    write_jsonl(args.output, predictions)
    print(f"Wrote {len(predictions)} predictions to {args.output}")


def _evaluate(args: argparse.Namespace) -> None:
    gold_rows = read_jsonl(args.gold)
    predictions = {}
    with Path(args.predictions).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{args.predictions}:{line_number}: invalid JSON") from error
            instance_id = row.get("instance_id")
            quads = row.get("quadruples")
            if not isinstance(instance_id, str) or not isinstance(quads, list):
                raise ValueError(
                    f"{args.predictions}:{line_number}: expected instance_id and quadruples"
                )
            if instance_id in predictions:
                raise ValueError(f"{args.predictions}:{line_number}: duplicate instance_id")
            normalized = []
            for quad in quads:
                if not isinstance(quad, dict):
                    raise ValueError(f"{args.predictions}:{line_number}: invalid quadruple")
                parsed = parse_quads(json.dumps({"quadruples": [quad]}, ensure_ascii=False))
                if parsed is None:
                    raise ValueError(f"{args.predictions}:{line_number}: invalid quadruple fields")
                normalized.extend(parsed)
            predictions[instance_id] = normalized
    metrics = exact_f1(_gold_map(gold_rows), predictions)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tree-cot-rt",
        description="Tree-CoT-RT aspect-category-opinion-sentiment prediction",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="run S1 multi-path, S2 GRPO, and S3 SFT")
    train_parser.add_argument("--train", required=True)
    train_parser.add_argument("--dev", required=True)
    train_parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    train_parser.add_argument("--final-model", default="Qwen/Qwen2.5-7B")
    train_parser.add_argument("--output-dir", default="runs/tree_cot_rt")
    train_parser.add_argument("--chains-per-type", type=int, default=10)
    train_parser.add_argument("--grpo-steps", type=int, default=10)
    train_parser.add_argument("--group-size", type=int, default=10, help="GRPO group size G")
    train_parser.add_argument("--grpo-batch-size", type=int, default=2)
    train_parser.add_argument("--grpo-gradient-accumulation", type=int, default=2)
    train_parser.add_argument("--clip-epsilon", type=float, default=0.2)
    train_parser.add_argument("--kl-coefficient", type=float, default=0.01)
    train_parser.add_argument("--selected-chains-per-type", type=int, default=2)
    train_parser.add_argument("--final-steps", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--gradient-accumulation", type=int, default=2)
    train_parser.add_argument("--learning-rate", type=float, default=5e-5)
    train_parser.add_argument("--max-length", type=int, default=2048)
    train_parser.add_argument("--max-new-tokens", type=int, default=1024)
    train_parser.add_argument("--seed", type=int, default=444)
    train_parser.set_defaults(func=_train)

    predict_parser = commands.add_parser("predict", help="predict with the final LoRA adapter")
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--adapter-dir", required=True)
    predict_parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B")
    predict_parser.add_argument("--output", required=True)
    predict_parser.add_argument(
        "--samples", type=int, default=5, help="self-consistency samples M (Sec. 3.6)"
    )
    predict_parser.add_argument("--max-new-tokens", type=int, default=1024)
    predict_parser.set_defaults(func=_predict)

    evaluate_parser = commands.add_parser("evaluate", help="compute exact quadruple micro-F1")
    evaluate_parser.add_argument("--gold", required=True)
    evaluate_parser.add_argument("--predictions", required=True)
    evaluate_parser.set_defaults(func=_evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
