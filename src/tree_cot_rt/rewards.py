from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

from .data import Instance
from .reasoning import TREE_PATHS, _json_object, chain_is_valid, parse_quads

REWARD_WEIGHTS = {
    "consistency": 0.25,
    "validity": 0.25,
    "implicit": 0.25,
    "diversity": 0.25,
}


def normalize(value: object) -> str | None:
    if value is None:
        return None
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value))).strip().casefold()


def _quad_key(quad: dict[str, object]) -> tuple[str | None, ...]:
    return tuple(
        normalize(quad.get(field))
        for field in ("aspect_term", "aspect_category", "opinion_term", "sentiment")
    )


def _f1(predicted: list[dict[str, object]], gold: list[dict[str, object]]) -> float:
    predicted_counts = Counter(_quad_key(q) for q in predicted)
    gold_counts = Counter(_quad_key(q) for q in gold)
    true_positive = sum((predicted_counts & gold_counts).values())
    precision = true_positive / sum(predicted_counts.values()) if predicted_counts else 0.0
    recall = true_positive / sum(gold_counts.values()) if gold_counts else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _implicit_alignment(chain: str, quad_type: str) -> float:
    if quad_type == "EE":
        return 0.0
    obj = _json_object(chain)
    reasoning = str(obj.get("reasoning", "")) if obj else ""
    cues = (
        "because", "therefore", "implies", "suggests", "indicates", "context",
        "evidence", "due to", "since", "which shows", "so that",
        "vì", "bởi vì", "do đó", "cho thấy", "dẫn đến", "nguyên nhân",
    )
    return float(any(cue in reasoning.casefold() for cue in cues))


def _diversity(chain: str, other_chains: Sequence[str]) -> float:
    if not other_chains:
        return 1.0
    others = list(other_chains)
    try:
        others.remove(chain)
    except ValueError:
        pass
    if not others:
        return 1.0
    tokens = set(re.findall(r"\w+", chain.casefold()))
    similarities = []
    for other in others:
        other_tokens = set(re.findall(r"\w+", other.casefold()))
        union = tokens | other_tokens
        similarities.append(len(tokens & other_tokens) / len(union) if union else 1.0)
    return 1.0 - max(similarities)


def score_chain(
    chain: str,
    instance: Instance,
    group: Sequence[str] = (),
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    active_weights = weights or REWARD_WEIGHTS
    if set(active_weights) != set(REWARD_WEIGHTS):
        raise ValueError(f"Reward weights must have keys {sorted(REWARD_WEIGHTS)}")
    if any(value < 0 for value in active_weights.values()) or abs(sum(active_weights.values()) - 1.0) > 1e-6:
        raise ValueError("Reward weights must be non-negative and sum to 1")

    predicted = parse_quads(chain)
    consistency = _f1(predicted or [], [quad.as_dict() for quad in instance.quads])
    validity = float(chain_is_valid(chain, instance.quad_type))
    implicit = _implicit_alignment(chain, instance.quad_type) if validity else 0.0
    diversity = _diversity(chain, group)
    components = {
        "consistency": consistency,
        "validity": validity,
        "implicit": implicit,
        "diversity": diversity,
    }
    components["total"] = sum(active_weights[name] * value for name, value in components.items())
    return components


def group_advantages(rewards: Sequence[float]) -> list[float]:
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    variance = sum((score - mean) ** 2 for score in rewards) / len(rewards)
    standard_deviation = variance**0.5
    if standard_deviation < 1e-8:
        return [0.0 for _ in rewards]
    return [(score - mean) / standard_deviation for score in rewards]


def valid_reasoning_path(text: str, instance: Instance) -> bool:
    return chain_is_valid(text, instance.quad_type) and instance.quad_type in TREE_PATHS
