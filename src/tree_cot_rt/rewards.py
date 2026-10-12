from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from itertools import permutations

from .data import Instance
from .reasoning import (
    TREE_PATHS,
    _json_object,
    aspect_is_implicit,
    opinion_is_implicit,
    parse_quads,
    path_constraints,
)

# Tree-CoT-RT Table 8: lambda_cons = lambda_valid = lambda_impl = lambda_div = 0.25.
REWARD_WEIGHTS = {
    "consistency": 0.25,
    "validity": 0.25,
    "implicit": 0.25,
    "diversity": 0.25,
}

#: Eq. 10 element weights w_A = w_C = w_O = w_S = 0.25.
ELEMENT_FIELDS = ("aspect_term", "aspect_category", "opinion_term", "sentiment")
ELEMENT_WEIGHT = 1.0 / len(ELEMENT_FIELDS)

#: Highest number of quads matched by brute-force bipartite matching before
#: falling back to a greedy assignment. 7! = 5040 pairings is still cheap.
_EXACT_MATCH_LIMIT = 7


def normalize(value: object) -> str | None:
    if value is None:
        return None
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value))).strip().casefold()


def _quad_key(quad: dict[str, object]) -> tuple[str | None, ...]:
    return tuple(normalize(quad.get(field)) for field in ELEMENT_FIELDS)


def _element_score(predicted: dict[str, object], gold: dict[str, object]) -> float:
    """Eq. 10 for a single quad pair: weighted count of matching elements."""
    return sum(
        ELEMENT_WEIGHT
        for field in ELEMENT_FIELDS
        if normalize(predicted.get(field)) == normalize(gold.get(field))
    )


def _best_assignment(
    predicted: list[dict[str, object]], gold: list[dict[str, object]]
) -> float:
    """Maximum bipartite matching between predicted and gold quads."""
    if len(gold) <= _EXACT_MATCH_LIMIT and len(predicted) <= _EXACT_MATCH_LIMIT:
        short, long = (predicted, gold) if len(predicted) <= len(gold) else (gold, predicted)
        return max(
            sum(_element_score(short[index], row) for index, row in enumerate(ordering))
            for ordering in permutations(long, len(short))
        )
    remaining = list(range(len(gold)))
    total = 0.0
    for row in predicted:
        if not remaining:
            break
        best = max(remaining, key=lambda index: _element_score(row, gold[index]))
        total += _element_score(row, gold[best])
        remaining.remove(best)
    return total


def consistency(predicted: list[dict[str, object]], gold: list[dict[str, object]]) -> float:
    """R_cons (Eq. 10): element-wise alignment, averaged over the gold quads.

    Partial credit matters here: a chain that gets three of four elements right
    must out-score one that gets none, otherwise GRPO sees no gradient early on.
    """
    if not gold:
        return 0.0
    if not predicted:
        return 0.0
    matched = _best_assignment(predicted, gold)
    # Normalise by the larger set so spurious extra quads are penalised too.
    return matched / max(len(gold), len(predicted))


def _implicit_alignment(chain: str, instance: Instance) -> float:
    """R_impl (Appendix C.1): normalised count of satisfied cue-link rules.

    The paper requires the chain to link at least one *sentence-grounded* cue
    span to the implicit element it infers, so a bare discourse connective is
    not enough on its own.
    """
    type_name = instance.quad_type
    if not aspect_is_implicit(type_name) and not opinion_is_implicit(type_name):
        return 0.0  # R_impl(o) = 0 for EAEO.
    obj = _json_object(chain)
    reasoning = str(obj.get("reasoning", "")) if obj else ""
    if not reasoning.strip():
        return 0.0
    folded = reasoning.casefold()

    rules: list[bool] = []
    # (1) The chain names the element it had to infer.
    if aspect_is_implicit(type_name):
        rules.append(any(word in folded for word in ("implicit aspect", "aspect is implicit", "unstated aspect")))
    if opinion_is_implicit(type_name):
        rules.append(any(word in folded for word in ("implicit opinion", "opinion is implicit", "unstated opinion")))
    # (2) The chain justifies the inference rather than asserting it.
    cues = (
        "because", "therefore", "implies", "suggests", "indicates", "evidence",
        "due to", "since", "shows", "so that", "points to", "infer",
        "vì", "bởi vì", "do đó", "cho thấy", "dẫn đến", "nguyên nhân", "suy ra",
    )
    rules.append(any(cue in folded for cue in cues))
    # (3) The justification is grounded in a span of this sentence.
    rules.append(_cites_sentence_span(folded, instance.text))
    return sum(1.0 for rule in rules if rule) / len(rules)


_STOPWORDS = frozenset(
    "a an and are as at be been but by for from had has have he i in is it its of on "
    "or she that the their them they this to was were will with you your not no".split()
)


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\w+", text.casefold())
        if len(token) > 2 and token not in _STOPWORDS
    }


def _cites_sentence_span(reasoning: str, sentence: str) -> bool:
    """True when the chain quotes or reuses content words from the sentence."""
    quoted = re.findall(r'"([^"]{3,})"|“([^”]{3,})”|\'([^\']{3,})\'', reasoning)
    folded_sentence = sentence.casefold()
    for groups in quoted:
        span = next((group for group in groups if group), "").casefold().strip()
        if span and span in folded_sentence:
            return True
    sentence_tokens = _content_tokens(sentence)
    if not sentence_tokens:
        return False
    shared = sentence_tokens & _content_tokens(reasoning)
    return len(shared) >= min(2, len(sentence_tokens))


def _similarity(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"\w+", left.casefold()))
    right_tokens = set(re.findall(r"\w+", right.casefold()))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def _diversity(chain: str, templates: Sequence[str]) -> float:
    """R_div (Eq. 11): 1 - max similarity against the Step-1 template chains."""
    others = [other for other in templates if other != chain]
    if not others:
        return 1.0
    return 1.0 - max(_similarity(chain, other) for other in others)


def score_chain(
    chain: str,
    instance: Instance,
    templates: Sequence[str] = (),
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Rule-based reward R(o) of Eq. 2 / Eq. 12.

    `templates` is the Step-1 template chain set T_temp that R_div measures
    dissimilarity against.
    """
    active_weights = weights or REWARD_WEIGHTS
    if set(active_weights) != set(REWARD_WEIGHTS):
        raise ValueError(f"Reward weights must have keys {sorted(REWARD_WEIGHTS)}")
    if any(value < 0 for value in active_weights.values()) or abs(sum(active_weights.values()) - 1.0) > 1e-6:
        raise ValueError("Reward weights must be non-negative and sum to 1")

    predicted = parse_quads(chain)
    constraints = path_constraints(chain, instance.quad_type)
    components = {
        "consistency": consistency(predicted or [], [quad.as_dict() for quad in instance.quads]),
        "validity": sum(1.0 for flag in constraints if flag) / len(constraints),
        "implicit": _implicit_alignment(chain, instance),
        "diversity": _diversity(chain, templates),
    }
    components["total"] = sum(active_weights[name] * value for name, value in components.items())
    return components


def group_advantages(rewards: Sequence[float]) -> list[float]:
    """Group-relative advantage of Eq. 3-5."""
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    variance = sum((score - mean) ** 2 for score in rewards) / len(rewards)
    standard_deviation = variance**0.5
    if standard_deviation < 1e-8:
        return [0.0 for _ in rewards]
    return [(score - mean) / standard_deviation for score in rewards]


def valid_reasoning_path(text: str, instance: Instance) -> bool:
    from .reasoning import chain_is_valid

    return chain_is_valid(text, instance.quad_type) and instance.quad_type in TREE_PATHS
