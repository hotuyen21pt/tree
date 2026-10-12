from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from .data import Instance, Quad, quad_type

TREE_PATHS = {
    "EE": (
        "Identify the explicit aspect and opinion together; infer their sentiment; "
        "then assign the aspect category."
    ),
    "EA": (
        "Identify the explicit aspect and sentiment-bearing context; assign the category; "
        "then infer the implicit opinion and its polarity from the sentence."
    ),
    "IE": (
        "Identify the explicit opinion and sentiment; infer the implicit aspect from its "
        "target and context; then assign the aspect category."
    ),
    "IA": (
        "Use the full sentence context to infer the implicit aspect and implicit opinion; "
        "then determine the sentiment and aspect category, citing textual evidence."
    ),
}

QUAD_SCHEMA = (
    '{"reasoning":"short evidence-grounded explanation",'
    '"quadruples":[{"aspect_term":string|null,"aspect_category":string,'
    '"opinion_term":string|null,"sentiment":string}]}'
)


def reasoning_prompt(text: str, quad_type: str, categories: list[str] | None = None) -> str:
    if quad_type not in TREE_PATHS:
        raise ValueError(f"Unknown ASQP type {quad_type!r}; expected one of {sorted(TREE_PATHS)}")
    labels = f"\nAllowed aspect categories: {json.dumps(categories, ensure_ascii=False)}" if categories else ""
    return (
        "Task: predict all aspect-category-opinion-sentiment quadruples (ASQP) in the "
        "sentence. Analyze only the supplied sentence. Use null for an implicit aspect "
        "or opinion, preserve exact text spans for explicit terms, and do not invent "
        "evidence.\n"
        f"Tree type: {quad_type} (aspect is {'implicit' if quad_type[0] == 'I' else 'explicit'}; "
        f"opinion is {'implicit' if quad_type[1] == 'A' else 'explicit'}).\n"
        f"Reasoning path: {TREE_PATHS[quad_type]}\n"
        f"Sentence: {text}{labels}\n"
        "Write reasoning as explicit ordered steps named Step 1, Step 2, and Step 3"
        + (" and Step 4" if quad_type != "EE" else "")
        + ". Follow the supplied path exactly, cite words or events from the sentence, "
        "and explain rather than repeat the answer. Return one JSON object only, with a "
        "reasoning field and a quadruples list "
        f"using this schema: {QUAD_SCHEMA}"
    )


def gold_target(instance: Instance) -> str:
    return json.dumps(
        {
            "reasoning": " ".join(
                f"Tree type {quad_type(quad.aspect_implicit, quad.opinion_implicit)}: "
                f"{TREE_PATHS[quad_type(quad.aspect_implicit, quad.opinion_implicit)]}"
                for quad in instance.quads
            ) + " Ground each decision in the sentence.",
            "quadruples": [quad.as_dict() for quad in instance.quads],
        },
        ensure_ascii=False,
    )


def _json_object(text: str) -> dict[str, Any] | None:
    candidates = [text.strip()]
    candidates.extend(re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            start, end = candidate.find("{"), candidate.rfind("}")
            if start < 0 or end <= start:
                continue
            try:
                value = json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
        if isinstance(value, dict):
            return value
    return None


def extract_reasoning(text: str) -> str:
    obj = _json_object(text)
    if obj is not None and isinstance(obj.get("reasoning"), str):
        return obj["reasoning"]
    return text.strip()


def parse_quads(text: str) -> list[dict[str, str | None]] | None:
    value = _json_object(text)
    if value is None:
        return None
    quads = value.get("quadruples", value.get("quads"))
    if not isinstance(quads, list):
        return None
    result = []
    required = ("aspect_term", "aspect_category", "opinion_term", "sentiment")
    for item in quads:
        if not isinstance(item, dict):
            return None
        if any(key not in item for key in required):
            return None
        if any(
            item[key] is not None and not isinstance(item[key], str)
            for key in ("aspect_term", "opinion_term")
        ):
            return None
        if any(not isinstance(item[key], str) or not item[key] for key in ("aspect_category", "sentiment")):
            return None
        result.append({key: item[key] for key in required})
    return result


def chain_is_valid(text: str, quad_type: str) -> bool:
    parsed = parse_quads(text)
    obj = _json_object(text)
    if (
        parsed is None
        or obj is None
        or not isinstance(obj.get("reasoning"), str)
        or not obj["reasoning"].strip()
    ):
        return False
    reasoning = obj["reasoning"].casefold()
    step_count = 3 if quad_type == "EE" else 4
    # Some instruction-tuned models ignore the requested `Step 1:` labels but
    # still return useful, type-correct reasoning. Accept that free-form text.
    # If the model does use numbered step labels, require the complete ordered
    # sequence so malformed Tree-CoT is still rejected.
    step_positions = []
    for number in range(1, step_count + 1):
        match = re.search(
            rf"(?:^|\s)(?:\*\*)?(?:step\s*)?{number}\s*[).:：\-–—]\s*",
            reasoning,
        )
        step_positions.append(match.start() if match else None)
    if any(position is not None for position in step_positions):
        if any(position is None for position in step_positions):
            return False
        ordered_positions = [position for position in step_positions if position is not None]
        if ordered_positions != sorted(ordered_positions):
            return False
    quad_sentiments = {}
    for quad in parsed:
        key = (quad["aspect_term"], quad["aspect_category"], quad["opinion_term"])
        if key in quad_sentiments and quad_sentiments[key] != quad["sentiment"]:
            return False
        quad_sentiments[key] = quad["sentiment"]
    return all(
        (quad["aspect_term"] is None) == (quad_type[0] == "I")
        and (quad["opinion_term"] is None) == (quad_type[1] == "A")
        for quad in parsed
    )


def majority_vote(
    samples: list[str],
    *,
    expected_type: str | None = None,
) -> list[dict[str, str | None]]:
    counts: Counter[tuple[str | None, str | None, str | None, str | None]] = Counter()
    examples: dict[tuple[str | None, str | None, str | None, str | None], dict[str, str | None]] = {}
    valid_sample_count = 0
    for sample in samples:
        parsed = parse_quads(sample)
        if parsed is None:
            continue
        sample_keys = set()
        has_expected_type = False
        for quad in parsed:
            if expected_type and (
                (quad["aspect_term"] is None) != (expected_type[0] == "I")
                or (quad["opinion_term"] is None) != (expected_type[1] == "A")
            ):
                continue
            has_expected_type = True
            key = (
                quad["aspect_term"],
                quad["aspect_category"],
                quad["opinion_term"],
                quad["sentiment"],
            )
            sample_keys.add(key)
            examples[key] = quad
        if parsed and not has_expected_type:
            continue
        valid_sample_count += 1
        counts.update(sample_keys)
    if not counts:
        return []
    threshold = valid_sample_count // 2 + 1
    selected = [key for key, count in counts.items() if count >= threshold]
    return [examples[key] for key in sorted(selected, key=lambda key: tuple(str(v) for v in key))]


def quads_from_instance(instance: Instance) -> list[dict[str, str | None]]:
    return [quad.as_dict() for quad in instance.quads]


def typed_instances(instance: Instance) -> list[Instance]:
    result = []
    for quad_type_name in TREE_PATHS:
        quads = tuple(
            quad
            for quad in instance.quads
            if quad_type(quad.aspect_implicit, quad.opinion_implicit) == quad_type_name
        )
        if quads:
            result.append(
                Instance(
                    instance.instance_id,
                    instance.text,
                    quads,
                    quad_type_name[0] == "I",
                    quad_type_name[1] == "A",
                )
            )
    return result
