from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from .data import Instance, Quad, quad_type

# Tree-CoT-RT Table 1 / Figure 4: one type-specific reasoning path per ASQP type.
# The order of the steps IS the method, so each path below follows the paper's
# "Reasoning Path Summary" column exactly.
TREE_PATHS = {
    # EAEO: Identify A+O -> Pair -> Sentiment, Category
    "EAEO": (
        "Identify the explicit aspect; identify the explicit opinion; pair them; "
        "derive the sentiment of that pair; then derive the aspect category."
    ),
    # EAIO: A -> Infer O -> Sentiment, Category
    "EAIO": (
        "Identify the explicit aspect; infer the implicit opinion from the described "
        "events or outcomes; then derive the sentiment and the aspect category."
    ),
    # IAEO: O -> Infer A -> Sentiment, Category
    "IAEO": (
        "Identify the explicit opinion; infer the implicit aspect that the opinion "
        "targets; then derive the sentiment and the aspect category of that aspect."
    ),
    # IAIO: Context -> Infer IA + IO -> Sentiment, Category
    "IAIO": (
        "Read the full sentence context; infer both the implicit aspect and the "
        "implicit opinion from the described events; then derive the sentiment and "
        "the aspect category."
    ),
}

#: Steps the rule-based validity checker expects per type, mirroring Table 1.
TREE_STEPS = {
    "EAEO": (
        "Explicit Aspect Identification",
        "Explicit Opinion Identification",
        "Sentiment Derivation",
        "Category Derivation",
    ),
    "EAIO": (
        "Explicit Aspect Identification",
        "Implicit Opinion Inference",
        "Sentiment Derivation",
        "Category Derivation",
    ),
    "IAEO": (
        "Explicit Opinion Identification",
        "Implicit Aspect Inference",
        "Sentiment Derivation",
        "Category Derivation",
    ),
    "IAIO": (
        "Contextual Understanding",
        "Implicit Aspect and Opinion Inference",
        "Sentiment Derivation",
        "Category Derivation",
    ),
}

QUAD_SCHEMA = (
    '{"reasoning":"short evidence-grounded explanation",'
    '"quadruples":[{"aspect_term":string|null,"aspect_category":string,'
    '"opinion_term":string|null,"sentiment":string}]}'
)


def aspect_is_implicit(type_name: str) -> bool:
    return type_name.startswith("IA")


def opinion_is_implicit(type_name: str) -> bool:
    return type_name.endswith("IO")


def _type_header(type_name: str) -> str:
    return (
        f"Tree type: {type_name} (aspect is "
        f"{'implicit' if aspect_is_implicit(type_name) else 'explicit'}; opinion is "
        f"{'implicit' if opinion_is_implicit(type_name) else 'explicit'}).\n"
        f"Reasoning path: {TREE_PATHS[type_name]}\n"
    )


_TASK_HEADER = (
    "Task: predict all aspect-category-opinion-sentiment quadruples (ASQP) in the "
    "sentence. Analyze only the supplied sentence. Use null for an implicit aspect "
    "or opinion, preserve exact text spans for explicit terms, and do not invent "
    "evidence.\n"
)

_ALL_TYPES_HINT = (
    "Decide for yourself which of the four ASQP types each quadruple belongs to and "
    "follow that type's reasoning path:\n"
    + "".join(f"- {name}: {path}\n" for name, path in TREE_PATHS.items())
)


def reasoning_prompt(
    text: str,
    quad_type: str | None = None,
    categories: list[str] | None = None,
) -> str:
    """Tree-CoT prompt.

    `quad_type` selects one type-specific tree, used in S1 chain generation and
    S2 GRPO. Passing `None` gives the type-agnostic prompt of S3/S4, where the
    gold type is unknown and the model must choose the path itself.
    """
    if quad_type is not None and quad_type not in TREE_PATHS:
        raise ValueError(f"Unknown ASQP type {quad_type!r}; expected one of {sorted(TREE_PATHS)}")
    labels = f"\nAllowed aspect categories: {json.dumps(categories, ensure_ascii=False)}" if categories else ""
    if quad_type is None:
        guidance = _ALL_TYPES_HINT
        step_instruction = (
            "Write reasoning as explicit ordered steps named Step 1, Step 2, Step 3 and "
            "Step 4, covering every quadruple you report."
        )
    else:
        guidance = _type_header(quad_type)
        step_instruction = (
            "Write reasoning as explicit ordered steps named Step 1, Step 2, Step 3 and "
            "Step 4."
        )
    return (
        _TASK_HEADER
        + guidance
        + f"Sentence: {text}{labels}\n"
        + step_instruction
        + " Follow the supplied path exactly, cite words or events from the sentence, "
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


def _step_positions(reasoning: str, step_count: int) -> list[int | None]:
    positions: list[int | None] = []
    for number in range(1, step_count + 1):
        match = re.search(
            rf"(?:^|\s)(?:\*\*)?(?:step\s*)?{number}\s*[).:：\-–—]\s*",
            reasoning,
        )
        positions.append(match.start() if match else None)
    return positions


def path_constraints(text: str, quad_type: str) -> list[bool]:
    """Rule-based Tree-CoT path checker behind R_valid (Appendix C.1).

    Returns one flag per constraint: (i) required step coverage, (ii) allowable
    step order for the type, (iii) absence of contradictions. `R_valid` is the
    normalised count of satisfied constraints, so the caller gets partial credit
    for a chain that is structurally close but not perfect.
    """
    parsed = parse_quads(text)
    obj = _json_object(text)
    reasoning = ""
    if obj is not None and isinstance(obj.get("reasoning"), str):
        reasoning = obj["reasoning"]
    if parsed is None or not reasoning.strip():
        return [False, False, False]

    folded = reasoning.casefold()
    step_count = len(TREE_STEPS[quad_type])
    positions = _step_positions(folded, step_count)
    labelled = [position for position in positions if position is not None]
    # Some instruction-tuned models drop the `Step 1:` labels but still produce
    # type-correct reasoning; that counts as covered, only partial labelling does not.
    coverage = not labelled or len(labelled) == step_count
    order = labelled == sorted(labelled)

    sentiments: dict[tuple[str | None, str, str | None], str] = {}
    contradiction = False
    for quad in parsed:
        key = (quad["aspect_term"], quad["aspect_category"], quad["opinion_term"])
        if key in sentiments and sentiments[key] != quad["sentiment"]:
            contradiction = True
            break
        sentiments[key] = quad["sentiment"]
    return [coverage, order, not contradiction]


def type_matches(quad: dict[str, str | None], quad_type: str) -> bool:
    return (quad["aspect_term"] is None) == aspect_is_implicit(quad_type) and (
        quad["opinion_term"] is None
    ) == opinion_is_implicit(quad_type)


def chain_is_valid(text: str, quad_type: str) -> bool:
    parsed = parse_quads(text)
    if parsed is None or not all(path_constraints(text, quad_type)):
        return False
    return all(type_matches(quad, quad_type) for quad in parsed)


def majority_vote(
    samples: list[str],
    *,
    expected_type: str | None = None,
) -> list[dict[str, str | None]]:
    """S4 self-consistency (Eq. 9): keep the quadruples most samples agree on."""
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
            if expected_type and not type_matches(quad, expected_type):
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
    """Split a sentence into one sub-instance per ASQP type it contains.

    S1 and S2 are type-specific, so a sentence carrying both an EAEO and an IAIO
    quadruple contributes one Tree-CoT chain per type.
    """
    result = []
    for type_name in TREE_PATHS:
        quads = tuple(
            quad
            for quad in instance.quads
            if quad_type(quad.aspect_implicit, quad.opinion_implicit) == type_name
        )
        if quads:
            result.append(
                Instance(
                    instance.instance_id,
                    instance.text,
                    quads,
                    aspect_is_implicit(type_name),
                    opinion_is_implicit(type_name),
                )
            )
    return result
