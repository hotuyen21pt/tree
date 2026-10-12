from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Quad:
    aspect_term: str | None
    aspect_category: str
    opinion_term: str | None
    sentiment: str
    aspect_implicit: bool = False
    opinion_implicit: bool = False

    def __post_init__(self):
        if self.aspect_term is None and not self.aspect_implicit:
            object.__setattr__(self, "aspect_implicit", True)
        if self.opinion_term is None and not self.opinion_implicit:
            object.__setattr__(self, "opinion_implicit", True)

    def as_dict(self) -> dict[str, str | None]:
        return {
            "aspect_term": self.aspect_term,
            "aspect_category": self.aspect_category,
            "opinion_term": self.opinion_term,
            "sentiment": self.sentiment,
        }


@dataclass(frozen=True)
class Instance:
    instance_id: str
    text: str
    quads: tuple[Quad, ...]
    aspect_implicit: bool
    opinion_implicit: bool

    @property
    def quad_type(self) -> str:
        return quad_type(self.aspect_implicit, self.opinion_implicit)


def quad_type(aspect_implicit: bool, opinion_implicit: bool) -> str:
    """One of the four Tree-CoT-RT ASQP types: EAEO, EAIO, IAEO, IAIO."""
    return ("IA" if aspect_implicit else "EA") + ("IO" if opinion_implicit else "EO")


def _as_optional_text(value: Any, field: str, source: Path, line: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{source}:{line}: {field} must be a string or null")
    return value


def read_jsonl(path: str | Path) -> list[Instance]:
    source = Path(path)
    instances: list[Instance] = []
    seen_ids: set[str] = set()
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(raw, dict):
                raise ValueError(f"{source}:{line_number}: each record must be an object")
            text = raw.get("text")
            if not isinstance(text, str):
                raise ValueError(f"{source}:{line_number}: text must be a string")
            instance_id = str(raw.get("instance_id", f"{source.stem}-{line_number}"))
            if instance_id in seen_ids:
                raise ValueError(f"{source}:{line_number}: duplicate instance_id {instance_id!r}")
            seen_ids.add(instance_id)
            raw_quads = raw.get("quads")
            if not isinstance(raw_quads, list):
                raise ValueError(f"{source}:{line_number}: quads must be a list")
            quads: list[Quad] = []
            for raw_quad in raw_quads:
                if not isinstance(raw_quad, dict):
                    raise ValueError(f"{source}:{line_number}: every quad must be an object")
                aspect = _as_optional_text(raw_quad.get("aspect_term"), "aspect_term", source, line_number)
                opinion = _as_optional_text(raw_quad.get("opinion_term"), "opinion_term", source, line_number)
                category = raw_quad.get("aspect_category")
                sentiment = raw_quad.get("sentiment")
                if not isinstance(category, str) or not category:
                    raise ValueError(f"{source}:{line_number}: aspect_category must be non-empty")
                if not isinstance(sentiment, str) or not sentiment:
                    raise ValueError(f"{source}:{line_number}: sentiment must be non-empty")
                aspect_implicit = raw_quad.get("aspect_implicit", aspect is None)
                opinion_implicit = raw_quad.get("opinion_implicit", opinion is None)
                if not isinstance(aspect_implicit, bool) or not isinstance(opinion_implicit, bool):
                    raise ValueError(f"{source}:{line_number}: implicit flags must be booleans")
                quads.append(
                    Quad(
                        aspect,
                        category,
                        opinion,
                        sentiment,
                        aspect_implicit or aspect is None,
                        opinion_implicit or opinion is None,
                    )
                )

            aspect_implicit = any(
                quad.aspect_implicit for quad in quads
            )
            opinion_implicit = any(
                quad.opinion_implicit for quad in quads
            )
            instances.append(
                Instance(
                    instance_id,
                    text,
                    tuple(quads),
                    aspect_implicit,
                    opinion_implicit,
                )
            )
    return instances


def category_labels(instances: list[Instance]) -> list[str]:
    return sorted({quad.aspect_category for row in instances for quad in row.quads})


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
