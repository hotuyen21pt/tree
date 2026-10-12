"""Wall-clock budgeting so a run always finishes inside a Kaggle session.

A Kaggle GPU session is capped (9 hours here), and an overrun loses the whole
run: the notebook is killed before the adapter is written. Every long phase
therefore takes a deadline and stops cleanly at it rather than being cut off.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .timing import format_duration


@dataclass
class Budget:
    """A wall-clock allowance that phases can carve sub-budgets out of."""

    total_seconds: float | None
    started_at: float = field(default_factory=time.perf_counter)
    #: Phases that hit their deadline instead of running to completion.
    truncated: list[str] = field(default_factory=list)

    @property
    def unlimited(self) -> bool:
        return self.total_seconds is None

    def elapsed(self) -> float:
        return time.perf_counter() - self.started_at

    def remaining(self) -> float:
        if self.total_seconds is None:
            return float("inf")
        return self.total_seconds - self.elapsed()

    def expired(self) -> bool:
        return self.remaining() <= 0

    def share(self, fraction: float, *, minimum: float = 60.0) -> float | None:
        """Seconds for a phase that should take `fraction` of what is left."""
        if self.total_seconds is None:
            return None
        return max(minimum, self.remaining() * fraction)

    def note_truncated(self, phase: str) -> None:
        if phase not in self.truncated:
            self.truncated.append(phase)
        print(
            f"[budget] {phase} stopped at its deadline; "
            f"{format_duration(max(0.0, self.remaining()))} left in the session budget.",
            flush=True,
        )

    def report(self) -> dict[str, Any]:
        return {
            "total_seconds": self.total_seconds,
            "elapsed_seconds": self.elapsed(),
            "remaining_seconds": None if self.total_seconds is None else self.remaining(),
            "truncated_phases": list(self.truncated),
        }


class Deadline:
    """A countdown for one phase, clamped to what the session has left."""

    def __init__(self, budget: Budget, seconds: float | None, name: str) -> None:
        self.budget = budget
        self.name = name
        self.started_at = time.perf_counter()
        if seconds is None and budget.unlimited:
            self.seconds: float | None = None
        else:
            allowed = budget.remaining() if seconds is None else seconds
            self.seconds = max(0.0, min(allowed, budget.remaining()))
        self._reported = False

    def elapsed(self) -> float:
        return time.perf_counter() - self.started_at

    def remaining(self) -> float:
        if self.seconds is None:
            return float("inf")
        return self.seconds - self.elapsed()

    def expired(self) -> bool:
        if self.remaining() > 0:
            return False
        if not self._reported:
            self._reported = True
            self.budget.note_truncated(self.name)
        return True

    def describe(self) -> str:
        if self.seconds is None:
            return f"{self.name}: no time limit"
        return f"{self.name}: {format_duration(self.seconds)} allowed"


def steps_within(deadline: Deadline, seconds_per_step: float, requested: int) -> int:
    """Largest step count that fits, used before a loop whose cost is known."""
    if deadline.seconds is None or seconds_per_step <= 0:
        return requested
    return max(1, min(requested, int(deadline.remaining() / seconds_per_step)))
