"""Intent priority helpers for the Agent Lab bridge handler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

INTENT_TASK_START = "task_start"
INTENT_COMPOUND_CONFIG = "compound_config"
INTENT_PROMPT_QUALITY = "prompt_quality"
INTENT_PROMPT_PRESET = "prompt_preset"
INTENT_MODEL_PROFILE_GUIDANCE = "model_profile_guidance"
INTENT_MODEL_PROFILE_COPY = "model_profile_copy"
INTENT_MODEL_UPGRADE = "model_upgrade"
INTENT_FALLBACK = "fallback"

INTENT_ROUTE_ORDER: tuple[str, ...] = (
    INTENT_TASK_START,
    INTENT_COMPOUND_CONFIG,
    INTENT_PROMPT_QUALITY,
    INTENT_PROMPT_PRESET,
    INTENT_MODEL_PROFILE_GUIDANCE,
    INTENT_MODEL_PROFILE_COPY,
    INTENT_MODEL_UPGRADE,
)


@dataclass(frozen=True)
class AgentIntent:
    kind: str


@dataclass(frozen=True)
class AgentIntentSignals:
    task_start: bool = False
    compound_config: bool = False
    prompt_quality: bool = False
    prompt_preset: bool = False
    model_profile_guidance: bool = False
    model_profile_copy: bool = False
    model_upgrade: bool = False

    def by_kind(self) -> Mapping[str, bool]:
        return {
            INTENT_TASK_START: self.task_start,
            INTENT_COMPOUND_CONFIG: self.compound_config,
            INTENT_PROMPT_QUALITY: self.prompt_quality,
            INTENT_PROMPT_PRESET: self.prompt_preset,
            INTENT_MODEL_PROFILE_GUIDANCE: self.model_profile_guidance,
            INTENT_MODEL_PROFILE_COPY: self.model_profile_copy,
            INTENT_MODEL_UPGRADE: self.model_upgrade,
        }


def classify_agent_intent(signals: AgentIntentSignals) -> AgentIntent:
    """Return the highest-priority Agent Lab intent for precomputed signals."""

    by_kind = signals.by_kind()
    for kind in INTENT_ROUTE_ORDER:
        if by_kind[kind]:
            return AgentIntent(kind)
    return AgentIntent(INTENT_FALLBACK)


def direct_intent_sequence(kind: str, *, allow_task_start: bool) -> tuple[str, ...]:
    """Return the ordered direct-response route candidates for an intent."""

    routes = INTENT_ROUTE_ORDER if kind == INTENT_FALLBACK else (kind,)
    if allow_task_start:
        return routes
    return tuple(route for route in routes if route != INTENT_TASK_START)
