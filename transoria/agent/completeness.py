"""Completeness checks for Agent Lab task-start drafts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from transoria.agent.schemas import AgentWorkspaceState


START_DRAFT_KINDS: frozenset[str] = frozenset(
    {
        "start_glossary_task",
        "start_glossary_review_task",
        "start_translation_task",
    }
)

_STAGE_BY_DRAFT_KIND: dict[str, str] = {
    "start_glossary_task": "term_extract",
    "start_glossary_review_task": "term_review",
    "start_translation_task": "translation",
}

_REQUIRED_PAYLOAD_FIELDS: dict[str, tuple[str, ...]] = {
    "start_glossary_task": (
        "input_dir",
        "output_dir",
        "source_language",
        "target_language",
        "novel_background",
    ),
    "start_glossary_review_task": ("glossary_task_id",),
    "start_translation_task": (
        "input_dir",
        "output_dir",
        "source_language",
        "target_language",
    ),
}

TASK_KIND_BY_DRAFT_KIND: dict[str, str] = {
    "start_glossary_task": "glossary",
    "start_glossary_review_task": "glossary_review",
    "start_translation_task": "translation",
}


@dataclass(frozen=True)
class StartDraftCompleteness:
    draft_kind: str
    stage: str
    missing: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing

    def to_dict(self) -> dict[str, object]:
        return {
            "draft_kind": self.draft_kind,
            "stage": self.stage,
            "complete": self.complete,
            "missing": list(self.missing),
        }


def assess_start_draft(
    *,
    draft_kind: str,
    state: AgentWorkspaceState,
    payload: Mapping[str, object],
) -> StartDraftCompleteness:
    """Return missing recipe slots and explicit task payload fields."""

    stage = _STAGE_BY_DRAFT_KIND[draft_kind]
    missing: list[str] = []
    if not state.stage_model_ids.get(stage):
        missing.append(f"stage_model_ids.{stage}")
    if not state.stage_prompt_ids.get(stage):
        missing.append(f"stage_prompt_ids.{stage}")
    for field in _REQUIRED_PAYLOAD_FIELDS[draft_kind]:
        if not _has_effective_value(payload.get(field)):
            missing.append(field)
    return StartDraftCompleteness(
        draft_kind=draft_kind,
        stage=stage,
        missing=tuple(missing),
    )


def all_start_draft_completeness(
    state: AgentWorkspaceState,
) -> list[dict[str, object]]:
    return [
        assess_start_draft(draft_kind=kind, state=state, payload={}).to_dict()
        for kind in sorted(START_DRAFT_KINDS)
    ]


def _has_effective_value(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


__all__ = [
    "START_DRAFT_KINDS",
    "TASK_KIND_BY_DRAFT_KIND",
    "StartDraftCompleteness",
    "all_start_draft_completeness",
    "assess_start_draft",
]
