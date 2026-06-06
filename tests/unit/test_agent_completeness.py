from __future__ import annotations

from transoria.agent.completeness import (
    all_start_draft_completeness,
    assess_start_draft,
)
from transoria.agent.schemas import AgentWorkspaceState


def test_translation_start_requires_stage_recipe_and_explicit_payload() -> None:
    state = AgentWorkspaceState.empty()

    result = assess_start_draft(
        draft_kind="start_translation_task",
        state=state,
        payload={},
    )

    assert result.complete is False
    assert result.missing == (
        "stage_model_ids.translation",
        "stage_prompt_ids.translation",
        "input_dir",
        "output_dir",
        "source_language",
        "target_language",
    )


def test_translation_start_complete_with_stage_recipe_and_payload() -> None:
    state = AgentWorkspaceState.empty().with_config(
        stage_model_ids={"translation": "profile-1"},
        stage_prompt_ids={"translation": "prompt-1"},
    )

    result = assess_start_draft(
        draft_kind="start_translation_task",
        state=state,
        payload={
            "input_dir": "/in",
            "output_dir": "/out",
            "source_language": "kr",
            "target_language": "zh",
        },
    )

    assert result.complete is True
    assert result.missing == ()


def test_glossary_review_start_uses_glossary_task_id_only_as_payload() -> None:
    state = AgentWorkspaceState.empty().with_config(
        stage_model_ids={"term_review": "profile-1"},
        stage_prompt_ids={"term_review": "prompt-1"},
    )

    result = assess_start_draft(
        draft_kind="start_glossary_review_task",
        state=state,
        payload={"glossary_task_id": "glossary-1"},
    )

    assert result.complete is True


def test_all_start_draft_completeness_reports_every_kind() -> None:
    state = AgentWorkspaceState.empty()

    kinds = {
        item["draft_kind"]
        for item in all_start_draft_completeness(state)
        if isinstance(item["draft_kind"], str)
    }

    assert kinds == {
        "start_glossary_task",
        "start_glossary_review_task",
        "start_translation_task",
    }
