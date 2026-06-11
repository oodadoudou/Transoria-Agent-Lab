from __future__ import annotations

from pathlib import Path

import pytest

from transoria.agent.schemas import (
    AgentActiveTask,
    AgentActionDraft,
    AgentWorkspaceState,
)
from transoria.bridge.errors import BridgeError
from transoria.bridge.handlers.agent_tasks import (
    apply_start_task_action,
    raise_if_active_task_locked,
    validate_start_task_action,
)


def test_active_task_lock_reports_running_task() -> None:
    state = AgentWorkspaceState.empty().with_active_task(
        AgentActiveTask.create(
            task_id="glossary-running",
            kind="glossary",
            conversation_id="conv-1",
            started_at="2026-06-10T00:00:00Z",
        )
    )

    with pytest.raises(BridgeError) as exc_info:
        raise_if_active_task_locked(state)

    assert exc_info.value.code == "bridge.conflict"
    assert exc_info.value.retryable is True
    assert exc_info.value.payload.details["task_id"] == "glossary-running"


def test_start_task_validation_reports_missing_recipe_and_payload() -> None:
    draft = AgentActionDraft.create(
        kind="start_glossary_task",
        title="Start glossary",
        summary="Start glossary.",
        payload={},
    )

    with pytest.raises(BridgeError) as exc_info:
        validate_start_task_action(
            draft,
            state=AgentWorkspaceState.empty(),
            task_service=object(),  # type: ignore[arg-type]
        )

    assert exc_info.value.code == "bridge.invalid_argument"
    missing = exc_info.value.payload.details["missing"]
    assert "stage_model_ids.term_extract" in missing
    assert "input_dir" in missing


def test_apply_start_task_action_records_active_task(tmp_path: Path) -> None:
    state = AgentWorkspaceState.empty().with_config(
        stage_model_ids={"term_extract": "profile-1"},
        stage_prompt_ids={"term_extract": "default-glossary"},
    )
    draft = AgentActionDraft.create(
        kind="start_glossary_task",
        title="Start glossary",
        summary="Start glossary.",
        payload={
            "input_dir": "/tmp/input",
            "output_dir": "/tmp/output",
            "source_language": "kr",
            "target_language": "zh",
            "novel_background": "现代 BL",
        },
    )
    captured: dict[str, object] = {}

    def fake_start_task(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "task_id": "glossary-created",
            "started_at": "2026-06-10T00:00:00Z",
        }

    updated, result = apply_start_task_action(
        draft,
        state=state,
        profile_store=object(),  # type: ignore[arg-type]
        cache_root=tmp_path,
        settings_store=object(),  # type: ignore[arg-type]
        task_service=object(),  # type: ignore[arg-type]
        start_task=fake_start_task,
    )

    assert captured["draft_kind"] == "start_glossary_task"
    assert captured["request_id"] == draft.id
    assert updated.active_task is not None
    assert updated.active_task.task_id == "glossary-created"
    assert updated.active_task.kind == "glossary"
    assert result["task"] == updated.active_task.to_dict()
