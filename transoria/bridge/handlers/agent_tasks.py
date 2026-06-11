"""Task-start helpers for the Agent Lab bridge handlers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

from transoria.agent.completeness import TASK_KIND_BY_DRAFT_KIND, assess_start_draft
from transoria.agent.project_store import AgentProjectStore
from transoria.agent.schemas import (
    AgentActiveTask,
    AgentActionDraft,
    AgentWorkspaceState,
)
from transoria.bridge.errors import BridgeError
from transoria.bridge.task_service import TaskService
from transoria.domain import TaskStatus
from transoria.model_profiles import ModelProfileStore
from transoria.runtime.cache import TaskNotFoundError
from transoria.settings import SettingsStore

StartAgentTask = Callable[..., Mapping[str, object]]


def validate_start_task_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    task_service: TaskService,
) -> None:
    raise_if_active_task_locked(state)
    completeness = assess_start_draft(
        draft_kind=draft.kind,
        state=state,
        payload=draft.payload,
    )
    if not completeness.complete:
        raise BridgeError.invalid_argument(
            "task-start draft is incomplete.",
            details=completeness.to_dict(),
        )
    validate_glossary_task_reference(draft, task_service=task_service)


def apply_start_task_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    cache_root: Path,
    settings_store: SettingsStore,
    task_service: TaskService,
    start_task: StartAgentTask,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    raise_if_active_task_locked(state)
    conversation = state.active()
    if conversation is None:  # pragma: no cover - load() always reseeds one
        raise BridgeError.not_found("no active conversation exists.")
    result = start_task(
        draft_kind=draft.kind,
        payload=draft.payload,
        state=state,
        task_service=task_service,
        settings_store=settings_store,
        profile_store=profile_store,
        cache_root=cache_root,
        request_id=draft.id,
    )
    task_id = str(result.get("task_id") or "")
    task_kind = TASK_KIND_BY_DRAFT_KIND[draft.kind]
    started_at = str(result.get("started_at") or "")
    active_task = AgentActiveTask.create(
        task_id=task_id,
        kind=task_kind,  # type: ignore[arg-type]
        conversation_id=conversation.id,
        started_at=started_at,
    )
    return state.with_active_task(active_task), {
        "kind": draft.kind,
        "task": active_task.to_dict(),
        "start_result": dict(result),
    }


def load_with_reconciled_active_task(
    project_store: AgentProjectStore,
    task_service: TaskService,
    *,
    reconcile_stale: bool = True,
) -> AgentWorkspaceState:
    state = project_store.load()
    if state.active_task is None:
        return state
    if active_task_is_terminal(
        state.active_task,
        task_service,
        reconcile_stale=reconcile_stale,
    ):
        return project_store.save(state.clear_active_task())
    return state


def active_task_is_terminal(
    active_task: AgentActiveTask,
    task_service: TaskService,
    *,
    reconcile_stale: bool,
) -> bool:
    terminal_statuses = {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.STOPPED,
    }
    try:
        record = task_service.cache.load_record(active_task.task_id)
    except (TaskNotFoundError, ValueError, OSError):
        record = None
    else:
        if record.status in terminal_statuses:
            return True
        if not reconcile_stale:
            return False

    try:
        response = task_service.read_snapshot(
            kind=active_task.kind,
            task_id=active_task.task_id,
        )
    except BridgeError:
        pass
    else:
        snapshot = response.get("snapshot")
        if isinstance(snapshot, Mapping):
            raw_status = snapshot.get("status")
            if isinstance(raw_status, str):
                try:
                    status = TaskStatus(raw_status)
                except ValueError:
                    return False
                return status in terminal_statuses
    try:
        refreshed_record = task_service.cache.load_record(active_task.task_id)
    except (TaskNotFoundError, ValueError, OSError):
        return record is None
    return refreshed_record.status in terminal_statuses


def raise_if_active_task_locked(state: AgentWorkspaceState) -> None:
    if state.active_task is None:
        return
    active = state.active_task
    raise BridgeError(
        "bridge.conflict",
        (
            f"检测到当前正在执行 {active.kind} 任务（任务 ID: {active.task_id}），"
            "暂时无法开始另一个任务。请等待任务结束后再继续。"
        ),
        retryable=True,
        details=active.to_dict(),
    )


def validate_glossary_task_reference(
    draft: AgentActionDraft,
    *,
    task_service: TaskService,
) -> None:
    if draft.kind not in {"start_glossary_review_task", "start_translation_task"}:
        return
    if draft.kind == "start_translation_task":
        review_task_id = draft.payload.get("glossary_review_task_id")
        if isinstance(review_task_id, str) and review_task_id.strip():
            task_service.read_glossary_review_final(task_id=review_task_id.strip())
            return
    task_id = draft.payload.get("glossary_task_id")
    if task_id is None and draft.kind == "start_translation_task":
        return
    if not isinstance(task_id, str) or not task_id.strip():
        raise BridgeError.invalid_argument(
            "glossary_task_id is required.",
            field="glossary_task_id",
        )
    task_service.read_artifacts(kind="glossary", task_id=task_id.strip())


__all__ = [
    "active_task_is_terminal",
    "apply_start_task_action",
    "load_with_reconciled_active_task",
    "raise_if_active_task_locked",
    "validate_glossary_task_reference",
    "validate_start_task_action",
]
