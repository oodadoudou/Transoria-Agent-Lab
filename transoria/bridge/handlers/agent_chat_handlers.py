"""Chat message and draft bridge handlers for Agent Lab."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

from transoria.agent.project_store import AgentProjectStore
from transoria.agent.schemas import (
    AgentActionDraft,
    AgentConversation,
    AgentMessage,
    AgentWorkspaceState,
)
from transoria.bridge.errors import BridgeError
from transoria.bridge.handlers._utils import expect_string
from transoria.bridge.handlers.agent_action_registry import (
    apply_action_draft,
    coerce_compound_action_drafts,
)
from transoria.bridge.handlers.agent_chat import draft_revision_prompt, generate_reply
from transoria.bridge.handlers.agent_response_routing import (
    should_discard_pending_for_new_message,
)
from transoria.bridge.handlers.agent_task_status import applied_draft_message
from transoria.bridge.handlers.agent_tasks import load_with_reconciled_active_task
from transoria.bridge.handlers.agent_wire import sanitize_draft_payload
from transoria.bridge.handlers.agent_workspace import workspace_response
from transoria.bridge.task_service import TaskService
from transoria.llm.client import LlmClient
from transoria.model_profiles import ModelProfileStore
from transoria.settings import SettingsStore

LlmClientFactory = Callable[[], LlmClient]
StartAgentTask = Callable[..., dict[str, object]]


def build_chat_handlers(
    *,
    cache_root: Path,
    project_store: AgentProjectStore,
    profile_store: ModelProfileStore,
    settings_store: SettingsStore,
    task_service: TaskService,
    llm_client_factory: LlmClientFactory,
    start_task: StartAgentTask,
) -> dict[str, object]:
    def respond(state: AgentWorkspaceState) -> dict[str, object]:
        return workspace_response(
            state,
            profile_store,
            cache_root,
            normalize_compound_action_drafts=coerce_compound_action_drafts,
        )

    def send_message(payload: Mapping[str, object]) -> dict[str, object]:
        content = expect_string(payload, "message").strip()
        if not content:
            raise BridgeError.invalid_argument(
                "message must not be empty.",
                field="message",
            )
        if len(content) > 8000:
            raise BridgeError.invalid_argument(
                "message is too long.",
                field="message",
                details={"max_length": 8000},
            )

        current = load_with_reconciled_active_task(
            project_store,
            task_service,
            reconcile_stale=False,
        )
        conversation = _require_active(current)
        previous_pending = conversation.pending_draft
        conversation = conversation.append_message(AgentMessage.create("user", content))
        if not conversation.title:
            conversation = conversation.with_title(_derive_title(content))
        state_with_user = current.with_active(conversation)

        reply, draft = generate_reply(
            state_with_user,
            user_message=content,
            profile_store=profile_store,
            cache_root=cache_root,
            settings_store=settings_store,
            task_service=task_service,
            llm_client_factory=llm_client_factory,
        )
        conversation = conversation.append_message(AgentMessage.create("assistant", reply))
        if previous_pending is not None and (
            draft is not None or should_discard_pending_for_new_message(content)
        ):
            conversation = conversation.archive_pending(
                "discarded",
                payload=_sanitize_draft_payload(previous_pending),
            )
        if draft is not None:
            conversation = conversation.with_pending_draft(draft)
        final_state = state_with_user.with_active(conversation)
        project_store.save(final_state)
        return respond(final_state)

    def apply_draft(payload: Mapping[str, object]) -> dict[str, object]:
        draft_id = expect_string(payload, "draft_id")
        current = load_with_reconciled_active_task(project_store, task_service)
        conversation = _require_active(current)
        draft = conversation.pending_draft
        if draft is None or draft.id != draft_id or draft.status != "pending":
            raise BridgeError.not_found(
                f"pending draft {draft_id!r} does not exist.",
                details={"draft_id": draft_id},
            )
        applied_state, result = _apply_draft(
            current,
            draft,
            profile_store=profile_store,
            cache_root=cache_root,
            settings_store=settings_store,
            task_service=task_service,
            start_task=start_task,
        )
        conversation = _require_active(applied_state)
        conversation = conversation.archive_pending(
            "applied",
            payload=_sanitize_draft_payload(draft),
        )
        conversation = conversation.append_message(
            AgentMessage.create(
                "assistant",
                applied_draft_message(draft, result, task_service=task_service),
            )
        )
        final_state = applied_state.with_active(conversation)
        project_store.save(final_state)
        return {**respond(final_state), "result": result}

    def discard_draft(payload: Mapping[str, object]) -> dict[str, object]:
        draft_id = expect_string(payload, "draft_id")
        current = project_store.load()
        conversation = _require_active(current)
        draft = conversation.pending_draft
        if draft is None or draft.id != draft_id or draft.status != "pending":
            raise BridgeError.not_found(
                f"pending draft {draft_id!r} does not exist.",
                details={"draft_id": draft_id},
            )
        conversation = conversation.archive_pending(
            "discarded",
            payload=_sanitize_draft_payload(draft),
        )
        final_state = current.with_active(conversation)
        project_store.save(final_state)
        return respond(final_state)

    def revise_draft(payload: Mapping[str, object]) -> dict[str, object]:
        draft_id = expect_string(payload, "draft_id")
        adjustment = expect_string(payload, "adjustment").strip()
        if not adjustment:
            raise BridgeError.invalid_argument(
                "adjustment must not be empty.",
                field="adjustment",
            )
        if len(adjustment) > 4000:
            raise BridgeError.invalid_argument(
                "adjustment is too long.",
                field="adjustment",
                details={"max_length": 4000},
            )

        current = load_with_reconciled_active_task(project_store, task_service)
        conversation = _require_active(current)
        draft = conversation.pending_draft
        if draft is None or draft.id != draft_id or draft.status != "pending":
            raise BridgeError.not_found(
                f"pending draft {draft_id!r} does not exist.",
                details={"draft_id": draft_id},
            )

        visible_request = f"调整草案：{adjustment}"
        conversation = conversation.append_message(
            AgentMessage.create("user", visible_request)
        )
        state_with_user = current.with_active(conversation)

        reply, revised_draft = generate_reply(
            state_with_user,
            user_message=draft_revision_prompt(adjustment, draft),
            profile_store=profile_store,
            cache_root=cache_root,
            settings_store=settings_store,
            task_service=task_service,
            llm_client_factory=llm_client_factory,
        )
        conversation = _require_active(state_with_user).append_message(
            AgentMessage.create("assistant", reply)
        )
        if revised_draft is not None:
            conversation = conversation.archive_pending(
                "discarded",
                payload=_sanitize_draft_payload(draft),
            )
            conversation = conversation.with_pending_draft(revised_draft)
        final_state = state_with_user.with_active(conversation)
        project_store.save(final_state)
        return respond(final_state)

    return {
        "agent.send_message": send_message,
        "agent.apply_draft": apply_draft,
        "agent.discard_draft": discard_draft,
        "agent.revise_draft": revise_draft,
    }


def _apply_draft(
    state: AgentWorkspaceState,
    draft: AgentActionDraft,
    *,
    profile_store: ModelProfileStore,
    cache_root: Path,
    settings_store: SettingsStore,
    task_service: TaskService,
    start_task: StartAgentTask,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    return apply_action_draft(
        state=state,
        draft=draft,
        profile_store=profile_store,
        cache_root=cache_root,
        settings_store=settings_store,
        task_service=task_service,
        start_task=start_task,
    )


def _require_active(state: AgentWorkspaceState) -> AgentConversation:
    conversation = state.active()
    if conversation is None:  # pragma: no cover - load() always reseeds one
        raise BridgeError.not_found("no active conversation exists.")
    return conversation


def _derive_title(content: str) -> str:
    flat = " ".join(content.split())
    return flat[:40]


def _sanitize_draft_payload(draft: AgentActionDraft) -> dict[str, object]:
    return sanitize_draft_payload(
        draft,
        normalize_compound_action_drafts=coerce_compound_action_drafts,
    )


__all__ = ["build_chat_handlers"]
