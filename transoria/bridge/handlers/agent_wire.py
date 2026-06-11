"""Wire serialization and preview sanitization for Agent Lab."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from transoria.agent.schemas import (
    AgentActionDraft,
    AgentConversation,
    AgentWorkspaceState,
)
from transoria.bridge.errors import BridgeError

CompoundDraftNormalizer = Callable[[Mapping[str, object]], Sequence[AgentActionDraft]]


def workspace_wire(
    state: AgentWorkspaceState,
    *,
    normalize_compound_action_drafts: CompoundDraftNormalizer,
) -> dict[str, object]:
    """Wire shape: workspace config plus the active conversation hoisted flat."""
    active = state.active()
    return {
        "workflow_model_id": state.workflow_model_id,
        "workflow_thinking_level": state.workflow_thinking_level,
        "active_recipe_id": resolved_active_recipe_id(state),
        "stage_model_ids": dict(state.stage_model_ids),
        "stage_prompt_ids": dict(state.stage_prompt_ids),
        "memories": list(state.memories),
        "recipes": [recipe.to_dict() for recipe in state.recipes],
        "active_task": state.active_task.to_dict() if state.active_task else None,
        "active_conversation_id": state.active_conversation_id,
        "conversations": [
            conversation_summary(conversation) for conversation in state.conversations
        ],
        "messages": [message.to_dict() for message in active.messages]
        if active
        else [],
        "pending_draft": draft_wire(
            active.pending_draft,
            normalize_compound_action_drafts=normalize_compound_action_drafts,
        )
        if active and active.pending_draft
        else None,
        "draft_history": [
            draft_wire(
                draft,
                normalize_compound_action_drafts=normalize_compound_action_drafts,
            )
            for draft in active.draft_history
        ]
        if active
        else [],
        "updated_at": state.updated_at,
    }


def draft_wire(
    draft: AgentActionDraft,
    *,
    normalize_compound_action_drafts: CompoundDraftNormalizer,
) -> dict[str, object]:
    return {
        **draft.to_dict(),
        "payload": sanitize_draft_payload(
            draft,
            normalize_compound_action_drafts=normalize_compound_action_drafts,
        ),
    }


def sanitize_draft_payload(
    draft: AgentActionDraft,
    *,
    normalize_compound_action_drafts: CompoundDraftNormalizer,
) -> dict[str, object]:
    payload = normalized_draft_payload_for_wire(
        draft,
        normalize_compound_action_drafts=normalize_compound_action_drafts,
    )
    return sanitize_preview_value(payload)  # type: ignore[return-value]


def normalized_draft_payload_for_wire(
    draft: AgentActionDraft,
    *,
    normalize_compound_action_drafts: CompoundDraftNormalizer,
) -> Mapping[str, object]:
    if draft.kind != "compound_config_update":
        return draft.payload
    try:
        actions = normalize_compound_action_drafts(draft.payload)
    except BridgeError:
        return draft.payload
    return {
        "actions": [
            {
                "kind": action.kind,
                "title": action.title,
                "summary": action.summary,
                "payload": dict(action.payload),
            }
            for action in actions
        ]
    }


def sanitize_preview_value(value: object, *, key: str | None = None) -> object:
    if is_sensitive_key(key):
        if isinstance(value, list):
            return ["<masked>" for item in value if item not in (None, "")]
        if value in (None, ""):
            return value
        return "<masked>"
    if isinstance(value, Mapping):
        sanitized: dict[str, object] = {}
        for child_key, child_value in value.items():
            child_key_str = str(child_key)
            if child_key_str == "custom_headers" and isinstance(child_value, list):
                sanitized[child_key_str] = sanitize_custom_headers(child_value)
            else:
                sanitized[child_key_str] = sanitize_preview_value(
                    child_value,
                    key=child_key_str,
                )
        return sanitized
    if isinstance(value, list):
        return [sanitize_preview_value(item) for item in value]
    return value


def sanitize_custom_headers(value: list[object]) -> list[object]:
    sanitized: list[object] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            sanitized.append(item)
            continue
        name = str(item[0])
        header_value = "<masked>" if is_sensitive_key(name) else item[1]
        sanitized.append([name, header_value])
    return sanitized


def is_sensitive_key(key: str | None) -> bool:
    if key is None:
        return False
    normalized = key.lower().replace("-", "_")
    return normalized in {
        "api_key",
        "api_keys",
        "authorization",
        "x_api_key",
        "proxy_authorization",
    }


def conversation_summary(conversation: AgentConversation) -> dict[str, object]:
    return {
        "id": conversation.id,
        "title": conversation.title,
        "message_count": len(conversation.messages),
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
    }


def resolved_active_recipe_id(state: AgentWorkspaceState) -> str | None:
    if state.active_recipe_id and state.get_recipe(state.active_recipe_id):
        return state.active_recipe_id
    return active_recipe_id(state)


def active_recipe_id(state: AgentWorkspaceState) -> str | None:
    for recipe in state.recipes:
        if (
            dict(recipe.stage_model_ids) == dict(state.stage_model_ids)
            and dict(recipe.stage_prompt_ids) == dict(state.stage_prompt_ids)
        ):
            return recipe.id
    return None
