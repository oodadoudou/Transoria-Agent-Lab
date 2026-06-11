"""Workspace config and context helpers for Agent Lab bridge handlers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from transoria.agent.completeness import all_start_draft_completeness
from transoria.agent.schemas import AgentActionDraft, AgentRecipe, AgentWorkspaceState
from transoria.bridge.errors import BridgeError
from transoria.bridge.handlers.agent_inventory import (
    inventory,
    settings_defaults,
)
from transoria.bridge.handlers.agent_recipe_actions import (
    coerce_model_id,
    coerce_model_slots,
    coerce_prompt_slots,
)
from transoria.bridge.handlers.agent_task_status import (
    AGENT_TASK_KINDS,
    recent_task_summaries,
)
from transoria.bridge.handlers.agent_wire import workspace_wire
from transoria.bridge.task_service import TaskService
from transoria.llm.config import ModelConfig, ThinkingLevel
from transoria.model_profiles import ModelProfileStore
from transoria.settings import SettingsStore


CompoundDraftNormalizer = Callable[[Mapping[str, object]], Sequence[AgentActionDraft]]


def apply_workspace_patch(
    state: AgentWorkspaceState,
    patch: Mapping[str, object],
    *,
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> AgentWorkspaceState:
    workflow_model_id = state.workflow_model_id
    if "workflow_model_id" in patch:
        workflow_model_id = coerce_model_id(
            patch.get("workflow_model_id"),
            profile_store=profile_store,
            field="workflow_model_id",
        )
    workflow_thinking_level = state.workflow_thinking_level
    if "workflow_model_id" in patch and "workflow_thinking_level" not in patch:
        profile = profile_store.get(workflow_model_id) if workflow_model_id else None
        workflow_thinking_level = (
            profile.thinking_level.value if profile is not None else "off"
        )
    if "workflow_thinking_level" in patch:
        workflow_thinking_level = coerce_workflow_thinking_level(
            patch.get("workflow_thinking_level"),
            workflow_model_id=workflow_model_id,
            profile_store=profile_store,
        )
    stage_model_ids = dict(state.stage_model_ids)
    if "stage_model_ids" in patch:
        stage_model_ids.update(
            coerce_model_slots(
                patch.get("stage_model_ids"),
                profile_store=profile_store,
            )
        )
    stage_prompt_ids = dict(state.stage_prompt_ids)
    if "stage_prompt_ids" in patch:
        stage_prompt_ids.update(
            coerce_prompt_slots(patch.get("stage_prompt_ids"), cache_root=cache_root)
        )
    active_recipe_id = state.active_recipe_id
    stage_selection_touched = "stage_model_ids" in patch or "stage_prompt_ids" in patch
    if "active_recipe_id" in patch:
        active_recipe_id = coerce_active_recipe_id(
            patch.get("active_recipe_id"),
            state=state,
        )
    elif stage_selection_touched:
        active_recipe_id = None
    return state.with_config(
        workflow_model_id=workflow_model_id,
        workflow_thinking_level=workflow_thinking_level,
        stage_model_ids=stage_model_ids,
        stage_prompt_ids=stage_prompt_ids,
        active_recipe_id=active_recipe_id,
    )


def coerce_workflow_thinking_level(
    value: object,
    *,
    workflow_model_id: str | None,
    profile_store: ModelProfileStore,
) -> str:
    level = str(value or "off")
    if level not in {item.value for item in ThinkingLevel}:
        raise BridgeError.invalid_argument(
            "workflow_thinking_level must be off, low, medium, or high.",
            field="workflow_thinking_level",
        )
    if level == ThinkingLevel.OFF.value:
        return level
    if workflow_model_id is None:
        raise BridgeError.invalid_argument(
            "Select a workflow model before enabling thinking.",
            field="workflow_thinking_level",
        )
    profile = profile_store.get(workflow_model_id)
    if profile is None or profile.thinking_level is ThinkingLevel.OFF:
        raise BridgeError.invalid_argument(
            "The selected workflow model does not expose thinking mode.",
            field="workflow_thinking_level",
        )
    return level


def profile_for_workflow_chat(
    profile: ModelConfig,
    workflow_thinking_level: str,
) -> ModelConfig:
    level = (
        ThinkingLevel(workflow_thinking_level)
        if workflow_thinking_level in {item.value for item in ThinkingLevel}
        else ThinkingLevel.OFF
    )
    return replace(profile, thinking_level=level)


def coerce_active_recipe_id(
    raw: object,
    *,
    state: AgentWorkspaceState,
) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise BridgeError.invalid_argument(
            "active_recipe_id must be a string or null.",
            field="active_recipe_id",
        )
    value = raw.strip()
    if not value:
        return None
    if state.get_recipe(value) is None:
        raise BridgeError.not_found(
            f"recipe {value!r} does not exist.",
            details={"recipe_id": value},
        )
    return value


def optional_agent_string(
    payload: Mapping[str, object],
    key: str,
) -> str | None:
    raw = payload.get(key)
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise BridgeError.invalid_argument(
            f"{key} must be a string.",
            field=key,
        )
    value = raw.strip()
    return value or None


def optional_agent_limit(
    payload: Mapping[str, object],
    *,
    default: int,
) -> int:
    raw = payload.get("limit")
    if raw is None:
        return default
    try:
        value = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise BridgeError.invalid_argument(
            "limit must be an integer.",
            field="limit",
        ) from exc
    if value < 0:
        raise BridgeError.invalid_argument(
            "limit must be >= 0.",
            field="limit",
        )
    return value


def workspace_response(
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    cache_root: Path,
    *,
    normalize_compound_action_drafts: CompoundDraftNormalizer,
) -> dict[str, object]:
    return {
        "workspace": workspace_wire(
            state,
            normalize_compound_action_drafts=normalize_compound_action_drafts,
        ),
        "inventory": inventory(profile_store, cache_root),
    }


def llm_context(
    state: AgentWorkspaceState,
    *,
    settings_store: SettingsStore,
    task_service: TaskService,
    max_context_messages: int,
) -> dict[str, object]:
    active = state.active()
    recent = active.messages[-max_context_messages:] if active else ()
    return {
        "workflow_model_id": state.workflow_model_id,
        "workflow_thinking_level": state.workflow_thinking_level,
        "stage_model_ids": dict(state.stage_model_ids),
        "stage_prompt_ids": dict(state.stage_prompt_ids),
        "active_task": state.active_task.to_dict() if state.active_task else None,
        "start_task_completeness": all_start_draft_completeness(state),
        "recent_task_summaries": {
            task_kind: recent_task_summaries(
                task_service,
                kind=task_kind,
                limit=3,
            )
            for task_kind in AGENT_TASK_KINDS
        },
        "settings_defaults": settings_defaults(settings_store),
        "memories": list(state.memories),
        "recipes": [recipe_context(recipe) for recipe in state.recipes],
        "recent_messages": [
            {"role": message.role, "content": message.content} for message in recent
        ],
    }


def recipe_context(recipe: AgentRecipe) -> dict[str, object]:
    return {
        "id": recipe.id,
        "name": recipe.name,
        "description": recipe.description,
        "stage_model_ids": dict(recipe.stage_model_ids),
        "stage_prompt_ids": dict(recipe.stage_prompt_ids),
    }


__all__ = [
    "apply_workspace_patch",
    "coerce_active_recipe_id",
    "coerce_workflow_thinking_level",
    "llm_context",
    "optional_agent_limit",
    "optional_agent_string",
    "profile_for_workflow_chat",
    "workspace_response",
]
