"""Legacy private helper exports for the Agent Lab bridge handler.

These wrappers keep older tests and ad-hoc debugging imports working while the
real implementation moves into focused modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from transoria.agent.project_store import AgentProjectStore
from transoria.agent.schemas import (
    AgentActiveTask,
    AgentActionDraft,
    AgentConversation,
    AgentRecipe,
    AgentWorkspaceState,
)
from transoria.bridge.errors import BridgeError
from transoria.bridge.handlers.agent_action_registry import (
    AgentActionSpec,
    action_spec as get_agent_action_spec_from_registry,
    action_specs as get_agent_action_specs_from_registry,
    append_model_quality_warnings as append_agent_model_quality_warnings,
    apply_action_draft as apply_agent_action_draft,
    coerce_compound_action_drafts as coerce_agent_compound_action_drafts,
    validate_action_draft as validate_agent_action_draft,
    validate_compound_action_kind as validate_agent_compound_action_kind,
)
from transoria.bridge.handlers.agent_intents import AgentIntent
from transoria.bridge.handlers.agent_inventory import (
    generate_prompt_id as generate_agent_prompt_id,
    inventory as read_agent_inventory,
    looks_like_placeholder_value,
    model_profile_has_placeholder_fields,
    prompt_body as agent_prompt_body,
    prompt_store_for as agent_prompt_store_for,
    prompt_summary as agent_prompt_summary,
    settings_defaults as read_agent_settings_defaults,
)
from transoria.bridge.handlers.agent_model_profile_actions import (
    coerce_api_keys as coerce_agent_api_keys,
    coerce_model_profile_patch as coerce_agent_model_profile_patch,
    coerce_model_profile_update_payload as coerce_agent_model_profile_update_payload,
    generate_model_profile_id as generate_agent_model_profile_id,
    model_profile_body as agent_model_profile_body,
    model_profile_from_draft_payload as agent_model_profile_from_draft_payload,
)
from transoria.bridge.handlers.agent_memory_actions import (
    coerce_memories_payload as coerce_agent_memories_payload,
    coerce_memory_items_payload as coerce_agent_memory_items_payload,
)
from transoria.bridge.handlers.agent_prompt_actions import (
    coerce_prompt_kind as coerce_agent_prompt_kind,
    coerce_prompt_patch as coerce_agent_prompt_patch,
    coerce_prompt_preset_payload as coerce_agent_prompt_preset_payload,
    coerce_prompt_update_payload as coerce_agent_prompt_update_payload,
    create_prompt_preset as create_agent_prompt_preset,
    prompt_body_from_payload as agent_prompt_body_from_payload,
    resolve_prompt_for_update as resolve_agent_prompt_for_update,
    update_prompt_preset as update_agent_prompt_preset,
)
from transoria.bridge.handlers.agent_recipe_actions import (
    coerce_model_id as coerce_agent_model_id,
    coerce_model_slots as coerce_agent_model_slots,
    coerce_prompt_slots as coerce_agent_prompt_slots,
    coerce_recipe_payload as coerce_agent_recipe_payload,
    coerce_recipe_update_payload as coerce_agent_recipe_update_payload,
    optional_str as optional_agent_str_value,
    recipe_body_from_payload as agent_recipe_body_from_payload,
    require_recipe_from_payload as require_agent_recipe_from_payload,
)
from transoria.bridge.handlers.agent_response_routing import (
    classify_message_intent,
    should_discard_pending_for_new_message as agent_should_discard_pending_for_new_message,
)
from transoria.bridge.handlers.agent_tasks import (
    active_task_is_terminal,
    load_with_reconciled_active_task,
    raise_if_active_task_locked,
    validate_glossary_task_reference,
)
from transoria.bridge.handlers.agent_workspace import (
    apply_workspace_patch as apply_agent_workspace_patch,
    coerce_active_recipe_id as coerce_agent_active_recipe_id,
    coerce_workflow_thinking_level as coerce_agent_workflow_thinking_level,
    llm_context as build_agent_llm_context,
    optional_agent_limit as coerce_optional_agent_limit,
    optional_agent_string as coerce_optional_agent_string,
    profile_for_workflow_chat as profile_for_agent_workflow_chat,
    workspace_response as build_agent_workspace_response,
)
from transoria.bridge.handlers.agent_wire import (
    sanitize_draft_payload as sanitize_draft_payload_for_wire,
    workspace_wire as workspace_wire_for_response,
)
from transoria.bridge.task_service import TaskService
from transoria.llm.config import ModelConfig
from transoria.model_profiles import ModelProfileStore
from transoria.prompts import PromptKind, PromptPreset, PromptPresetStore
from transoria.settings import SettingsStore
from transoria.workflows.agent.task_starts import start_agent_task

_MAX_CONTEXT_MESSAGES = 20


def _classify_agent_intent(text: str) -> AgentIntent:
    return classify_message_intent(text)


def _should_discard_pending_for_new_message(text: str) -> bool:
    return agent_should_discard_pending_for_new_message(text)


def _apply_workspace_patch(
    state: AgentWorkspaceState,
    patch: Mapping[str, object],
    *,
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> AgentWorkspaceState:
    return apply_agent_workspace_patch(
        state,
        patch,
        profile_store=profile_store,
        cache_root=cache_root,
    )


def _apply_draft(
    state: AgentWorkspaceState,
    draft: AgentActionDraft,
    *,
    profile_store: ModelProfileStore,
    cache_root: Path,
    settings_store: SettingsStore,
    task_service: TaskService,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    return apply_agent_action_draft(
        state=state,
        draft=draft,
        profile_store=profile_store,
        cache_root=cache_root,
        settings_store=settings_store,
        task_service=task_service,
        start_task=start_agent_task,
    )


def _validate_draft(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    cache_root: Path,
    task_service: TaskService,
) -> None:
    validate_agent_action_draft(
        draft,
        state=state,
        profile_store=profile_store,
        cache_root=cache_root,
        task_service=task_service,
    )


def _append_model_quality_warnings(
    reply: str,
    *,
    draft: AgentActionDraft,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
) -> str:
    return append_agent_model_quality_warnings(
        reply,
        draft=draft,
        state=state,
        profile_store=profile_store,
    )


def _agent_action_spec(kind: str) -> AgentActionSpec:
    return get_agent_action_spec_from_registry(kind)


def _agent_action_specs() -> dict[str, AgentActionSpec]:
    return get_agent_action_specs_from_registry()


def _coerce_compound_action_drafts(
    payload: Mapping[str, object],
) -> list[AgentActionDraft]:
    return coerce_agent_compound_action_drafts(payload)


def _validate_compound_action_kind(kind: str) -> None:
    validate_agent_compound_action_kind(kind)


def _load_with_reconciled_active_task(
    project_store: AgentProjectStore,
    task_service: TaskService,
    *,
    reconcile_stale: bool = True,
) -> AgentWorkspaceState:
    return load_with_reconciled_active_task(
        project_store,
        task_service,
        reconcile_stale=reconcile_stale,
    )


def _active_task_is_terminal(
    active_task: AgentActiveTask,
    task_service: TaskService,
    *,
    reconcile_stale: bool,
) -> bool:
    return active_task_is_terminal(
        active_task,
        task_service,
        reconcile_stale=reconcile_stale,
    )


def _raise_if_active_task_locked(state: AgentWorkspaceState) -> None:
    raise_if_active_task_locked(state)


def _validate_glossary_task_reference(
    draft: AgentActionDraft,
    *,
    task_service: TaskService,
) -> None:
    validate_glossary_task_reference(draft, task_service=task_service)


def _create_prompt_preset(
    payload: Mapping[str, object],
    *,
    cache_root: Path,
) -> PromptPreset:
    return create_agent_prompt_preset(payload, cache_root=cache_root)


def _update_prompt_preset(
    payload: Mapping[str, object],
    *,
    cache_root: Path,
) -> PromptPreset:
    return update_agent_prompt_preset(payload, cache_root=cache_root)


def _coerce_prompt_update_payload(
    payload: Mapping[str, object],
) -> tuple[str, dict[str, object]]:
    return coerce_agent_prompt_update_payload(payload)


def _coerce_prompt_patch(patch: Mapping[str, object]) -> dict[str, object]:
    return coerce_agent_prompt_patch(patch)


def _resolve_prompt_for_update(
    preset_id: str,
    patch: Mapping[str, object],
    *,
    cache_root: Path,
) -> tuple[PromptKind, list[PromptPreset], int]:
    return resolve_agent_prompt_for_update(
        preset_id,
        patch,
        cache_root=cache_root,
    )


def _coerce_prompt_preset_payload(
    payload: Mapping[str, object],
) -> tuple[PromptKind, str, str, str, bool]:
    return coerce_agent_prompt_preset_payload(payload)


def _coerce_prompt_kind(value: object, *, fallback_text: str = "") -> PromptKind:
    return coerce_agent_prompt_kind(value, fallback_text=fallback_text)


def _recipe_body_from_payload(payload: Mapping[str, object]) -> dict[str, object]:
    return agent_recipe_body_from_payload(payload)


def _prompt_body_from_payload(payload: Mapping[str, object]) -> str:
    return agent_prompt_body_from_payload(payload)


def _coerce_recipe_update_payload(
    payload: Mapping[str, object],
    *,
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> tuple[
    str,
    dict[str, str],
    dict[str, str | None] | None,
    dict[str, str | None] | None,
]:
    return coerce_agent_recipe_update_payload(
        payload,
        profile_store=profile_store,
        cache_root=cache_root,
    )


def _require_recipe_from_payload(
    payload: Mapping[str, object],
    *,
    state: AgentWorkspaceState,
) -> AgentRecipe:
    return require_agent_recipe_from_payload(payload, state=state)


def _coerce_recipe_payload(
    payload: Mapping[str, object],
    *,
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> tuple[str, str, dict[str, str | None], dict[str, str | None]]:
    return coerce_agent_recipe_payload(
        payload,
        profile_store=profile_store,
        cache_root=cache_root,
    )


def _coerce_memory_items_payload(payload: Mapping[str, object]) -> tuple[str, ...]:
    return coerce_agent_memory_items_payload(payload)


def _coerce_memories_payload(payload: Mapping[str, object]) -> tuple[str, ...]:
    return coerce_agent_memories_payload(payload)


def _model_profile_from_draft_payload(
    payload: Mapping[str, object],
    *,
    profile_store: ModelProfileStore | None = None,
) -> ModelConfig:
    return agent_model_profile_from_draft_payload(
        payload,
        profile_store=profile_store,
    )


def _coerce_model_profile_update_payload(
    payload: Mapping[str, object],
) -> tuple[str, dict[str, object], object | None]:
    return coerce_agent_model_profile_update_payload(payload)


def _coerce_model_profile_patch(patch: Mapping[str, object]) -> dict[str, object]:
    return coerce_agent_model_profile_patch(patch)


def _looks_like_placeholder_value(value: object) -> bool:
    return looks_like_placeholder_value(value)


def _coerce_api_keys(value: object) -> tuple[str, ...]:
    return coerce_agent_api_keys(value)


def _model_profile_body(
    profile: ModelConfig,
    *,
    profile_store: ModelProfileStore,
) -> dict[str, object]:
    return agent_model_profile_body(profile, profile_store=profile_store)


def _generate_model_profile_id(body: Mapping[str, object]) -> str:
    return generate_agent_model_profile_id(body)


def _coerce_model_slots(
    value: object,
    *,
    profile_store: ModelProfileStore,
) -> dict[str, str | None]:
    return coerce_agent_model_slots(value, profile_store=profile_store)


def _coerce_prompt_slots(value: object, *, cache_root: Path) -> dict[str, str | None]:
    return coerce_agent_prompt_slots(value, cache_root=cache_root)


def _coerce_workflow_thinking_level(
    value: object,
    *,
    workflow_model_id: str | None,
    profile_store: ModelProfileStore,
) -> str:
    return coerce_agent_workflow_thinking_level(
        value,
        workflow_model_id=workflow_model_id,
        profile_store=profile_store,
    )


def _profile_for_workflow_chat(
    profile: ModelConfig,
    workflow_thinking_level: str,
) -> ModelConfig:
    return profile_for_agent_workflow_chat(profile, workflow_thinking_level)


def _coerce_model_id(
    value: object,
    *,
    profile_store: ModelProfileStore,
    field: str,
) -> str | None:
    return coerce_agent_model_id(value, profile_store=profile_store, field=field)


def _optional_str(value: object) -> str | None:
    return optional_agent_str_value(value)


def _require_active(state: AgentWorkspaceState) -> AgentConversation:
    conversation = state.active()
    if conversation is None:  # pragma: no cover - load() always reseeds one
        raise BridgeError.not_found("no active conversation exists.")
    return conversation


def _derive_title(content: str) -> str:
    flat = " ".join(content.split())
    return flat[:40]


def _workspace_response(
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> dict[str, object]:
    return build_agent_workspace_response(
        state,
        profile_store,
        cache_root,
        normalize_compound_action_drafts=_coerce_compound_action_drafts,
    )


def _workspace_wire(state: AgentWorkspaceState) -> dict[str, object]:
    return workspace_wire_for_response(
        state,
        normalize_compound_action_drafts=_coerce_compound_action_drafts,
    )


def _sanitize_draft_payload(draft: AgentActionDraft) -> dict[str, object]:
    return sanitize_draft_payload_for_wire(
        draft,
        normalize_compound_action_drafts=_coerce_compound_action_drafts,
    )


def _coerce_active_recipe_id(
    raw: object,
    *,
    state: AgentWorkspaceState,
) -> str | None:
    return coerce_agent_active_recipe_id(raw, state=state)


def _optional_agent_string(
    payload: Mapping[str, object],
    key: str,
) -> str | None:
    return coerce_optional_agent_string(payload, key)


def _optional_agent_limit(
    payload: Mapping[str, object],
    *,
    default: int,
) -> int:
    return coerce_optional_agent_limit(payload, default=default)


def _llm_context(
    state: AgentWorkspaceState,
    *,
    settings_store: SettingsStore,
    task_service: TaskService,
) -> dict[str, object]:
    return build_agent_llm_context(
        state,
        settings_store=settings_store,
        task_service=task_service,
        max_context_messages=_MAX_CONTEXT_MESSAGES,
    )


def _settings_defaults(settings_store: SettingsStore) -> dict[str, object]:
    return read_agent_settings_defaults(settings_store)


def _inventory(profile_store: ModelProfileStore, cache_root: Path) -> dict[str, object]:
    return read_agent_inventory(profile_store, cache_root)


def _model_profile_has_placeholder_fields(profile: ModelConfig) -> bool:
    return model_profile_has_placeholder_fields(profile)


def _prompt_summary(preset: PromptPreset) -> dict[str, object]:
    return agent_prompt_summary(preset)


def _prompt_body(preset: PromptPreset) -> dict[str, object]:
    return agent_prompt_body(preset)


def _prompt_store_for(cache_root: Path, kind: PromptKind) -> PromptPresetStore:
    return agent_prompt_store_for(cache_root, kind)


def _generate_prompt_id(
    name: str,
    kind: PromptKind,
    existing: list[PromptPreset],
) -> str:
    return generate_agent_prompt_id(name, kind, existing)

