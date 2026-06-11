"""``agent.*`` bridge handlers for the experimental Agent Lab surface."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from transoria.agent.project_store import AgentProjectStore
from transoria.bridge.handlers.agent_chat_handlers import (
    build_chat_handlers as build_agent_chat_handlers,
)
from transoria.bridge.handlers.agent_compat import (
    _MAX_CONTEXT_MESSAGES,
    _active_task_is_terminal,
    _agent_action_spec,
    _agent_action_specs,
    _append_model_quality_warnings,
    _apply_draft,
    _apply_workspace_patch,
    _classify_agent_intent,
    _coerce_active_recipe_id,
    _coerce_api_keys,
    _coerce_compound_action_drafts,
    _coerce_memories_payload,
    _coerce_memory_items_payload,
    _coerce_model_id,
    _coerce_model_profile_patch,
    _coerce_model_profile_update_payload,
    _coerce_model_slots,
    _coerce_prompt_kind,
    _coerce_prompt_patch,
    _coerce_prompt_preset_payload,
    _coerce_prompt_slots,
    _coerce_prompt_update_payload,
    _coerce_recipe_payload,
    _coerce_recipe_update_payload,
    _coerce_workflow_thinking_level,
    _create_prompt_preset,
    _derive_title,
    _generate_model_profile_id,
    _generate_prompt_id,
    _inventory,
    _llm_context,
    _load_with_reconciled_active_task,
    _looks_like_placeholder_value,
    _model_profile_body,
    _model_profile_from_draft_payload,
    _model_profile_has_placeholder_fields,
    _optional_agent_limit,
    _optional_agent_string,
    _optional_str,
    _profile_for_workflow_chat,
    _prompt_body,
    _prompt_body_from_payload,
    _prompt_store_for,
    _prompt_summary,
    _raise_if_active_task_locked,
    _recipe_body_from_payload,
    _require_active,
    _require_recipe_from_payload,
    _resolve_prompt_for_update,
    _sanitize_draft_payload,
    _settings_defaults,
    _should_discard_pending_for_new_message,
    _update_prompt_preset,
    _validate_compound_action_kind,
    _validate_draft,
    _validate_glossary_task_reference,
    _workspace_response,
    _workspace_wire,
)
from transoria.bridge.handlers.agent_read_handlers import (
    build_read_handlers as build_agent_read_handlers,
)
from transoria.bridge.handlers.agent_state_handlers import (
    build_state_handlers as build_agent_state_handlers,
)
from transoria.bridge.router import BridgeRouter
from transoria.bridge.task_service import TaskService
from transoria.llm.client import LlmClient
from transoria.model_profiles import ModelProfileStore
from transoria.settings import SettingsStore
from transoria.workflows.agent.task_starts import start_agent_task

LlmClientFactory = Callable[[], LlmClient]


def _build_handlers(
    *,
    cache_root: Path,
    project_store: AgentProjectStore,
    profile_store: ModelProfileStore,
    settings_store: SettingsStore,
    task_service: TaskService,
    llm_client_factory: LlmClientFactory,
) -> dict[str, object]:
    return {
        **build_agent_read_handlers(
            cache_root=cache_root,
            project_store=project_store,
            profile_store=profile_store,
            task_service=task_service,
        ),
        **build_agent_state_handlers(
            cache_root=cache_root,
            project_store=project_store,
            profile_store=profile_store,
        ),
        **build_agent_chat_handlers(
            cache_root=cache_root,
            project_store=project_store,
            profile_store=profile_store,
            settings_store=settings_store,
            task_service=task_service,
            llm_client_factory=llm_client_factory,
            start_task=start_agent_task,
        ),
    }


def register(
    router: BridgeRouter,
    *,
    cache_root: Path,
    profile_store: ModelProfileStore,
    settings_store: SettingsStore,
    task_service: TaskService,
    llm_client_factory: LlmClientFactory,
) -> None:
    handlers = _build_handlers(
        cache_root=cache_root,
        project_store=AgentProjectStore.from_cache_root(cache_root),
        profile_store=profile_store,
        settings_store=settings_store,
        task_service=task_service,
        llm_client_factory=llm_client_factory,
    )
    for method, handler in handlers.items():
        router.register(method, handler)  # type: ignore[arg-type]


__all__ = ["register"]
