"""``agent.*`` bridge handlers for the experimental Agent Lab surface."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

from transoria.agent.configuration_agent import (
    AGENT_REPAIR_SYSTEM_PROMPT,
    AGENT_RESPONSE_JSON_SCHEMA,
    AGENT_SYSTEM_PROMPT,
    AgentResponseParseError,
    build_repair_prompt,
    build_user_prompt,
    build_validation_repair_prompt,
    parse_agent_response,
)
from transoria.agent.completeness import (
    START_DRAFT_KINDS,
    assess_start_draft,
)
from transoria.agent.project_store import AgentProjectStore
from transoria.agent.schemas import (
    MODEL_SLOTS,
    AgentActiveTask,
    AgentActionDraft,
    AgentConversation,
    AgentMessage,
    AgentRecipe,
    AgentWorkspaceState,
)
from transoria.bridge.errors import BridgeError
from transoria.bridge.handlers._utils import expect_string
from transoria.bridge.handlers.agent_action_registry import (
    AgentActionSpec,
    MAX_RECIPES as _MAX_RECIPES,
    action_spec as get_agent_action_spec_from_registry,
    action_specs as get_agent_action_specs_from_registry,
    append_model_quality_warnings as append_agent_model_quality_warnings,
    apply_action_draft as apply_agent_action_draft,
    coerce_compound_action_drafts as coerce_agent_compound_action_drafts,
    compound_raw_actions as agent_compound_raw_actions,
    validate_action_draft as validate_agent_action_draft,
    validate_compound_action_kind as validate_agent_compound_action_kind,
)
from transoria.bridge.handlers.agent_draft_salvage import (
    extract_model_concurrency_update as extract_salvage_model_concurrency_update,
    extract_named_value as extract_salvage_named_value,
    extract_prompt_rename as extract_salvage_prompt_rename,
    extract_quoted_name as extract_salvage_quoted_name,
    extract_recipe_save_name as extract_salvage_recipe_save_name,
    resolve_prompt_by_name as resolve_salvage_prompt_by_name,
    salvage_compound_draft as salvage_agent_compound_draft,
    salvage_create_prompt_draft as salvage_agent_create_prompt_draft,
    salvage_create_recipe_draft as salvage_agent_create_recipe_draft,
    salvage_invalid_agent_draft as salvage_agent_invalid_draft,
)
from transoria.bridge.handlers.agent_intents import (
    INTENT_COMPOUND_CONFIG,
    INTENT_FALLBACK,
    INTENT_MODEL_PROFILE_COPY,
    INTENT_MODEL_PROFILE_GUIDANCE,
    INTENT_MODEL_UPGRADE,
    INTENT_PROMPT_PRESET,
    INTENT_PROMPT_QUALITY,
    INTENT_TASK_START,
    AgentIntent,
    AgentIntentSignals,
    classify_agent_intent,
    direct_intent_sequence,
)
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
from transoria.bridge.handlers.agent_model_intents import (
    current_workspace_profiles as current_agent_workspace_profiles,
    extract_model_copy_display_name as extract_agent_model_copy_display_name,
    extract_model_copy_source_labels as extract_agent_model_copy_source_labels,
    extract_provider_model_id_override as extract_agent_provider_model_id_override,
    extract_task_model_reference_labels as extract_agent_task_model_reference_labels,
    find_existing_upgrade_profile as find_agent_existing_upgrade_profile,
    infer_provider_model_id_from_copy_request as infer_agent_provider_model_id_from_copy_request,
    is_same_model_family as is_agent_same_model_family,
    is_usable_task_model_label as is_usable_agent_task_model_label,
    lookup_tokens as agent_lookup_tokens,
    looks_like_model_profile_copy_continuation as looks_like_agent_model_profile_copy_continuation,
    looks_like_model_profile_copy_request as looks_like_agent_model_profile_copy_request,
    model_family_tokens_for_profile as agent_model_family_tokens_for_profile,
    model_family_tokens_from_text as agent_model_family_tokens_from_text,
    model_profile_copy_payload as agent_model_profile_copy_payload,
    model_quality_score as agent_model_quality_score,
    normalize_lookup_label as normalize_agent_lookup_label,
    requested_existing_model_for_task as requested_agent_existing_model_for_task,
    requests_different_provider_model_id as agent_requests_different_provider_model_id,
    resolve_model_profile_by_fuzzy_text as resolve_agent_model_profile_by_fuzzy_text,
    resolve_model_profile_by_label as resolve_agent_model_profile_by_label,
    resolve_model_profile_from_copy_request as resolve_agent_model_profile_from_copy_request,
)
from transoria.bridge.handlers.agent_memory_actions import (
    apply_add_memory_action as apply_agent_add_memory_action,
    apply_delete_memory_action as apply_agent_delete_memory_action,
    apply_update_memory_action as apply_agent_update_memory_action,
    coerce_memories_payload as coerce_agent_memories_payload,
    coerce_memory_items_payload as coerce_agent_memory_items_payload,
    validate_add_memory_action as validate_agent_add_memory_action,
    validate_delete_memory_action as validate_agent_delete_memory_action,
    validate_update_memory_action as validate_agent_update_memory_action,
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
from transoria.bridge.handlers.agent_prompt_intents import (
    direct_prompt_preset_response as build_direct_prompt_preset_response,
    direct_prompt_quality_response as build_direct_prompt_quality_response,
    extract_prompt_body_from_request as extract_agent_prompt_body_from_request,
    extract_prompt_create_name as extract_agent_prompt_create_name,
    has_prompt_creation_requirements as has_agent_prompt_creation_requirements,
    looks_like_direct_prompt_preset_request as looks_like_agent_direct_prompt_preset_request,
    looks_like_prompt_quality_request as looks_like_agent_prompt_quality_request,
    prompt_kind_label as agent_prompt_kind_label,
    prompt_slot_for_kind as agent_prompt_slot_for_kind,
    quality_issue_markers as agent_quality_issue_markers,
    quality_prompt_addendum as agent_quality_prompt_addendum,
    quality_prompt_kind as agent_quality_prompt_kind,
    quality_prompt_name as agent_quality_prompt_name,
    selected_prompt_for_kind as selected_agent_prompt_for_kind,
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
from transoria.bridge.handlers.agent_task_drafts import (
    build_start_task_draft,
    build_start_with_default_stage_config,
)
from transoria.bridge.handlers.agent_task_intents import (
    asks_for_reviewed_glossary as _asks_for_reviewed_glossary,
    extract_absolute_path_candidates as _extract_absolute_path_candidates,
    extract_glossary_review_task_id as _extract_glossary_review_task_id,
    extract_glossary_task_dirs as _extract_glossary_task_dirs,
    extract_glossary_task_id as _extract_glossary_task_id,
    extract_novel_background as _extract_novel_background,
    extract_source_language as _extract_source_language,
    extract_target_language as _extract_target_language,
    is_output_inside_input as _is_output_inside_input,
    latest_completed_glossary_review_task_id as _latest_completed_glossary_review_task_id,
    latest_completed_glossary_task_id as _latest_completed_glossary_task_id,
    looks_like_glossary_extraction_continuation as _looks_like_glossary_extraction_continuation,
    looks_like_glossary_extraction_request as _looks_like_glossary_extraction_request,
    looks_like_glossary_review_followup_request as _looks_like_glossary_review_followup_request,
    looks_like_glossary_review_request as _looks_like_glossary_review_request,
    looks_like_same_directory_output as _looks_like_same_directory_output,
    looks_like_task_start_request as _looks_like_task_start_request,
    looks_like_translation_after_review_followup_request as _looks_like_translation_after_review_followup_request,
    looks_like_translation_task_request as _looks_like_translation_task_request,
    safe_translation_output_dir as _safe_translation_output_dir,
)
from transoria.bridge.handlers.agent_tasks import (
    active_task_is_terminal,
    apply_start_task_action,
    load_with_reconciled_active_task,
    raise_if_active_task_locked,
    validate_glossary_task_reference,
    validate_start_task_action,
)
from transoria.bridge.handlers.agent_task_status import (
    AGENT_TASK_KINDS,
    applied_draft_message,
    artifact_availability,
    coerce_int as coerce_agent_status_int,
    direct_glossary_stage_status_response as build_direct_glossary_stage_status_response,
    direct_stage_status_response as build_direct_stage_status_response,
    direct_translation_status_response as build_direct_translation_status_response,
    extract_translation_task_id as extract_agent_translation_task_id,
    format_glossary_review_status_response as format_agent_glossary_review_status_response,
    format_glossary_status_response as format_agent_glossary_status_response,
    latest_status_kind as latest_agent_status_kind,
    latest_task_id as latest_agent_task_id,
    latest_translation_task_id as latest_agent_translation_task_id,
    looks_like_stage_status_query as looks_like_agent_stage_status_query,
    looks_like_translation_status_query as looks_like_agent_translation_status_query,
    low_confidence_count as agent_low_confidence_count,
    low_confidence_examples as agent_low_confidence_examples,
    read_json_artifact as read_agent_json_artifact,
    read_translation_statistics as read_agent_translation_statistics,
    recent_task_summaries,
    requested_glossary_status_kind as requested_agent_glossary_status_kind,
    string_list as agent_string_list,
    task_header_or_none,
    task_status_context,
    task_status_label,
    validate_agent_task_kind,
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
    resolved_active_recipe_id as _resolved_active_recipe_id,
    sanitize_draft_payload as sanitize_draft_payload_for_wire,
    workspace_wire as workspace_wire_for_response,
)
from transoria.bridge.router import BridgeRouter
from transoria.bridge.task_service import TaskService
from transoria.domain import TaskKind
from transoria.llm.client import ChatRequest, ChatResponse, LlmClient, LlmRequestError
from transoria.llm.config import ModelConfig
from transoria.model_profiles import ModelProfileStore
from transoria.prompts import (
    PromptKind,
    PromptPreset,
    PromptPresetStore,
)
from transoria.settings import SettingsStore
from transoria.workflows.agent.task_starts import start_agent_task

LlmClientFactory = Callable[[], LlmClient]

_MAX_TITLE_LENGTH = 120
_MAX_CONTEXT_MESSAGES = 20


def _build_handlers(
    *,
    cache_root: Path,
    project_store: AgentProjectStore,
    profile_store: ModelProfileStore,
    settings_store: SettingsStore,
    task_service: TaskService,
    llm_client_factory: LlmClientFactory,
) -> dict[str, object]:
    def respond(state: AgentWorkspaceState) -> dict[str, object]:
        return _workspace_response(state, profile_store, cache_root)

    def read_workspace(_payload: Mapping[str, object]) -> dict[str, object]:
        return respond(_load_with_reconciled_active_task(project_store, task_service))

    def list_model_profiles(_payload: Mapping[str, object]) -> dict[str, object]:
        return {"profiles": _inventory(profile_store, cache_root)["profiles"]}

    def list_prompt_presets(payload: Mapping[str, object]) -> dict[str, object]:
        prompts = _inventory(profile_store, cache_root)["prompts"]
        kind = _optional_agent_string(payload, "kind")
        if kind is None:
            return {"prompts": prompts}
        if kind not in {prompt_kind.value for prompt_kind in PromptKind}:
            raise BridgeError.invalid_argument(
                f"unsupported prompt kind: {kind!r}",
                field="kind",
            )
        return {"kind": kind, "presets": prompts[kind]}  # type: ignore[index]

    def list_recipes(_payload: Mapping[str, object]) -> dict[str, object]:
        state = _load_with_reconciled_active_task(project_store, task_service)
        return {
            "recipes": [recipe.to_dict() for recipe in state.recipes],
            "active_recipe_id": _resolved_active_recipe_id(state),
        }

    def get_active_task(_payload: Mapping[str, object]) -> dict[str, object]:
        state = _load_with_reconciled_active_task(project_store, task_service)
        active = state.active_task
        if active is None:
            return {"active_task": None, "task": None}
        return {
            "active_task": active.to_dict(),
            "task": task_header_or_none(task_service, active.kind, active.task_id),
        }

    def list_recent_task_summaries(payload: Mapping[str, object]) -> dict[str, object]:
        kind = _optional_agent_string(payload, "kind")
        limit = _optional_agent_limit(payload, default=5)
        if kind is not None:
            validate_agent_task_kind(kind)
            return {
                "kind": kind,
                "tasks": recent_task_summaries(
                    task_service,
                    kind=kind,
                    limit=limit,
                ),
            }
        return {
            "tasks_by_kind": {
                task_kind: recent_task_summaries(
                    task_service,
                    kind=task_kind,
                    limit=limit,
                )
                for task_kind in AGENT_TASK_KINDS
            }
        }

    def get_artifact_availability(payload: Mapping[str, object]) -> dict[str, object]:
        kind = expect_string(payload, "kind").strip()
        validate_agent_task_kind(kind)
        task_id = expect_string(payload, "task_id").strip()
        if not task_id:
            raise BridgeError.invalid_argument(
                "task_id must not be empty.",
                field="task_id",
            )
        return artifact_availability(task_service, kind=kind, task_id=task_id)

    def update_workspace(payload: Mapping[str, object]) -> dict[str, object]:
        patch = payload.get("patch")
        if not isinstance(patch, Mapping):
            raise BridgeError.invalid_argument(
                "patch object is required.",
                field="patch",
            )
        state = project_store.update(
            lambda current: _apply_workspace_patch(
                current,
                patch,
                profile_store=profile_store,
                cache_root=cache_root,
            )
        )
        return respond(state)

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

        current = _load_with_reconciled_active_task(
            project_store,
            task_service,
            reconcile_stale=False,
        )
        conversation = _require_active(current)
        previous_pending = conversation.pending_draft
        conversation = conversation.append_message(
            AgentMessage.create("user", content)
        )
        if not conversation.title:
            conversation = conversation.with_title(_derive_title(content))
        state_with_user = current.with_active(conversation)

        reply, draft = _generate_reply(
            state_with_user,
            user_message=content,
            profile_store=profile_store,
            cache_root=cache_root,
            settings_store=settings_store,
            task_service=task_service,
            llm_client_factory=llm_client_factory,
        )
        conversation = conversation.append_message(
            AgentMessage.create("assistant", reply)
        )
        if previous_pending is not None and (
            draft is not None or _should_discard_pending_for_new_message(content)
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
        current = _load_with_reconciled_active_task(project_store, task_service)
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
        )
        conversation = _require_active(applied_state)
        conversation = conversation.archive_pending(
            "applied",
            payload=_sanitize_draft_payload(draft),
        )
        conversation = conversation.append_message(
            AgentMessage.create(
                "assistant",
                applied_draft_message(
                    draft,
                    result,
                    task_service=task_service,
                ),
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

        current = _load_with_reconciled_active_task(project_store, task_service)
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

        reply, revised_draft = _generate_reply(
            state_with_user,
            user_message=_draft_revision_prompt(adjustment, draft),
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

    def create_conversation(payload: Mapping[str, object]) -> dict[str, object]:
        raw_title = payload.get("title")
        title = raw_title.strip() if isinstance(raw_title, str) else ""
        if len(title) > _MAX_TITLE_LENGTH:
            raise BridgeError.invalid_argument(
                "title is too long.",
                field="title",
                details={"max_length": _MAX_TITLE_LENGTH},
            )
        conversation = AgentConversation.seeded()
        if title:
            conversation = conversation.with_title(title)
        state = project_store.update(
            lambda current: current.add_conversation(conversation)
        )
        return respond(state)

    def switch_conversation(payload: Mapping[str, object]) -> dict[str, object]:
        conversation_id = expect_string(payload, "conversation_id")
        current = project_store.load()
        if all(conv.id != conversation_id for conv in current.conversations):
            raise BridgeError.not_found(
                f"conversation {conversation_id!r} does not exist.",
                details={"conversation_id": conversation_id},
            )
        state = project_store.save(current.set_active(conversation_id))
        return respond(state)

    def rename_conversation(payload: Mapping[str, object]) -> dict[str, object]:
        conversation_id = expect_string(payload, "conversation_id")
        title = expect_string(payload, "title").strip()
        if not title:
            raise BridgeError.invalid_argument("title must not be empty.", field="title")
        if len(title) > _MAX_TITLE_LENGTH:
            raise BridgeError.invalid_argument(
                "title is too long.",
                field="title",
                details={"max_length": _MAX_TITLE_LENGTH},
            )
        current = project_store.load()
        target = next(
            (conv for conv in current.conversations if conv.id == conversation_id),
            None,
        )
        if target is None:
            raise BridgeError.not_found(
                f"conversation {conversation_id!r} does not exist.",
                details={"conversation_id": conversation_id},
            )
        state = project_store.save(
            current.replace_conversation(target.with_title(title))
        )
        return respond(state)

    def delete_conversation(payload: Mapping[str, object]) -> dict[str, object]:
        conversation_id = expect_string(payload, "conversation_id")
        current = project_store.load()
        if all(conv.id != conversation_id for conv in current.conversations):
            raise BridgeError.not_found(
                f"conversation {conversation_id!r} does not exist.",
                details={"conversation_id": conversation_id},
            )
        next_state = current.remove_conversation(conversation_id)
        if not next_state.conversations:
            next_state = next_state.add_conversation(AgentConversation.seeded())
        state = project_store.save(next_state)
        return respond(state)

    def update_memory(payload: Mapping[str, object]) -> dict[str, object]:
        memories = _coerce_memories_payload(payload)
        state = project_store.update(lambda current: current.with_memories(memories))
        return respond(state)

    def delete_memory(payload: Mapping[str, object]) -> dict[str, object]:
        memory = expect_string(payload, "memory").strip()
        state = project_store.update(
            lambda current: current.with_memories(
                tuple(item for item in current.memories if item != memory)
            )
        )
        return respond(state)

    def create_recipe(payload: Mapping[str, object]) -> dict[str, object]:
        name, description, stage_models, stage_prompts = _coerce_recipe_payload(
            payload,
            profile_store=profile_store,
            cache_root=cache_root,
        )

        def updater(current: AgentWorkspaceState) -> AgentWorkspaceState:
            if len(current.recipes) >= _MAX_RECIPES:
                raise BridgeError.invalid_argument(
                    "too many recipes.",
                    field="recipes",
                    details={"max_count": _MAX_RECIPES},
                )
            recipe = AgentRecipe.create(
                name=name,
                description=description,
                stage_model_ids=stage_models,
                stage_prompt_ids=stage_prompts,
            )
            return current.add_recipe(recipe)

        return respond(project_store.update(updater))

    def update_recipe(payload: Mapping[str, object]) -> dict[str, object]:
        recipe_id = expect_string(payload, "recipe_id")
        name, description, stage_models, stage_prompts = _coerce_recipe_payload(
            payload,
            profile_store=profile_store,
            cache_root=cache_root,
        )

        def updater(current: AgentWorkspaceState) -> AgentWorkspaceState:
            recipe = current.get_recipe(recipe_id)
            if recipe is None:
                raise BridgeError.not_found(
                    f"recipe {recipe_id!r} does not exist.",
                    details={"recipe_id": recipe_id},
                )
            updated = recipe.with_updates(
                name=name,
                description=description,
                stage_model_ids=stage_models,
                stage_prompt_ids=stage_prompts,
            )
            next_state = current.replace_recipe(updated)
            if _resolved_active_recipe_id(current) != recipe_id:
                return next_state
            return next_state.with_config(
                stage_model_ids=dict(updated.stage_model_ids),
                stage_prompt_ids=dict(updated.stage_prompt_ids),
                active_recipe_id=updated.id,
            )

        return respond(project_store.update(updater))

    def delete_recipe(payload: Mapping[str, object]) -> dict[str, object]:
        recipe_id = expect_string(payload, "recipe_id")

        def updater(current: AgentWorkspaceState) -> AgentWorkspaceState:
            if current.get_recipe(recipe_id) is None:
                raise BridgeError.not_found(
                    f"recipe {recipe_id!r} does not exist.",
                    details={"recipe_id": recipe_id},
                )
            return current.remove_recipe(recipe_id)

        return respond(project_store.update(updater))

    def apply_recipe(payload: Mapping[str, object]) -> dict[str, object]:
        recipe_id = expect_string(payload, "recipe_id")

        def updater(current: AgentWorkspaceState) -> AgentWorkspaceState:
            recipe = current.get_recipe(recipe_id)
            if recipe is None:
                raise BridgeError.not_found(
                    f"recipe {recipe_id!r} does not exist.",
                    details={"recipe_id": recipe_id},
                )
            return _apply_workspace_patch(
                current,
                {
                    "active_recipe_id": recipe.id,
                    "stage_model_ids": dict(recipe.stage_model_ids),
                    "stage_prompt_ids": dict(recipe.stage_prompt_ids),
                },
                profile_store=profile_store,
                cache_root=cache_root,
            )

        return respond(project_store.update(updater))

    return {
        "agent.read_workspace": read_workspace,
        "agent.list_model_profiles": list_model_profiles,
        "agent.list_prompt_presets": list_prompt_presets,
        "agent.list_recipes": list_recipes,
        "agent.get_active_task": get_active_task,
        "agent.list_recent_task_summaries": list_recent_task_summaries,
        "agent.get_artifact_availability": get_artifact_availability,
        "agent.update_workspace": update_workspace,
        "agent.send_message": send_message,
        "agent.apply_draft": apply_draft,
        "agent.discard_draft": discard_draft,
        "agent.revise_draft": revise_draft,
        "agent.create_conversation": create_conversation,
        "agent.switch_conversation": switch_conversation,
        "agent.rename_conversation": rename_conversation,
        "agent.delete_conversation": delete_conversation,
        "agent.update_memory": update_memory,
        "agent.delete_memory": delete_memory,
        "agent.create_recipe": create_recipe,
        "agent.update_recipe": update_recipe,
        "agent.delete_recipe": delete_recipe,
        "agent.apply_recipe": apply_recipe,
    }


def _generate_reply(
    state: AgentWorkspaceState,
    *,
    user_message: str,
    profile_store: ModelProfileStore,
    cache_root: Path,
    settings_store: SettingsStore,
    task_service: TaskService,
    llm_client_factory: LlmClientFactory,
) -> tuple[str, AgentActionDraft | None]:
    workflow_model_id = state.workflow_model_id
    conversation_context = _recent_conversation_context(state)
    if not user_message.startswith("The user clicked Adjust on the current pending draft."):
        status_state = task_status_context(task_service)
        if not (
            _looks_like_glossary_review_followup_request(user_message, status_state)
            or _looks_like_translation_after_review_followup_request(
                user_message,
                status_state,
            )
        ):
            direct = _direct_stage_status_response(
                user_message=user_message,
                current_state=status_state,
                task_service=task_service,
            )
            if direct is not None:
                return direct
    current_state = _llm_context(
        state,
        settings_store=settings_store,
        task_service=task_service,
    )
    is_adjust_request = user_message.startswith(
        "The user clicked Adjust on the current pending draft."
    )
    intent = _classify_agent_intent(user_message)
    if not workflow_model_id:
        if not is_adjust_request and intent.kind == INTENT_TASK_START:
            return (
                "请先选择一个工作模型。启动术语、术语审查或翻译任务时，Agent 需要用工作模型补齐缺失的阶段配置并生成确认草案。",
                None,
        )
        direct = None
        if not is_adjust_request:
            direct = _direct_response_for_intent(
                intent=intent,
                user_message=user_message,
                conversation_context=conversation_context,
                state=state,
                current_state=current_state,
                profile_store=profile_store,
                cache_root=cache_root,
                allow_task_start=False,
            )
        if direct is not None:
            reply, draft = direct
            if draft is not None:
                try:
                    _validate_draft(
                        draft,
                        state=state,
                        profile_store=profile_store,
                        cache_root=cache_root,
                        task_service=task_service,
                    )
                except BridgeError as exc:
                    return (f"{reply}\n\n（已忽略无法应用的草案：{exc}）".strip(), None)
                reply = _append_model_quality_warnings(
                    reply,
                    draft=draft,
                    state=state,
                    profile_store=profile_store,
                )
            return reply, draft
        if intent.kind == INTENT_TASK_START:
            return (
                "请先选择一个工作模型。启动术语、术语审查或翻译任务时，Agent 需要用工作模型补齐缺失的阶段配置并生成确认草案。",
                None,
            )
        return (
            "请选择一个工作模型。当前聊天已经可记录；模型/Prompt 等明确配置请求仍可生成确认草案，但需要工作模型才能处理更开放的任务规划。",
            None,
        )
    profile = profile_store.get(workflow_model_id)
    if profile is None:
        return (
            "当前选择的工作模型不存在。请在侧边栏重新选择一个模型配置。",
            None,
        )
    if not profile.api_keys:
        return (
            "当前工作模型没有可用 API key。请先在模型页补全 key，或切换到已配置的模型。",
            None,
        )
    inventory = _inventory(profile_store, cache_root)
    direct = None
    if not is_adjust_request:
        direct = _direct_response_for_intent(
            intent=intent,
            user_message=user_message,
            conversation_context=conversation_context,
            state=state,
            current_state=current_state,
            profile_store=profile_store,
            cache_root=cache_root,
            allow_task_start=True,
        )
    if direct is not None:
        reply, draft = direct
        if draft is not None:
            try:
                _validate_draft(
                    draft,
                    state=state,
                    profile_store=profile_store,
                    cache_root=cache_root,
                    task_service=task_service,
                )
            except BridgeError as exc:
                return (f"{reply}\n\n（已忽略无法应用的草案：{exc}）".strip(), None)
            reply = _append_model_quality_warnings(
                reply,
                draft=draft,
                state=state,
                profile_store=profile_store,
            )
        return reply, draft
    prompt = build_user_prompt(
        user_message=user_message,
        inventory=inventory,
        current_state=current_state,
        conversation_context=conversation_context,
    )
    model = _profile_for_workflow_chat(profile, state.workflow_thinking_level)
    client = llm_client_factory()
    request = ChatRequest(
        model=model,
        system_prompt=AGENT_SYSTEM_PROMPT,
        user_prompt=prompt,
        temperature=0.2,
        stream=False,
        json_response_schema=AGENT_RESPONSE_JSON_SCHEMA,
        json_response_schema_name="agent_configuration_response",
        log_label="agent configuration chat",
    )
    try:
        response = _run_agent_chat_with_schema_fallback(client, request)
    except LlmRequestError as exc:
        return (
            f"工作模型调用失败：[{exc.code}] {exc}",
            None,
        )
    except Exception as exc:  # noqa: BLE001 - keep chat surface recoverable
        return (
            f"工作模型调用失败：{type(exc).__name__}: {exc}",
            None,
        )
    reply, draft = _parse_or_repair_agent_response(
        client,
        model=model,
        user_message=user_message,
        raw_response=response.content,
    )
    if draft is not None:
        try:
            _validate_draft(
                draft,
                state=state,
                profile_store=profile_store,
                cache_root=cache_root,
                task_service=task_service,
            )
        except BridgeError as exc:
            repaired = _repair_invalid_agent_draft(
                client,
                model=model,
                user_message=user_message,
                reply=reply,
                draft=draft,
                validation_error=str(exc),
                inventory=inventory,
                current_state=current_state,
            )
            if repaired is None:
                salvaged = _salvage_invalid_agent_draft(
                    user_message=user_message,
                    reply=reply,
                    draft=draft,
                    current_state=current_state,
                    profile_store=profile_store,
                    cache_root=cache_root,
                )
                if salvaged is not None:
                    try:
                        _validate_draft(
                            salvaged,
                            state=state,
                            profile_store=profile_store,
                            cache_root=cache_root,
                            task_service=task_service,
                        )
                    except BridgeError:
                        salvaged = None
                if salvaged is not None:
                    draft = salvaged
                    reply = (
                        f"{reply}\n\n我已根据你的原始请求补齐为可确认的配置草案，请检查后再应用。"
                    )
                    reply = _append_model_quality_warnings(
                        reply,
                        draft=draft,
                        state=state,
                        profile_store=profile_store,
                    )
                    return reply, draft
                return (f"{reply}\n\n（已忽略无法应用的草案：{exc}）".strip(), None)
            reply, draft = repaired
            try:
                _validate_draft(
                    draft,
                    state=state,
                    profile_store=profile_store,
                    cache_root=cache_root,
                    task_service=task_service,
                )
            except BridgeError as repaired_exc:
                salvaged = _salvage_invalid_agent_draft(
                    user_message=user_message,
                    reply=reply,
                    draft=draft,
                    current_state=current_state,
                    profile_store=profile_store,
                    cache_root=cache_root,
                )
                if salvaged is not None:
                    try:
                        _validate_draft(
                            salvaged,
                            state=state,
                            profile_store=profile_store,
                            cache_root=cache_root,
                            task_service=task_service,
                        )
                    except BridgeError:
                        salvaged = None
                if salvaged is not None:
                    draft = salvaged
                    reply = (
                        f"{reply}\n\n我已根据你的原始请求补齐为可确认的配置草案，请检查后再应用。"
                    )
                    reply = _append_model_quality_warnings(
                        reply,
                        draft=draft,
                        state=state,
                        profile_store=profile_store,
                    )
                    return reply, draft
                return (
                    f"{reply}\n\n（已忽略无法应用的草案：{repaired_exc}）".strip(),
                    None,
                )
        reply = _append_model_quality_warnings(
            reply,
            draft=draft,
            state=state,
            profile_store=profile_store,
        )
    return reply, draft


def _direct_response_for_intent(
    *,
    intent: AgentIntent,
    user_message: str,
    conversation_context: list[Mapping[str, object]],
    state: AgentWorkspaceState,
    current_state: Mapping[str, object],
    profile_store: ModelProfileStore,
    cache_root: Path,
    allow_task_start: bool,
) -> tuple[str, AgentActionDraft | None] | None:
    for kind in direct_intent_sequence(
        intent.kind,
        allow_task_start=allow_task_start,
    ):
        direct = _direct_response_for_kind(
            kind=kind,
            user_message=user_message,
            conversation_context=conversation_context,
            state=state,
            current_state=current_state,
            profile_store=profile_store,
            cache_root=cache_root,
        )
        if direct is not None:
            return direct
    return None


def _direct_response_for_kind(
    *,
    kind: str,
    user_message: str,
    conversation_context: list[Mapping[str, object]],
    state: AgentWorkspaceState,
    current_state: Mapping[str, object],
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> tuple[str, AgentActionDraft | None] | None:
    if kind == INTENT_TASK_START:
        return _direct_task_start_response(
            user_message=user_message,
            conversation_context=conversation_context,
            state=state,
            current_state=current_state,
            profile_store=profile_store,
            cache_root=cache_root,
        )
    if kind == INTENT_COMPOUND_CONFIG:
        return _direct_compound_config_response(
            user_message=user_message,
            current_state=current_state,
            profile_store=profile_store,
            cache_root=cache_root,
        )
    if kind == INTENT_PROMPT_QUALITY:
        return _direct_prompt_quality_response(
            user_message=user_message,
            state=state,
            cache_root=cache_root,
        )
    if kind == INTENT_PROMPT_PRESET:
        return _direct_prompt_preset_response(user_message=user_message)
    if kind == INTENT_MODEL_PROFILE_GUIDANCE:
        return _direct_vague_model_profile_guidance_response(
            user_message=user_message,
            profile_store=profile_store,
        )
    if kind == INTENT_MODEL_PROFILE_COPY:
        return _direct_model_profile_copy_response(
            user_message=user_message,
            conversation_context=conversation_context,
            profile_store=profile_store,
        )
    if kind == INTENT_MODEL_UPGRADE:
        return _direct_model_upgrade_response(
            user_message=user_message,
            state=state,
            profile_store=profile_store,
        )
    return None


def _direct_task_start_response(
    *,
    user_message: str,
    conversation_context: list[Mapping[str, object]],
    state: AgentWorkspaceState,
    current_state: Mapping[str, object],
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> tuple[str, AgentActionDraft | None] | None:
    direct = _direct_translation_response(
        user_message=user_message,
        conversation_context=conversation_context,
        state=state,
        current_state=current_state,
        profile_store=profile_store,
        cache_root=cache_root,
    )
    if direct is None:
        direct = _direct_glossary_review_response(
            user_message=user_message,
            conversation_context=conversation_context,
            state=state,
            current_state=current_state,
            profile_store=profile_store,
            cache_root=cache_root,
        )
    if direct is None:
        direct = _direct_glossary_extraction_response(
            user_message=user_message,
            conversation_context=conversation_context,
            state=state,
            current_state=current_state,
            profile_store=profile_store,
            cache_root=cache_root,
        )
    return direct


def _run_agent_chat_with_schema_fallback(
    client: LlmClient,
    request: ChatRequest,
) -> ChatResponse:
    try:
        return asyncio.run(client.chat(request))
    except LlmRequestError:
        if request.json_response_schema is None:
            raise
        fallback = replace(request, json_response_schema=None)
        return asyncio.run(client.chat(fallback))


def _parse_or_repair_agent_response(
    client: LlmClient,
    *,
    model: ModelConfig,
    user_message: str,
    raw_response: str,
) -> tuple[str, AgentActionDraft | None]:
    try:
        return parse_agent_response(raw_response, require_json=True)
    except AgentResponseParseError:
        pass

    repair_request = ChatRequest(
        model=model,
        system_prompt=AGENT_REPAIR_SYSTEM_PROMPT,
        user_prompt=build_repair_prompt(
            user_message=user_message,
            raw_response=raw_response,
        ),
        temperature=0.0,
        stream=False,
        json_response_schema=AGENT_RESPONSE_JSON_SCHEMA,
        json_response_schema_name="agent_configuration_response",
        log_label="agent draft repair",
    )
    try:
        repaired = _run_agent_chat_with_schema_fallback(client, repair_request)
        return parse_agent_response(repaired.content, require_json=True)
    except (AgentResponseParseError, LlmRequestError):
        return (
            "工作模型返回了无法转换为配置草案的内容。请重试，或把这次配置要求拆短一些。",
            None,
        )


def _repair_invalid_agent_draft(
    client: LlmClient,
    *,
    model: ModelConfig,
    user_message: str,
    reply: str,
    draft: AgentActionDraft,
    validation_error: str,
    inventory: Mapping[str, object],
    current_state: Mapping[str, object],
) -> tuple[str, AgentActionDraft] | None:
    repair_request = ChatRequest(
        model=model,
        system_prompt=AGENT_REPAIR_SYSTEM_PROMPT,
        user_prompt=build_validation_repair_prompt(
            user_message=user_message,
            invalid_reply=reply,
            invalid_draft=draft,
            validation_error=validation_error,
            inventory=inventory,
            current_state=current_state,
        ),
        temperature=0.0,
        stream=False,
        json_response_schema=AGENT_RESPONSE_JSON_SCHEMA,
        json_response_schema_name="agent_configuration_response",
        log_label="agent draft validation repair",
    )
    try:
        repaired = _run_agent_chat_with_schema_fallback(client, repair_request)
        repaired_reply, repaired_draft = parse_agent_response(
            repaired.content,
            require_json=True,
        )
    except (AgentResponseParseError, LlmRequestError):
        return None
    if repaired_draft is None:
        return None
    return repaired_reply, repaired_draft


def _salvage_invalid_agent_draft(
    *,
    user_message: str,
    reply: str,
    draft: AgentActionDraft,
    current_state: Mapping[str, object],
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> AgentActionDraft | None:
    return salvage_agent_invalid_draft(
        user_message=user_message,
        reply=reply,
        draft=draft,
        current_state=current_state,
        profile_store=profile_store,
        cache_root=cache_root,
    )


def _recent_conversation_context(
    state: AgentWorkspaceState,
) -> list[Mapping[str, object]]:
    conversation = state.active()
    if conversation is None:
        return []
    return [
        {
            "role": message.role,
            "content": message.content[:1200],
        }
        for message in conversation.messages[-8:]
    ]


def _conversation_context_text(
    conversation_context: list[Mapping[str, object]],
    *,
    role: str | None = None,
) -> str:
    parts: list[str] = []
    for item in conversation_context:
        if role is not None and item.get("role") != role:
            continue
        content = str(item.get("content") or "").strip()
        if content:
            parts.append(content)
    return "\n".join(parts)


def _latest_user_novel_background(
    conversation_context: list[Mapping[str, object]],
    *,
    current_message: str,
) -> str:
    current = current_message.strip()
    for item in reversed(conversation_context):
        if item.get("role") != "user":
            continue
        content = str(item.get("content") or "").strip()
        if not content or content == current:
            continue
        background = _extract_novel_background(content)
        if background:
            return background
    return ""


def _direct_compound_config_response(
    *,
    user_message: str,
    current_state: Mapping[str, object],
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> tuple[str, AgentActionDraft | None] | None:
    if not _looks_like_direct_compound_config_request(user_message):
        return None
    draft = _salvage_compound_draft(
        AgentActionDraft.create(
            kind="compound_config_update",
            title="复合配置修改",
            summary="根据你的请求准备多个配置修改，并在确认后一次性应用。",
            payload={"actions": []},
        ),
        text=user_message,
        current_state=current_state,
        profile_store=profile_store,
        cache_root=cache_root,
    )
    if draft is None:
        return None
    actions = _compound_raw_actions(draft.payload)
    count = len(actions) if isinstance(actions, list) else 0
    return (
        (
            f"我已准备好包含 {count} 项修改的配置草案。"
            "请检查每项修改，确认后我才会写入配置。"
        ),
        draft,
    )


def _direct_prompt_preset_response(
    *,
    user_message: str,
) -> tuple[str, AgentActionDraft | None] | None:
    return build_direct_prompt_preset_response(
        user_message=user_message,
        excluded_request=_looks_like_task_start_request,
    )


def _direct_prompt_quality_response(
    *,
    user_message: str,
    state: AgentWorkspaceState,
    cache_root: Path,
) -> tuple[str, AgentActionDraft | None] | None:
    return build_direct_prompt_quality_response(
        user_message=user_message,
        state=state,
        cache_root=cache_root,
    )


def _has_prompt_creation_requirements(text: str) -> bool:
    return has_agent_prompt_creation_requirements(text)


def _looks_like_direct_prompt_preset_request(text: str) -> bool:
    return looks_like_agent_direct_prompt_preset_request(
        text,
        excluded_request=_looks_like_task_start_request,
    )


def _extract_prompt_create_name(text: str) -> str:
    return extract_agent_prompt_create_name(text)


def _extract_prompt_body_from_request(text: str) -> str:
    return extract_agent_prompt_body_from_request(text)


def _prompt_kind_label(kind: PromptKind) -> str:
    return agent_prompt_kind_label(kind)


def _looks_like_prompt_quality_request(text: str) -> bool:
    return looks_like_agent_prompt_quality_request(text)


def _quality_prompt_kind(text: str) -> PromptKind:
    return agent_quality_prompt_kind(text)


def _quality_issue_markers(text: str) -> list[str]:
    return agent_quality_issue_markers(text)


def _selected_prompt_for_kind(
    kind: PromptKind,
    *,
    state: AgentWorkspaceState,
    cache_root: Path,
) -> PromptPreset | None:
    return selected_agent_prompt_for_kind(kind, state=state, cache_root=cache_root)


def _prompt_slot_for_kind(kind: PromptKind) -> str:
    return agent_prompt_slot_for_kind(kind)


def _quality_prompt_addendum(*, issues: Sequence[str], user_message: str) -> str:
    return agent_quality_prompt_addendum(issues=issues, user_message=user_message)


def _quality_prompt_name(
    text: str,
    kind: PromptKind,
    source: PromptPreset | None,
) -> str:
    return agent_quality_prompt_name(text, kind, source)


def _looks_like_direct_compound_config_request(text: str) -> bool:
    if _looks_like_translation_task_request(text):
        return False
    if _looks_like_glossary_review_request(text):
        return False
    if _looks_like_glossary_extraction_request(text):
        return False
    if _looks_like_model_profile_copy_request(text):
        return False
    normalized = text.lower()
    return any(
        marker in normalized
        for marker in (
            "并发",
            "concurrency",
            "重命名",
            "改名",
            "保存当前",
            "保存成",
            "保存为",
            "存成",
            "存为",
        )
    )


def _direct_model_profile_copy_response(
    *,
    user_message: str,
    conversation_context: list[Mapping[str, object]],
    profile_store: ModelProfileStore,
) -> tuple[str, AgentActionDraft | None] | None:
    context_text = _conversation_context_text(conversation_context, role="user")
    current_request = _looks_like_model_profile_copy_request(user_message)
    contextual_confirmation = (
        _looks_like_model_profile_copy_request(context_text)
        and _looks_like_model_profile_copy_continuation(user_message)
    )
    if not current_request and not contextual_confirmation:
        return None
    combined_text = "\n".join((context_text, user_message))
    request_text = combined_text if contextual_confirmation else user_message

    source = _resolve_model_profile_from_copy_request(
        request_text,
        profile_store=profile_store,
    )
    if source is None:
        return (
            "我理解你想基于已有模型配置复制创建一个新配置，但没有在当前模型库中唯一匹配到源模型。请明确要复制哪个模型配置名称。",
            None,
        )

    display_name = _extract_model_copy_display_name(user_message)
    if not display_name:
        display_name = _extract_model_copy_display_name(request_text)
    if not display_name:
        return (
            f"我可以基于 {source.display_name} 复制创建新模型配置。请明确新配置的显示名称。",
            None,
        )

    provider_model_id = _extract_provider_model_id_override(user_message)
    if provider_model_id is None:
        provider_model_id = _extract_provider_model_id_override(request_text)
    if provider_model_id is None:
        provider_model_id = _infer_provider_model_id_from_copy_request(
            source,
            display_name=display_name,
            text=request_text,
        )
    if _requests_different_provider_model_id(user_message) and not provider_model_id:
        return (
            (
                f"我可以复用 {source.display_name} 的 provider_format、base_url、并发/限速和 API key 状态，"
                f"并把新配置显示名称设为 {display_name}。但你明确要求不要复用相同的模型 ID，"
                "请给出要写入 provider 的准确 model_id。"
            ),
            None,
        )

    profile_payload = _model_profile_copy_payload(
        source,
        display_name=display_name,
        provider_model_id=provider_model_id,
    )
    model_id_note = (
        f"provider model_id 改为 {provider_model_id}"
        if provider_model_id
        else f"provider model_id 保持为 {source.model_id}"
    )
    draft = AgentActionDraft.create(
        kind="create_model_profile",
        title=f"复制模型配置为 {display_name}",
        summary=(
            f"基于 {source.display_name} 创建新模型配置 {display_name}，"
            f"{model_id_note}，其余运行参数保持一致。"
        ),
        payload={"profile": profile_payload},
    )
    return (
        (
            f"我会基于 {source.display_name} 复制创建新模型配置 {display_name}。"
            f"{model_id_note}；provider_format、base_url、并发/限速、重试、思考配置和已配置的 API key 状态保持一致。"
            "请在草案中确认后应用。"
        ),
        draft,
    )


def _direct_vague_model_profile_guidance_response(
    *,
    user_message: str,
    profile_store: ModelProfileStore,
) -> tuple[str, AgentActionDraft | None] | None:
    if not _looks_like_vague_model_profile_request(user_message):
        return None
    usable_profiles = [
        profile
        for profile in profile_store.load()
        if profile.api_keys and not _model_profile_has_placeholder_fields(profile)
    ]
    examples = ""
    if usable_profiles:
        labels = "、".join(profile.display_name for profile in usable_profiles[:5])
        examples = f"\n\n当前可以复用的已有模型配置：{labels}。"
    return (
        (
            "我可以帮你配置模型，但不能替你猜 provider model_id、base_url 或 API key。"
            "如果你不知道具体信息，请选择一种方式：\n"
            "1. 说“按照某个已有模型复制一个”，并告诉我要改成的准确 provider model_id；\n"
            "2. 直接提供接口类型、base_url、provider model_id 和 API key，我会生成确认草案；\n"
            "3. 如果只是想提高质量，可以让我先查看现有模型，切换到已经配置好的更强模型。\n\n"
            "任何写入都会先生成草案，API key 在预览里会遮罩。"
            f"{examples}"
        ),
        None,
    )


def _direct_model_upgrade_response(
    *,
    user_message: str,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
) -> tuple[str, AgentActionDraft | None] | None:
    if not _looks_like_model_upgrade_request(user_message):
        return None
    target = _find_existing_upgrade_profile(
        user_message,
        state=state,
        profile_store=profile_store,
    )
    if target is None:
        return (
            (
                "我理解你想换成更强的模型配置，但当前模型库里没有唯一可复用的更高质量配置。"
                "请告诉我要使用的准确 provider model_id，或者先在模型页添加一个可用模型；"
                "如果你只是想先用现有配置，我也可以帮你把当前流程切到已有模型。"
            ),
            None,
        )

    stage_models = {slot: target.id for slot in MODEL_SLOTS}
    payload: dict[str, object] = {"stage_model_ids": stage_models}
    current_workflow = (
        profile_store.get(state.workflow_model_id) if state.workflow_model_id else None
    )
    if current_workflow is None or _is_same_model_family(current_workflow, target):
        payload["workflow_model_id"] = target.id
    draft = AgentActionDraft.create(
        kind="update_workspace",
        title=f"切换到更强模型 {target.display_name}",
        summary=(
            f"将翻译、术语提取和术语审查阶段切换到现有模型 {target.display_name}。"
            "如果当前工作模型属于同一模型系列，也会同步切换工作模型。"
        ),
        payload=payload,
    )
    return (
        (
            f"我在现有模型库里找到了更适合质量测试的模型 {target.display_name}。"
            "我不会直接保存；下面是把翻译流程各阶段切换到这个模型的确认草案。"
        ),
        draft,
    )


def _direct_glossary_extraction_response(
    *,
    user_message: str,
    conversation_context: list[Mapping[str, object]],
    state: AgentWorkspaceState,
    current_state: Mapping[str, object],
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> tuple[str, AgentActionDraft | None] | None:
    if _looks_like_glossary_review_request(user_message):
        return None
    context_text = "\n".join(
        str(item.get("content") or "") for item in conversation_context
    )
    current_request = _looks_like_glossary_extraction_request(user_message)
    implicit_current_request = _looks_like_glossary_extraction_continuation(
        user_message
    ) and not _looks_like_model_profile_copy_request(user_message)
    contextual_continuation = (
        _looks_like_glossary_extraction_request(context_text)
        and _looks_like_glossary_extraction_continuation(user_message)
    )
    if (
        not current_request
        and not contextual_continuation
        and not implicit_current_request
    ):
        return None
    combined_text = "\n".join((context_text, user_message))
    extraction_text = combined_text if contextual_continuation else user_message
    input_dir, output_dir, output_defaulted = _extract_glossary_task_dirs(user_message)
    if not input_dir and contextual_continuation:
        input_dir, output_dir, output_defaulted = _extract_glossary_task_dirs(
            extraction_text
        )
    if not input_dir:
        return (
            "我理解你想提取术语。请提供 input 目录；如果不提供 output 目录，我会默认使用 input 目录作为输出目录。",
            None,
        )
    novel_background = _extract_novel_background(user_message)
    if not novel_background and contextual_continuation:
        novel_background = _latest_user_novel_background(
            conversation_context,
            current_message=user_message,
        )
    if not novel_background:
        return (
            "我理解你想提取术语。请再提供小说背景，这会进入本次任务配置，不会写入手动设置页。",
            None,
        )
    source_language = _extract_source_language(user_message) or _settings_default_value(
        current_state,
        "glossary",
        "source_language",
    )
    target_language = _extract_target_language(user_message) or _settings_default_value(
        current_state,
        "glossary",
        "target_language",
    )
    payload = {
        "input_dir": input_dir,
        "output_dir": output_dir or input_dir,
        "source_language": source_language,
        "target_language": target_language,
        "novel_background": novel_background,
    }
    completeness = assess_start_draft(
        draft_kind="start_glossary_task",
        state=state,
        payload=payload,
    )
    requested_stage_model = _requested_existing_model_for_task(
        user_message,
        profile_store=profile_store,
    )
    if not completeness.complete:
        dumb_start = _draft_start_with_default_stage_config(
            draft_kind="start_glossary_task",
            payload=payload,
            state=state,
            completeness_missing=completeness.missing,
            cache_root=cache_root,
            title="补齐术语提取配置并启动任务",
            start_title="启动术语提取",
            start_summary="使用补齐后的术语提取模型和 Prompt，从指定目录提取术语表。",
            reply=(
                "我可以按傻瓜模式处理：先用当前工作模型补齐术语提取模型，"
                "用内置默认术语提取 Prompt 补齐缺失项，然后再启动术语提取任务。"
                "下面是一个组合草案，只有点击「应用」后才会写入工作区配置并启动任务。"
            ),
            requested_stage_model=requested_stage_model,
        )
        if dumb_start is not None:
            return dumb_start
        missing = "、".join(completeness.missing)
        return (
            f"我理解你想提取术语，但当前术语提取任务配置还不完整：{missing}。请先补齐对应模型、Prompt 或语言设置。",
            None,
        )
    note = "未提供 output 目录，草案会默认输出到 input 目录。" if output_defaulted else ""
    draft = build_start_task_draft(
        draft_kind="start_glossary_task",
        title="启动术语提取",
        summary="使用当前术语提取模型和 Prompt，从指定目录提取术语表。",
        payload=payload,
    )
    return (
        (
            "我已准备好术语提取任务草案。"
            f"{note}请确认后再启动；本次目录和背景只作为任务覆盖项，不会写入手动设置。"
        ),
        draft,
    )


def _direct_glossary_review_response(
    *,
    user_message: str,
    conversation_context: list[Mapping[str, object]],
    state: AgentWorkspaceState,
    current_state: Mapping[str, object],
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> tuple[str, AgentActionDraft | None] | None:
    if not (
        _looks_like_glossary_review_request(user_message)
        or _looks_like_glossary_review_followup_request(user_message, current_state)
    ):
        return None
    context_text = _conversation_context_text(conversation_context)
    task_id = (
        _extract_glossary_task_id(user_message)
        or _extract_glossary_task_id(context_text)
        or _latest_completed_glossary_task_id(current_state)
    )
    if not task_id:
        return (
            "我理解你想启动术语审查。请提供已完成的 glossary task ID，或先完成一次术语提取任务。",
            None,
        )
    novel_background = _extract_novel_background(
        user_message
    ) or _latest_user_novel_background(
        conversation_context,
        current_message=user_message,
    )
    payload: dict[str, object] = {
        "glossary_task_id": task_id,
    }
    if novel_background:
        payload["novel_background"] = novel_background
    completeness = assess_start_draft(
        draft_kind="start_glossary_review_task",
        state=state,
        payload=payload,
    )
    requested_stage_model = _requested_existing_model_for_task(
        user_message,
        profile_store=profile_store,
    )
    if not completeness.complete:
        dumb_start = _draft_start_with_default_stage_config(
            draft_kind="start_glossary_review_task",
            payload=payload,
            state=state,
            completeness_missing=completeness.missing,
            cache_root=cache_root,
            title="补齐术语审查配置并启动任务",
            start_title="启动术语审查",
            start_summary=f"使用术语提取任务 {task_id} 的术语表和参考文本启动术语审查。",
            reply=(
                "我可以先用当前工作模型补齐术语审查模型，"
                "用内置默认术语审查 Prompt 补齐缺失项，然后再启动术语审查任务。"
                "下面是一个组合草案，只有点击「应用」后才会写入工作区配置并启动任务。"
            ),
            requested_stage_model=requested_stage_model,
        )
        if dumb_start is not None:
            return dumb_start
        missing = "、".join(completeness.missing)
        return (
            f"我理解你想审查术语，但当前术语审查任务配置还不完整：{missing}。请先补齐对应模型、Prompt 或语言设置。",
            None,
        )
    draft = build_start_task_draft(
        draft_kind="start_glossary_review_task",
        title="启动术语审查",
        summary=f"使用术语提取任务 {task_id} 的术语表和参考文本启动术语审查。",
        payload=payload,
    )
    return (
        (
            f"我已准备好术语审查任务草案，会读取术语提取任务 {task_id} 的输出产物。"
            "请确认后再启动；本次背景只作为任务覆盖项，不会写入手动设置。"
        ),
        draft,
    )


def _direct_translation_response(
    *,
    user_message: str,
    conversation_context: list[Mapping[str, object]],
    state: AgentWorkspaceState,
    current_state: Mapping[str, object],
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> tuple[str, AgentActionDraft | None] | None:
    followup_request = _looks_like_translation_after_review_followup_request(
        user_message,
        current_state,
    )
    if not _looks_like_translation_task_request(user_message) and not followup_request:
        return None
    context_text = "\n".join(
        str(item.get("content") or "") for item in conversation_context
    )
    combined_text = "\n".join((context_text, user_message))
    input_dir, output_dir, _ = _extract_glossary_task_dirs(user_message)
    if not input_dir:
        input_dir, output_dir, _ = _extract_glossary_task_dirs(combined_text)
    same_dir_requested = _looks_like_same_directory_output(
        user_message
    ) or _looks_like_same_directory_output(combined_text)
    if not input_dir:
        if not _extract_absolute_path_candidates(combined_text):
            if followup_request:
                return (
                    "我理解你已经确认术语表，下一步是启动翻译。请提供 input 目录和一个独立的 output 目录；翻译输出不能默认写回输入目录。",
                    None,
                )
            return None
        return (
            "我理解你想启动翻译。请提供 input 目录和一个独立的 output 目录；翻译输出不能默认写回输入目录。",
            None,
        )
    output_adjustment_note = ""
    if not output_dir and same_dir_requested:
        output_dir = _safe_translation_output_dir(input_dir)
        output_adjustment_note = (
            f"你提到同目录输出；但翻译不能写回 input，我已在草案中改为安全同级输出目录：{output_dir}。"
        )
    if not output_dir:
        return (
            "我理解你想启动翻译，但还缺 output 目录。翻译输出需要使用独立目录，避免覆盖源文件。",
            None,
        )
    if input_dir.rstrip("/") == output_dir.rstrip("/"):
        if same_dir_requested:
            output_dir = _safe_translation_output_dir(input_dir)
            output_adjustment_note = (
                f"你提到同目录输出；但翻译不能写回 input，我已在草案中改为安全同级输出目录：{output_dir}。"
            )
        else:
            return (
                "我理解你想启动翻译，但 input 和 output 目录不能相同。请提供一个独立的 output 目录。",
                None,
            )
    if _is_output_inside_input(input_dir, output_dir):
        return (
            "我理解你想启动翻译，但 output 目录不能放在 input 目录里面，否则译后文件会被下一次扫描当成源文。请提供一个独立的 output 目录。",
            None,
        )
    source_language = _extract_source_language(user_message) or _settings_default_value(
        current_state,
        "translation",
        "source_language",
    )
    target_language = _extract_target_language(user_message) or _settings_default_value(
        current_state,
        "translation",
        "target_language",
    )
    payload: dict[str, object] = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "source_language": source_language,
        "target_language": target_language,
    }
    glossary_review_task_id = (
        _extract_glossary_review_task_id(user_message)
        or _extract_glossary_review_task_id(context_text)
        or (
            _latest_completed_glossary_review_task_id(current_state)
            if (_asks_for_reviewed_glossary(combined_text) or followup_request)
            else None
        )
    )
    glossary_task_id = None
    if glossary_review_task_id:
        payload["glossary_review_task_id"] = glossary_review_task_id
    else:
        glossary_task_id = _extract_glossary_task_id(
            user_message
        ) or _extract_glossary_task_id(context_text)
        if glossary_task_id:
            payload["glossary_task_id"] = glossary_task_id
    completeness = assess_start_draft(
        draft_kind="start_translation_task",
        state=state,
        payload=payload,
    )
    requested_stage_model = _requested_existing_model_for_task(
        user_message,
        profile_store=profile_store,
    )
    if not completeness.complete:
        dumb_start = _draft_start_with_default_stage_config(
            draft_kind="start_translation_task",
            payload=payload,
            state=state,
            completeness_missing=completeness.missing,
            cache_root=cache_root,
            title="补齐翻译配置并启动任务",
            start_title="启动翻译",
            start_summary="使用补齐后的翻译模型和 Prompt 翻译指定目录。",
            reply=(
                f"{output_adjustment_note}"
                "我可以按当前工作区的傻瓜模式处理：先用当前工作模型补齐翻译模型，"
                "用内置默认翻译 Prompt 补齐缺失项，然后再启动翻译任务。"
                "下面是一个组合草案，只有点击「应用」后才会写入工作区配置并启动任务。"
            ),
            requested_stage_model=requested_stage_model,
        )
        if dumb_start is not None:
            return dumb_start
        missing = "、".join(completeness.missing)
        return (
            f"我理解你想启动翻译，但当前翻译任务配置还不完整：{missing}。请先补齐对应模型、Prompt 或语言设置。",
            None,
        )
    glossary_note = (
        f"并引用术语审查任务 {glossary_review_task_id} 的确认术语表。"
        if glossary_review_task_id
        else f"并引用术语提取任务 {glossary_task_id} 的术语表。"
        if glossary_task_id
        else "不引用历史术语任务，使用当前翻译设置中的术语表。"
    )
    draft = build_start_task_draft(
        draft_kind="start_translation_task",
        title="启动翻译",
        summary=f"使用当前翻译模型和 Prompt 翻译指定目录，{glossary_note}",
        payload=payload,
    )
    return (
        (
            "我已准备好翻译任务草案。"
            f"{output_adjustment_note}"
            f"{glossary_note}请确认后再启动；本次目录和语言只作为任务覆盖项，不会写入手动设置。"
        ),
        draft,
    )


def _draft_start_with_default_stage_config(
    *,
    draft_kind: str,
    payload: Mapping[str, object],
    state: AgentWorkspaceState,
    completeness_missing: Sequence[str],
    cache_root: Path,
    title: str,
    start_title: str,
    start_summary: str,
    reply: str,
    requested_stage_model: ModelConfig | None = None,
) -> tuple[str, AgentActionDraft] | None:
    return build_start_with_default_stage_config(
        draft_kind=draft_kind,
        payload=payload,
        state=state,
        completeness_missing=completeness_missing,
        cache_root=cache_root,
        title=title,
        start_title=start_title,
        start_summary=start_summary,
        reply=reply,
        requested_stage_model=requested_stage_model,
    )


def _direct_stage_status_response(
    *,
    user_message: str,
    current_state: Mapping[str, object],
    task_service: TaskService,
) -> tuple[str, None] | None:
    return build_direct_stage_status_response(
        user_message=user_message,
        current_state=current_state,
        task_service=task_service,
        extract_glossary_task_id=_extract_glossary_task_id,
        extract_glossary_review_task_id=_extract_glossary_review_task_id,
    )


def _direct_glossary_stage_status_response(
    *,
    user_message: str,
    current_state: Mapping[str, object],
    task_service: TaskService,
) -> tuple[str, None] | None:
    return build_direct_glossary_stage_status_response(
        user_message=user_message,
        current_state=current_state,
        task_service=task_service,
        extract_glossary_task_id=_extract_glossary_task_id,
        extract_glossary_review_task_id=_extract_glossary_review_task_id,
    )


def _direct_translation_status_response(
    *,
    user_message: str,
    current_state: Mapping[str, object],
    task_service: TaskService,
) -> tuple[str, None] | None:
    return build_direct_translation_status_response(
        user_message=user_message,
        current_state=current_state,
        task_service=task_service,
    )


def _requested_glossary_status_kind(
    text: str,
    current_state: Mapping[str, object],
) -> str | None:
    return requested_agent_glossary_status_kind(
        text,
        current_state,
        extract_glossary_task_id=_extract_glossary_task_id,
        extract_glossary_review_task_id=_extract_glossary_review_task_id,
    )


def _looks_like_stage_status_query(normalized: str) -> bool:
    return looks_like_agent_stage_status_query(normalized)


def _latest_status_kind(
    current_state: Mapping[str, object],
    *,
    kinds: tuple[str, ...],
) -> str | None:
    return latest_agent_status_kind(current_state, kinds=kinds)


def _latest_task_id(current_state: Mapping[str, object], *, kind: str) -> str | None:
    return latest_agent_task_id(current_state, kind=kind)


def _format_glossary_status_response(
    *,
    task_id: str,
    status: str,
    artifacts: Mapping[str, object],
) -> str:
    return format_agent_glossary_status_response(
        task_id=task_id,
        status=status,
        artifacts=artifacts,
    )


def _format_glossary_review_status_response(
    *,
    task_id: str,
    status: str,
    artifacts: Mapping[str, object],
) -> str:
    return format_agent_glossary_review_status_response(
        task_id=task_id,
        status=status,
        artifacts=artifacts,
    )


def _looks_like_translation_status_query(text: str) -> bool:
    return looks_like_agent_translation_status_query(text)


def _extract_translation_task_id(text: str) -> str | None:
    return extract_agent_translation_task_id(text)


def _latest_translation_task_id(current_state: Mapping[str, object]) -> str | None:
    return latest_agent_translation_task_id(current_state)


def _read_translation_statistics(
    artifacts: Mapping[str, object],
) -> Mapping[str, object]:
    return read_agent_translation_statistics(artifacts)


def _read_json_artifact(
    artifacts: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    return read_agent_json_artifact(artifacts, key)


def _low_confidence_count(value: object) -> int | None:
    return agent_low_confidence_count(value)


def _low_confidence_examples(value: object) -> list[str]:
    return agent_low_confidence_examples(value)


def _coerce_int(value: object) -> int | None:
    return coerce_agent_status_int(value)


def _string_list(value: object) -> list[str]:
    return agent_string_list(value)


def _classify_agent_intent(text: str) -> AgentIntent:
    return classify_agent_intent(
        AgentIntentSignals(
            task_start=_looks_like_task_start_request(text),
            compound_config=_looks_like_direct_compound_config_request(text),
            prompt_quality=_looks_like_prompt_quality_request(text),
            prompt_preset=_looks_like_direct_prompt_preset_request(text),
            model_profile_guidance=_looks_like_vague_model_profile_request(text),
            model_profile_copy=_looks_like_model_profile_copy_request(text),
            model_upgrade=_looks_like_model_upgrade_request(text),
        )
    )


def _should_discard_pending_for_new_message(text: str) -> bool:
    return _classify_agent_intent(text).kind != INTENT_FALLBACK


def _settings_default_value(
    current_state: Mapping[str, object],
    section: str,
    key: str,
) -> str:
    defaults = current_state.get("settings_defaults")
    if not isinstance(defaults, Mapping):
        return ""
    section_defaults = defaults.get(section)
    if not isinstance(section_defaults, Mapping):
        return ""
    value = section_defaults.get(key)
    return value.strip() if isinstance(value, str) else ""


def _looks_like_model_profile_copy_request(text: str) -> bool:
    return looks_like_agent_model_profile_copy_request(text)


def _looks_like_model_profile_copy_continuation(text: str) -> bool:
    return looks_like_agent_model_profile_copy_continuation(text)


def _looks_like_vague_model_profile_request(text: str) -> bool:
    normalized = text.lower()
    if (
        _looks_like_model_profile_copy_request(text)
        or _looks_like_model_upgrade_request(text)
        or _looks_like_translation_task_request(text)
        or _looks_like_glossary_extraction_request(text)
        or _looks_like_glossary_review_request(text)
        or _looks_like_direct_prompt_preset_request(text)
    ):
        return False
    has_model = any(marker in normalized for marker in ("模型", "model", "profile"))
    has_create_or_config = any(
        marker in normalized
        for marker in (
            "新增",
            "添加",
            "加一个",
            "新建",
            "创建",
            "配置",
            "接入",
            "add",
            "create",
            "configure",
        )
    )
    has_uncertainty = any(
        marker in normalized
        for marker in (
            "不知道",
            "不清楚",
            "不会",
            "不懂",
            "不了解",
            "随便",
            "你帮我",
            "帮我配",
            "帮我配置",
            "不确定",
        )
    )
    return has_model and has_create_or_config and has_uncertainty


def _looks_like_model_upgrade_request(text: str) -> bool:
    normalized = text.lower()
    if (
        _looks_like_model_profile_copy_request(text)
        or _looks_like_translation_task_request(text)
        or _looks_like_glossary_extraction_request(text)
        or _looks_like_glossary_review_request(text)
        or _looks_like_direct_prompt_preset_request(text)
    ):
        return False
    has_create_or_config = any(
        marker in normalized
        for marker in (
            "新增",
            "添加",
            "加一个",
            "新建",
            "创建",
            "配置",
            "接入",
            "add",
            "create",
            "configure",
        )
    )
    has_uncertainty = any(
        marker in normalized
        for marker in (
            "不知道",
            "不清楚",
            "不会",
            "不懂",
            "不了解",
            "随便",
            "你帮我",
            "帮我配",
            "帮我配置",
            "不确定",
        )
    )
    has_existing_inventory_intent = any(
        marker in normalized
        for marker in (
            "现有配置",
            "已有配置",
            "当前配置",
            "模型库",
            "已经配置",
            "已配置",
            "inventory",
        )
    )
    if has_create_or_config and has_uncertainty and not has_existing_inventory_intent:
        return False
    has_model = any(marker in normalized for marker in ("模型", "model", "profile"))
    has_quality_intent = any(
        marker in normalized
        for marker in (
            "更厉害",
            "更强",
            "更好",
            "高质量",
            "高规格",
            "高级",
            "pro",
            "质量",
            "升级",
            "换成",
            "切换",
            "better",
            "stronger",
            "upgrade",
        )
    )
    has_agent_help = any(
        marker in normalized
        for marker in (
            "不知道",
            "不清楚",
            "不会",
            "你帮我",
            "帮我看",
            "看看",
            "帮我",
            "配置",
            "修改方案",
            "方案",
        )
    )
    return has_model and has_quality_intent and has_agent_help


def _find_existing_upgrade_profile(
    user_message: str,
    *,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
) -> ModelConfig | None:
    return find_agent_existing_upgrade_profile(
        user_message,
        state=state,
        profile_store=profile_store,
    )


def _current_workspace_profiles(
    state: AgentWorkspaceState,
    *,
    profile_store: ModelProfileStore,
) -> list[ModelConfig]:
    return current_agent_workspace_profiles(state, profile_store=profile_store)


def _is_same_model_family(left: ModelConfig, right: ModelConfig) -> bool:
    return is_agent_same_model_family(left, right)


def _model_family_tokens_for_profile(profile: ModelConfig) -> set[str]:
    return agent_model_family_tokens_for_profile(profile)


def _model_family_tokens_from_text(text: str) -> set[str]:
    return agent_model_family_tokens_from_text(text)


def _model_quality_score(profile: ModelConfig) -> int:
    return agent_model_quality_score(profile)


def _model_profile_copy_payload(
    source: ModelConfig,
    *,
    display_name: str,
    provider_model_id: str | None,
) -> dict[str, object]:
    return agent_model_profile_copy_payload(
        source,
        display_name=display_name,
        provider_model_id=provider_model_id,
    )


def _resolve_model_profile_from_copy_request(
    text: str,
    *,
    profile_store: ModelProfileStore,
) -> ModelConfig | None:
    return resolve_agent_model_profile_from_copy_request(
        text,
        profile_store=profile_store,
    )


def _requested_existing_model_for_task(
    text: str,
    *,
    profile_store: ModelProfileStore,
) -> ModelConfig | None:
    return requested_agent_existing_model_for_task(text, profile_store=profile_store)


def _extract_task_model_reference_labels(text: str) -> list[str]:
    return extract_agent_task_model_reference_labels(text)


def _is_usable_task_model_label(label: str) -> bool:
    return is_usable_agent_task_model_label(label)


def _extract_model_copy_source_labels(text: str) -> list[str]:
    return extract_agent_model_copy_source_labels(text)


def _resolve_model_profile_by_fuzzy_text(
    text: str,
    profile_store: ModelProfileStore,
) -> ModelConfig | None:
    return resolve_agent_model_profile_by_fuzzy_text(text, profile_store)


def _lookup_tokens(value: str) -> set[str]:
    return agent_lookup_tokens(value)


def _extract_model_copy_display_name(text: str) -> str:
    return extract_agent_model_copy_display_name(text)


def _extract_provider_model_id_override(text: str) -> str | None:
    return extract_agent_provider_model_id_override(text)


def _infer_provider_model_id_from_copy_request(
    source: ModelConfig,
    *,
    display_name: str,
    text: str,
) -> str | None:
    return infer_agent_provider_model_id_from_copy_request(
        source,
        display_name=display_name,
        text=text,
    )


def _requests_different_provider_model_id(text: str) -> bool:
    return agent_requests_different_provider_model_id(text)


def _salvage_create_prompt_draft(
    draft: AgentActionDraft,
    *,
    text: str,
) -> AgentActionDraft | None:
    return salvage_agent_create_prompt_draft(draft, text=text)


def _salvage_create_recipe_draft(
    draft: AgentActionDraft,
    *,
    text: str,
    current_state: Mapping[str, object],
) -> AgentActionDraft | None:
    return salvage_agent_create_recipe_draft(
        draft,
        text=text,
        current_state=current_state,
    )


def _salvage_compound_draft(
    draft: AgentActionDraft,
    *,
    text: str,
    current_state: Mapping[str, object],
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> AgentActionDraft | None:
    return salvage_agent_compound_draft(
        draft,
        text=text,
        current_state=current_state,
        profile_store=profile_store,
        cache_root=cache_root,
    )


def _extract_quoted_name(text: str) -> str:
    return extract_salvage_quoted_name(text)


def _extract_named_value(text: str) -> str:
    return extract_salvage_named_value(text)


def _extract_model_concurrency_update(
    text: str,
    *,
    profile_store: ModelProfileStore,
) -> tuple[str, int] | None:
    return extract_salvage_model_concurrency_update(
        text,
        profile_store=profile_store,
    )


def _resolve_model_profile_by_label(
    label: str,
    profile_store: ModelProfileStore,
) -> ModelConfig | None:
    return resolve_agent_model_profile_by_label(label, profile_store)


def _extract_prompt_rename(
    text: str,
    *,
    cache_root: Path,
) -> tuple[PromptPreset, str, str] | None:
    return extract_salvage_prompt_rename(text, cache_root=cache_root)


def _resolve_prompt_by_name(
    name: str,
    *,
    kinds: tuple[PromptKind, ...],
    cache_root: Path,
) -> PromptPreset | None:
    return resolve_salvage_prompt_by_name(
        name,
        kinds=kinds,
        cache_root=cache_root,
    )


def _extract_recipe_save_name(text: str) -> str:
    return extract_salvage_recipe_save_name(text)


def _normalize_lookup_label(value: str) -> str:
    return normalize_agent_lookup_label(value)


def _draft_revision_prompt(adjustment: str, draft: AgentActionDraft) -> str:
    sanitized = {
        "kind": draft.kind,
        "title": draft.title,
        "summary": draft.summary,
        "payload": _sanitize_draft_payload(draft),
    }
    return "\n\n".join(
        (
            "The user clicked Adjust on the current pending draft.",
            "Revise that draft according to the adjustment below.",
            "Do not apply, save, or execute anything. Return compact JSON with one revised draft, or draft null if the adjustment is ambiguous.",
            "Keep the same action kind unless the user explicitly asks for a different action.",
            "If the draft involves masked secrets, do not invent or echo secret values.",
            "User adjustment:",
            adjustment,
            "Current pending draft:",
            json.dumps(sanitized, ensure_ascii=False, indent=2),
        )
    )


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


def _compound_raw_actions(payload: Mapping[str, object]) -> object:
    return agent_compound_raw_actions(payload)


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
