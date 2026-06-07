"""``agent.*`` bridge handlers for the experimental Agent Lab surface."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from secrets import token_hex
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
    TASK_KIND_BY_DRAFT_KIND,
    all_start_draft_completeness,
    assess_start_draft,
)
from transoria.agent.project_store import AgentProjectStore
from transoria.agent.schemas import (
    MODEL_SLOTS,
    PROMPT_SLOTS,
    AgentActiveTask,
    AgentActionDraft,
    AgentConversation,
    AgentMessage,
    AgentRecipe,
    AgentWorkspaceState,
)
from transoria.bridge.errors import BridgeError
from transoria.bridge.handlers._utils import expect_string
from transoria.bridge.router import BridgeRouter
from transoria.bridge.task_service import TaskService
from transoria.domain import TaskKind, TaskStatus
from transoria.llm.client import ChatRequest, ChatResponse, LlmClient, LlmRequestError
from transoria.llm.config import ModelConfig, ProviderFormat, ThinkingLevel
from transoria.model_profiles import ModelProfileStore
from transoria.prompts import (
    DEFAULT_GLOSSARY_PRESET_ID,
    DEFAULT_GLOSSARY_REVIEW_PRESET_ID,
    DEFAULT_TRANSLATION_PRESET_ID,
    PromptKind,
    PromptPreset,
    PromptPresetStore,
)
from transoria.runtime.cache import TaskNotFoundError
from transoria.settings import SettingsStore
from transoria.workflows.agent.task_starts import start_agent_task

LlmClientFactory = Callable[[], LlmClient]

_PROMPT_KIND_BY_SLOT = {
    "translation": PromptKind.TRANSLATION,
    "term_extract": PromptKind.GLOSSARY,
    "term_review": PromptKind.GLOSSARY_REVIEW,
}

_DEFAULT_PROMPT_ID_BY_SLOT = {
    "translation": DEFAULT_TRANSLATION_PRESET_ID,
    "term_extract": DEFAULT_GLOSSARY_PRESET_ID,
    "term_review": DEFAULT_GLOSSARY_REVIEW_PRESET_ID,
}

_PROMPT_KIND_ALIASES = {
    "translation": PromptKind.TRANSLATION,
    "translation_prompt": PromptKind.TRANSLATION,
    "translate_prompt": PromptKind.TRANSLATION,
    "translate": PromptKind.TRANSLATION,
    "翻译": PromptKind.TRANSLATION,
    "翻译prompt": PromptKind.TRANSLATION,
    "翻译提示词": PromptKind.TRANSLATION,
    "翻译预设": PromptKind.TRANSLATION,
    "glossary": PromptKind.GLOSSARY,
    "term_extract": PromptKind.GLOSSARY,
    "term_extraction": PromptKind.GLOSSARY,
    "glossary_extraction": PromptKind.GLOSSARY,
    "glossary_prompt": PromptKind.GLOSSARY,
    "term_extract_prompt": PromptKind.GLOSSARY,
    "术语": PromptKind.GLOSSARY,
    "术语提取": PromptKind.GLOSSARY,
    "术语提取prompt": PromptKind.GLOSSARY,
    "术语提取提示词": PromptKind.GLOSSARY,
    "术语提取预设": PromptKind.GLOSSARY,
    "glossary_review": PromptKind.GLOSSARY_REVIEW,
    "glossary_review_prompt": PromptKind.GLOSSARY_REVIEW,
    "term_review": PromptKind.GLOSSARY_REVIEW,
    "term_review_prompt": PromptKind.GLOSSARY_REVIEW,
    "term_audit": PromptKind.GLOSSARY_REVIEW,
    "术语审核": PromptKind.GLOSSARY_REVIEW,
    "术语审查": PromptKind.GLOSSARY_REVIEW,
    "术语审核prompt": PromptKind.GLOSSARY_REVIEW,
    "术语审查prompt": PromptKind.GLOSSARY_REVIEW,
    "术语审核提示词": PromptKind.GLOSSARY_REVIEW,
    "术语审查提示词": PromptKind.GLOSSARY_REVIEW,
    "术语审核预设": PromptKind.GLOSSARY_REVIEW,
    "术语审查预设": PromptKind.GLOSSARY_REVIEW,
}

_MAX_TITLE_LENGTH = 120
_MAX_CONTEXT_MESSAGES = 20
_MAX_RECIPE_NAME_LENGTH = 120
_MAX_RECIPE_DESCRIPTION_LENGTH = 400
_MAX_RECIPES = 50
_MAX_COMPOUND_ACTIONS = 8
_MAX_NOVEL_BACKGROUND_LENGTH = 6000
_AGENT_TASK_KINDS: tuple[str, ...] = (
    "translation",
    "glossary",
    "glossary_review",
)


@dataclass(frozen=True)
class _AgentActionSpec:
    kind: str
    mutates: bool
    requires_confirmation: bool
    starts_task: bool
    validate: Callable[..., None]
    apply: Callable[..., tuple[AgentWorkspaceState, dict[str, object]]]


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
            "task": _task_header_or_none(task_service, active.kind, active.task_id),
        }

    def list_recent_task_summaries(payload: Mapping[str, object]) -> dict[str, object]:
        kind = _optional_agent_string(payload, "kind")
        limit = _optional_agent_limit(payload, default=5)
        if kind is not None:
            _validate_agent_task_kind(kind)
            return {
                "kind": kind,
                "tasks": _recent_task_summaries(
                    task_service,
                    kind=kind,
                    limit=limit,
                ),
            }
        return {
            "tasks_by_kind": {
                task_kind: _recent_task_summaries(
                    task_service,
                    kind=task_kind,
                    limit=limit,
                )
                for task_kind in _AGENT_TASK_KINDS
            }
        }

    def get_artifact_availability(payload: Mapping[str, object]) -> dict[str, object]:
        kind = expect_string(payload, "kind").strip()
        _validate_agent_task_kind(kind)
        task_id = expect_string(payload, "task_id").strip()
        if not task_id:
            raise BridgeError.invalid_argument(
                "task_id must not be empty.",
                field="task_id",
            )
        return _artifact_availability(task_service, kind=kind, task_id=task_id)

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
                _applied_draft_message(
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
    if not workflow_model_id:
        return (
            "请选择一个工作模型。当前聊天已经可记录，但还不会调用模型生成配置草案。",
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
    conversation_context = _recent_conversation_context(state)
    if not user_message.startswith("The user clicked Adjust on the current pending draft."):
        status_state = _task_status_context(task_service)
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
    inventory = _inventory(profile_store, cache_root)
    current_state = _llm_context(
        state,
        settings_store=settings_store,
        task_service=task_service,
    )
    direct = None
    if not user_message.startswith("The user clicked Adjust on the current pending draft."):
        direct = _direct_translation_response(
            user_message=user_message,
            conversation_context=conversation_context,
            state=state,
            current_state=current_state,
            cache_root=cache_root,
        )
        if direct is None:
            direct = _direct_glossary_review_response(
                user_message=user_message,
                conversation_context=conversation_context,
                state=state,
                current_state=current_state,
                cache_root=cache_root,
            )
        if direct is None:
            direct = _direct_glossary_extraction_response(
                user_message=user_message,
                conversation_context=conversation_context,
                state=state,
                current_state=current_state,
                cache_root=cache_root,
            )
        if direct is None:
            direct = _direct_compound_config_response(
                user_message=user_message,
                current_state=current_state,
                profile_store=profile_store,
                cache_root=cache_root,
            )
        if direct is None:
            direct = _direct_prompt_quality_response(
                user_message=user_message,
                state=state,
                cache_root=cache_root,
            )
        if direct is None:
            direct = _direct_prompt_preset_response(user_message=user_message)
        if direct is None:
            direct = _direct_vague_model_profile_guidance_response(
                user_message=user_message,
                profile_store=profile_store,
            )
        if direct is None:
            direct = _direct_model_profile_copy_response(
                user_message=user_message,
                conversation_context=conversation_context,
                profile_store=profile_store,
            )
        if direct is None:
            direct = _direct_model_upgrade_response(
                user_message=user_message,
                state=state,
                profile_store=profile_store,
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
    text = "\n".join((user_message, reply, draft.title, draft.summary))
    if draft.kind == "create_prompt_preset":
        return _salvage_create_prompt_draft(draft, text=text)
    if draft.kind == "create_recipe":
        return _salvage_create_recipe_draft(
            draft,
            text=text,
            current_state=current_state,
        )
    if draft.kind == "compound_config_update":
        return _salvage_compound_draft(
            draft,
            text=text,
            current_state=current_state,
            profile_store=profile_store,
            cache_root=cache_root,
        )
    return None


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
    if not _looks_like_direct_prompt_preset_request(user_message):
        return None
    name = _extract_prompt_create_name(user_message)
    if not name:
        return None
    if not _has_prompt_creation_requirements(user_message):
        return None
    try:
        prompt_kind = _coerce_prompt_kind(None, fallback_text=user_message)
    except BridgeError:
        return (
            "我理解你想创建 Prompt 预设，但还不清楚它属于哪个阶段。请说明是翻译、术语提取还是术语审查 Prompt。",
            None,
        )
    system_prompt = _extract_prompt_body_from_request(user_message)
    if not system_prompt:
        system_prompt = _build_prompt_body_from_request(
            prompt_kind=prompt_kind,
            user_request=user_message,
        )
    title = f"创建{name}"
    payload = {
        "kind": prompt_kind.value,
        "name": name,
        "description": f"由 Agent Lab 根据聊天请求创建的{name}。",
        "system_prompt": system_prompt,
        "enabled": True,
    }
    draft = AgentActionDraft.create(
        kind="create_prompt_preset",
        title=title,
        summary=f"创建一套 {_prompt_kind_label(prompt_kind)} Prompt 预设：{name}。",
        payload=payload,
    )
    return (
        (
            f"我已准备好创建 Prompt 预设「{name}」的草案。"
            "请检查内容，确认后才会写入 Prompt 配置。"
        ),
        draft,
    )


def _direct_prompt_quality_response(
    *,
    user_message: str,
    state: AgentWorkspaceState,
    cache_root: Path,
) -> tuple[str, AgentActionDraft | None] | None:
    if not _looks_like_prompt_quality_request(user_message):
        return None
    prompt_kind = _quality_prompt_kind(user_message)
    issues = _quality_issue_markers(user_message)
    if not issues:
        return (
            (
                "我可以帮你优化 Prompt，但还缺少可操作的问题描述。"
                "请贴一小段原文/译文，或说明具体问题类型，例如源文残留、"
                "人名不一致、术语漂移、文风太直译、漏翻或翻译腔。"
            ),
            None,
        )

    source = _selected_prompt_for_kind(
        prompt_kind,
        state=state,
        cache_root=cache_root,
    )
    addendum = _quality_prompt_addendum(
        issues=issues,
        user_message=user_message,
    )
    label = _prompt_kind_label(prompt_kind)
    if source is not None and not source.is_system:
        system_prompt = f"{source.system_prompt.rstrip()}\n\n{addendum}"
        draft = AgentActionDraft.create(
            kind="update_prompt_preset",
            title=f"优化{source.name}",
            summary=f"在现有 {label} Prompt「{source.name}」中追加本次质量修正规则。",
            payload={
                "id": source.id,
                "patch": {
                    "system_prompt": system_prompt,
                    "description": source.description
                    or f"由 Agent Lab 根据质量反馈优化的{label} Prompt。",
                },
            },
        )
        return (
            (
                f"我会在当前 {label} Prompt「{source.name}」里追加质量修正规则，"
                "解决你指出的问题。请先检查草案，确认后才会写入配置。"
            ),
            draft,
        )

    base_prompt = (
        source.system_prompt
        if source is not None
        else _build_prompt_body_from_request(
            prompt_kind=prompt_kind,
            user_request=user_message,
        )
    )
    name = _quality_prompt_name(user_message, prompt_kind, source)
    draft = AgentActionDraft.create(
        kind="create_prompt_preset",
        title=f"创建{name}",
        summary=f"基于当前 {label} Prompt 创建可编辑副本，并加入本次质量修正规则。",
        payload={
            "kind": prompt_kind.value,
            "name": name,
            "description": f"由 Agent Lab 根据质量反馈创建的{label} Prompt。",
            "system_prompt": f"{base_prompt.rstrip()}\n\n{addendum}",
            "enabled": True,
        },
    )
    return (
        (
            f"当前选中的 {label} Prompt 是系统预设或尚未选择可编辑预设，"
            f"我会创建一套新的「{name}」供你确认保存。"
        ),
        draft,
    )


def _has_prompt_creation_requirements(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "要求",
            "规则",
            "内容",
            "正文",
            "强调",
            "用于",
            "适合",
            "风格",
            "规范",
            "保留",
            "禁止",
        )
    )


def _looks_like_direct_prompt_preset_request(text: str) -> bool:
    if _looks_like_translation_task_request(text):
        return False
    if _looks_like_glossary_review_request(text):
        return False
    if _looks_like_glossary_extraction_request(text):
        return False
    normalized = text.lower()
    if "prompt" not in normalized and "提示词" not in text:
        return False
    return any(marker in text for marker in ("创建", "新建", "新增", "保存", "配置", "做一套", "准备一套"))


def _extract_prompt_create_name(text: str) -> str:
    patterns = (
        r"(?:名字|名称|命名为|名为|叫做|叫)\s*[「『“\"'](.+?)[」』”\"']",
        r"(?:名字|名称|命名为|名为|叫做|叫)\s*[:：]?\s*([^\n，。,.]+)",
        r"(?:创建|新建|新增|配置|保存|准备)(?:一套|一个|新的)?\s*[「『“\"'](.+?)[」』”\"']\s*(?:的)?(?:Prompt|prompt|提示词)",
        r"(?:创建|新建|新增|配置|保存|准备)(?:一套|一个|新的)?\s*(.+?)(?:Prompt|prompt|提示词)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        value = match.group(1).strip(" \t\r\n「」『』“”\"'：:")
        value = re.sub(r"^(?:翻译|术语提取|术语审核|术语审查)\s*", "", value).strip()
        if value:
            return value[:_MAX_RECIPE_NAME_LENGTH]
    return ""


def _extract_prompt_body_from_request(text: str) -> str:
    marker_pattern = (
        r"(?:内容|正文|规则|要求|prompt|Prompt|提示词)\s*(?:如下|是|为)?\s*[:：]\s*(.+)"
    )
    match = re.search(marker_pattern, text, flags=re.DOTALL)
    if match:
        value = match.group(1).strip()
        if value:
            return value
    return ""


def _prompt_kind_label(kind: PromptKind) -> str:
    return {
        PromptKind.TRANSLATION: "翻译",
        PromptKind.GLOSSARY: "术语提取",
        PromptKind.GLOSSARY_REVIEW: "术语审查",
    }[kind]


def _looks_like_prompt_quality_request(text: str) -> bool:
    normalized = text.lower()
    if "prompt" not in normalized and "提示词" not in text:
        return False
    if not any(marker in text for marker in ("改", "修改", "优化", "调整", "重写", "重新设计")):
        return False
    return any(
        marker in text
        for marker in (
            "效果不好",
            "质量不好",
            "不好",
            "问题",
            "翻译腔",
            "不自然",
            "源文残留",
            "原文残留",
            "漏翻",
            "错译",
            "直译",
            "文风",
            "人名",
            "术语",
            "一致",
            "低置信",
            "质量",
        )
    )


def _quality_prompt_kind(text: str) -> PromptKind:
    try:
        return _coerce_prompt_kind(None, fallback_text=text)
    except BridgeError:
        return PromptKind.TRANSLATION


def _quality_issue_markers(text: str) -> list[str]:
    candidates = (
        ("源文残留", ("源文残留", "原文残留", "韩文残留", "日文残留", "英文残留")),
        ("人名不一致", ("人名不一致", "名字不一致", "称呼不一致", "人名")),
        ("术语不一致", ("术语不一致", "术语漂移", "术语")),
        ("文风不自然", ("文风不自然", "不自然", "翻译腔", "太直译", "直译", "文风")),
        ("漏翻", ("漏翻", "缺句", "少翻")),
        ("错译", ("错译", "误译", "理解错")),
        ("低置信度", ("低置信", "不确定")),
        ("长度或分段异常", ("长度", "分段", "行数")),
    )
    issues: list[str] = []
    for label, markers in candidates:
        if any(marker in text for marker in markers):
            issues.append(label)
    if issues == ["术语不一致"] and "术语" in text and "翻译" not in text:
        return []
    return issues


def _selected_prompt_for_kind(
    kind: PromptKind,
    *,
    state: AgentWorkspaceState,
    cache_root: Path,
) -> PromptPreset | None:
    slot = _prompt_slot_for_kind(kind)
    prompt_id = state.stage_prompt_ids.get(slot) or _DEFAULT_PROMPT_ID_BY_SLOT.get(slot)
    if not prompt_id:
        return None
    for preset in _prompt_store_for(cache_root, kind).load():
        if preset.id == prompt_id:
            return preset
    return None


def _prompt_slot_for_kind(kind: PromptKind) -> str:
    for slot, prompt_kind in _PROMPT_KIND_BY_SLOT.items():
        if prompt_kind == kind:
            return slot
    raise BridgeError.invalid_argument(
        f"unsupported prompt kind: {kind.value}",
        field="kind",
    )


def _quality_prompt_addendum(*, issues: Sequence[str], user_message: str) -> str:
    issue_text = "、".join(dict.fromkeys(issues))
    return "\n".join(
        (
            "本次质量优化要求：",
            f"- 重点修正：{issue_text}。",
            "- 处理前先识别上下文、人名、称呼、术语和叙事视角，避免只逐句直译。",
            "- 输出时优先保持原文信息完整和分段稳定，不新增原文不存在的情节、心理解释或评价。",
            "- 对人名、称呼、组织、作品内专有名词保持一致；不确定时保留可校对的稳定译名。",
            "- 主动规避源语言残留、漏翻、重复漂移和明显翻译腔。",
            f"- 用户反馈原文：{user_message.strip()}",
        )
    )


def _quality_prompt_name(
    text: str,
    kind: PromptKind,
    source: PromptPreset | None,
) -> str:
    explicit = _extract_prompt_create_name(text)
    if explicit:
        return explicit
    base = source.name if source is not None else f"{_prompt_kind_label(kind)} Prompt"
    name = f"{base} - 质量优化"
    return name[:_MAX_RECIPE_NAME_LENGTH]


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
            "1. 说“按照某个已有模型复制一个”，并告诉我要改成的 provider model_id；\n"
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
        )
        if dumb_start is not None:
            return dumb_start
        missing = "、".join(completeness.missing)
        return (
            f"我理解你想提取术语，但当前术语提取任务配置还不完整：{missing}。请先补齐对应模型、Prompt 或语言设置。",
            None,
        )
    note = "未提供 output 目录，草案会默认输出到 input 目录。" if output_defaulted else ""
    draft = AgentActionDraft.create(
        kind="start_glossary_task",
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
        )
        if dumb_start is not None:
            return dumb_start
        missing = "、".join(completeness.missing)
        return (
            f"我理解你想审查术语，但当前术语审查任务配置还不完整：{missing}。请先补齐对应模型、Prompt 或语言设置。",
            None,
        )
    draft = AgentActionDraft.create(
        kind="start_glossary_review_task",
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
    draft = AgentActionDraft.create(
        kind="start_translation_task",
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
) -> tuple[str, AgentActionDraft] | None:
    stage = {
        "start_glossary_task": "term_extract",
        "start_glossary_review_task": "term_review",
        "start_translation_task": "translation",
    }.get(draft_kind)
    if stage is None:
        return None

    missing = set(completeness_missing)
    model_key = f"stage_model_ids.{stage}"
    prompt_key = f"stage_prompt_ids.{stage}"
    if not missing or not missing.issubset({model_key, prompt_key}):
        return None

    workspace_patch: dict[str, object] = {}
    if model_key in missing:
        if not state.workflow_model_id:
            return None
        workspace_patch["stage_model_ids"] = {stage: state.workflow_model_id}
    if prompt_key in missing:
        prompt_id = _DEFAULT_PROMPT_ID_BY_SLOT.get(stage)
        if not prompt_id:
            return None
        prompt_kind = _PROMPT_KIND_BY_SLOT[stage]
        presets = _prompt_store_for(cache_root, prompt_kind).load()
        if not any(preset.id == prompt_id for preset in presets):
            return None
        workspace_patch["stage_prompt_ids"] = {stage: prompt_id}
    if not workspace_patch:
        return None

    actions: list[dict[str, object]] = [
        {
            "kind": "update_workspace",
            "title": "补齐当前流程阶段配置",
            "summary": "使用当前工作模型和系统默认 Prompt 补齐本次任务需要的阶段配置。",
            "payload": workspace_patch,
        },
        {
            "kind": draft_kind,
            "title": start_title,
            "summary": start_summary,
            "payload": dict(payload),
        },
    ]
    draft = AgentActionDraft.create(
        kind="compound_config_update",
        title=title,
        summary="先补齐缺失的阶段模型/Prompt，再启动对应任务。所有动作都会在你点击「应用」后按顺序执行。",
        payload={"actions": actions},
    )
    return reply, draft


def _direct_stage_status_response(
    *,
    user_message: str,
    current_state: Mapping[str, object],
    task_service: TaskService,
) -> tuple[str, None] | None:
    direct = _direct_glossary_stage_status_response(
        user_message=user_message,
        current_state=current_state,
        task_service=task_service,
    )
    if direct is not None:
        return direct
    return _direct_translation_status_response(
        user_message=user_message,
        current_state=current_state,
        task_service=task_service,
    )


def _direct_glossary_stage_status_response(
    *,
    user_message: str,
    current_state: Mapping[str, object],
    task_service: TaskService,
) -> tuple[str, None] | None:
    requested_kind = _requested_glossary_status_kind(user_message, current_state)
    if requested_kind is None:
        return None

    if requested_kind == "glossary_review":
        task_id = _extract_glossary_review_task_id(user_message) or _latest_task_id(
            current_state,
            kind="glossary_review",
        )
        label = "术语审查"
        missing = "我还没有找到最近的术语审查任务记录。请提供 glossary-review- 开头的任务 ID，或先启动术语审查。"
        task_kind = TaskKind.GLOSSARY_REVIEW
    else:
        task_id = _extract_glossary_task_id(user_message) or _latest_task_id(
            current_state,
            kind="glossary",
        )
        label = "术语提取"
        missing = "我还没有找到最近的术语提取任务记录。请提供 glossary- 开头的任务 ID，或先启动术语提取。"
        task_kind = TaskKind.GLOSSARY
    if not task_id:
        return missing, None

    try:
        record = task_service.cache.load_record(task_id)
    except (TaskNotFoundError, ValueError, OSError):
        return (f"我没有找到{label}任务 {task_id}。请确认任务 ID 是否正确。", None)
    if record.kind is not task_kind:
        return (f"{task_id} 不是{label}任务，不能按{label}进度读取。", None)
    status = record.status.value

    artifacts: Mapping[str, object] = {}
    try:
        raw_artifacts = task_service.read_artifacts(kind=requested_kind, task_id=task_id)
        if isinstance(raw_artifacts, Mapping):
            artifacts = raw_artifacts
    except BridgeError:
        artifacts = {}

    if requested_kind == "glossary_review":
        return _format_glossary_review_status_response(
            task_id=task_id,
            status=status,
            artifacts=artifacts,
        ), None
    return _format_glossary_status_response(
        task_id=task_id,
        status=status,
        artifacts=artifacts,
    ), None


def _direct_translation_status_response(
    *,
    user_message: str,
    current_state: Mapping[str, object],
    task_service: TaskService,
) -> tuple[str, None] | None:
    if not _looks_like_translation_status_query(user_message):
        return None
    task_id = _extract_translation_task_id(user_message) or _latest_translation_task_id(
        current_state
    )
    if not task_id:
        return (
            "我还没有找到最近的翻译任务记录。请提供 translation- 开头的任务 ID，或先启动一次翻译任务。",
            None,
        )
    try:
        record = task_service.cache.load_record(task_id)
    except (TaskNotFoundError, ValueError, OSError):
        return (f"我没有找到翻译任务 {task_id}。请确认任务 ID 是否正确。", None)
    if record.kind is not TaskKind.TRANSLATION:
        return (f"{task_id} 不是翻译任务，不能按翻译进度读取。", None)

    artifacts: Mapping[str, object] = {}
    try:
        raw_artifacts = task_service.read_artifacts(kind="translation", task_id=task_id)
        if isinstance(raw_artifacts, Mapping):
            artifacts = raw_artifacts
    except BridgeError:
        artifacts = {}
    statistics = _read_translation_statistics(artifacts)
    completed_segments = _coerce_int(
        statistics.get("completed_segments")
        if statistics
        else artifacts.get("completed_segments")
    )
    total_segments = _coerce_int(
        statistics.get("total_segments") if statistics else artifacts.get("total_segments")
    )
    failed_subtasks = _coerce_int(statistics.get("failed_subtasks") if statistics else None)
    low_confidence_raw = statistics.get("low_confidence_segments") if statistics else None
    low_confidence_count = _low_confidence_count(low_confidence_raw)
    low_confidence_examples = _low_confidence_examples(low_confidence_raw)
    translated_files = _string_list(
        artifacts.get("translated_files")
        or artifacts.get("output_files")
        or (statistics.get("translated_outputs") if statistics else None)
    )

    status = record.status.value
    lines = [f"最近的翻译任务 {task_id} 当前状态：{_task_status_label(status)}。"]
    if completed_segments is not None and total_segments is not None:
        lines.append(f"进度：{completed_segments}/{total_segments} 段。")
    elif completed_segments is not None:
        lines.append(f"已完成段数：{completed_segments}。")
    if failed_subtasks is not None:
        lines.append(f"失败子任务：{failed_subtasks}。")
    if low_confidence_count is not None:
        detail = f"低置信：{low_confidence_count} 条"
        if low_confidence_examples:
            detail += f"（示例：{', '.join(low_confidence_examples)}）"
        lines.append(detail + "。")
    else:
        lines.append("我没有在该任务的统计文件里读到低置信数量。")
    if translated_files:
        lines.append(f"输出文件：{translated_files[0]}")
    if status == TaskStatus.COMPLETED.value:
        lines.append(
            "下一步建议进入「翻译 > 校对」页面，选择这个任务，优先处理低置信、原文残留和疑似重复问题。"
        )
    elif status in {TaskStatus.RUNNING.value, TaskStatus.PENDING.value}:
        lines.append("你可以在翻译 dashboard 查看实时进度；任务结束前 Agent Lab 不会再启动其他任务。")
    else:
        lines.append("如果需要继续处理，请先在对应 dashboard 查看错误或产物状态。")
    return "\n".join(lines), None


def _requested_glossary_status_kind(
    text: str,
    current_state: Mapping[str, object],
) -> str | None:
    normalized = text.lower()
    if not _looks_like_stage_status_query(normalized):
        return None
    if _extract_glossary_review_task_id(text):
        return "glossary_review"
    if _extract_glossary_task_id(text):
        return "glossary"
    has_review_context = any(
        marker in normalized
        for marker in (
            "术语审查",
            "术语审核",
            "术语复审",
            "审核术语",
            "审查术语",
            "复审术语",
            "glossary review",
            "review glossary",
            "glossary-review",
        )
    )
    has_extract_context = any(
        marker in normalized
        for marker in (
            "术语提取",
            "术语抽取",
            "提取术语",
            "抽取术语",
            "处理术语",
            "整理术语",
            "glossary extraction",
            "glossary task",
        )
    )
    if has_review_context:
        return "glossary_review"
    if has_extract_context:
        return "glossary"
    if "术语" not in normalized and "glossary" not in normalized:
        return None
    return _latest_status_kind(current_state, kinds=("glossary_review", "glossary"))


def _looks_like_stage_status_query(normalized: str) -> bool:
    return any(
        marker in normalized
        for marker in (
            "完成了吗",
            "完成了没",
            "做完了吗",
            "结束了吗",
            "结束了没",
            "是否完成",
            "是否结束",
            "状态",
            "进度",
            "下一步",
            "现在到哪",
            "进行到哪",
            "做到哪",
            "产物",
            "结果",
            "报告",
            "哪里看",
            "去哪个页面",
            "去哪",
            "status",
            "progress",
            "next",
            "done?",
            "finished?",
        )
    )


def _latest_status_kind(
    current_state: Mapping[str, object],
    *,
    kinds: tuple[str, ...],
) -> str | None:
    recent_by_kind = current_state.get("recent_task_summaries")
    if not isinstance(recent_by_kind, Mapping):
        return None
    best_kind: str | None = None
    best_updated = ""
    for kind in kinds:
        tasks = recent_by_kind.get(kind)
        if not isinstance(tasks, list) or not tasks:
            continue
        item = tasks[0]
        if not isinstance(item, Mapping):
            continue
        task_id = str(item.get("id") or "").strip()
        if not task_id:
            continue
        updated_at = str(item.get("updated_at") or item.get("created_at") or "")
        if best_kind is None or updated_at >= best_updated:
            best_kind = kind
            best_updated = updated_at
    return best_kind


def _latest_task_id(current_state: Mapping[str, object], *, kind: str) -> str | None:
    recent_by_kind = current_state.get("recent_task_summaries")
    if not isinstance(recent_by_kind, Mapping):
        return None
    tasks = recent_by_kind.get(kind)
    if not isinstance(tasks, list):
        return None
    for item in tasks:
        if not isinstance(item, Mapping):
            continue
        task_id = str(item.get("id") or "").strip()
        if task_id:
            return task_id
    return None


def _format_glossary_status_response(
    *,
    task_id: str,
    status: str,
    artifacts: Mapping[str, object],
) -> str:
    statistics = _read_json_artifact(artifacts, "statistics_json_path")
    candidate_count = _coerce_int(statistics.get("candidate_count"))
    final_entry_count = _coerce_int(statistics.get("final_entry_count"))
    processed_files = _string_list(statistics.get("processed_files"))
    per_novel = artifacts.get("per_novel_artifacts")
    artifact_count = len(per_novel) if isinstance(per_novel, list) else None
    combined = artifacts.get("combined_artifact")
    combined_xlsx = None
    if isinstance(combined, Mapping):
        raw_path = combined.get("xlsx_path")
        if isinstance(raw_path, str) and raw_path.strip():
            combined_xlsx = raw_path.strip()

    lines = [f"最近的术语提取任务 {task_id} 当前状态：{_task_status_label(status)}。"]
    if processed_files:
        lines.append(f"处理文件：{len(processed_files)} 个。")
    if candidate_count is not None:
        lines.append(f"候选术语：{candidate_count} 条。")
    if final_entry_count is not None:
        lines.append(f"最终术语：{final_entry_count} 条。")
    elif artifact_count is not None:
        lines.append(f"已生成术语产物：{artifact_count} 份。")
    if combined_xlsx:
        lines.append(f"合并术语表：{combined_xlsx}")
    if status == TaskStatus.COMPLETED.value:
        lines.append(
            "下一步建议启动「术语审查」任务。我可以基于这个 glossary task ID 生成术语审查草案，仍然需要你确认后才会执行。"
        )
    elif status in {TaskStatus.RUNNING.value, TaskStatus.PENDING.value}:
        lines.append("你可以在术语提取 dashboard 查看实时进度；任务结束前 Agent Lab 不会再启动其他任务。")
    else:
        lines.append("如果需要继续处理，请先在术语提取 dashboard 查看错误和产物状态。")
    return "\n".join(lines)


def _format_glossary_review_status_response(
    *,
    task_id: str,
    status: str,
    artifacts: Mapping[str, object],
) -> str:
    changed_count = _coerce_int(artifacts.get("changed_count"))
    output_path = artifacts.get("output_path")
    report_path = artifacts.get("report_path")
    lines = [f"最近的术语审查任务 {task_id} 当前状态：{_task_status_label(status)}。"]
    if changed_count is not None:
        lines.append(f"模型审查修改：{changed_count} 条。")
    if isinstance(output_path, str) and output_path.strip():
        lines.append(f"确认术语表：{output_path.strip()}")
    if isinstance(report_path, str) and report_path.strip():
        lines.append(f"审查报告：{report_path.strip()}")
    if status == TaskStatus.COMPLETED.value:
        lines.append(
            "下一步建议进入「术语审查」页面的数据表校对区，人工确认术语表。确认无误后，可以在聊天里说“术语表已确认，用它翻译”，我再生成翻译任务草案。"
        )
    elif status in {TaskStatus.RUNNING.value, TaskStatus.PENDING.value}:
        lines.append("你可以在术语审查 dashboard 查看实时进度；任务结束前 Agent Lab 不会再启动其他任务。")
    else:
        lines.append("如果需要继续处理，请先在术语审查 dashboard 查看错误、报告或最终表状态。")
    return "\n".join(lines)


def _looks_like_translation_status_query(text: str) -> bool:
    normalized = text.lower()
    has_translation_context = (
        "翻译" in normalized
        or "translation-" in normalized
        or "低置信" in normalized
        or "原文残留" in normalized
        or "校对" in normalized
        or "proofread" in normalized
    )
    if not has_translation_context:
        return False
    return any(
        marker in normalized
        for marker in (
            "完成",
            "结束",
            "状态",
            "进度",
            "低置信",
            "原文残留",
            "下一步",
            "哪里校对",
            "去哪校对",
            "校对页",
            "proofread",
            "status",
            "progress",
        )
    )


def _extract_translation_task_id(text: str) -> str | None:
    match = re.search(r"\b(translation-[a-zA-Z0-9-]+)\b", text)
    if match is None:
        return None
    return match.group(1).strip()


def _latest_translation_task_id(current_state: Mapping[str, object]) -> str | None:
    recent_by_kind = current_state.get("recent_task_summaries")
    if not isinstance(recent_by_kind, Mapping):
        return None
    translation_tasks = recent_by_kind.get("translation")
    if not isinstance(translation_tasks, list):
        return None
    for item in translation_tasks:
        if not isinstance(item, Mapping):
            continue
        task_id = str(item.get("id") or "").strip()
        if task_id:
            return task_id
    return None


def _read_translation_statistics(
    artifacts: Mapping[str, object],
) -> Mapping[str, object]:
    return _read_json_artifact(artifacts, "statistics_json_path")


def _read_json_artifact(
    artifacts: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    statistics_path = artifacts.get(key)
    if not isinstance(statistics_path, str) or not statistics_path.strip():
        return {}
    path = Path(statistics_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, Mapping) else {}


def _low_confidence_count(value: object) -> int | None:
    if isinstance(value, list):
        return len(value)
    return _coerce_int(value)


def _low_confidence_examples(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    examples: list[str] = []
    for item in value[:3]:
        if isinstance(item, Mapping):
            segment_id = str(item.get("segment_id") or "").strip()
            if segment_id:
                examples.append(segment_id)
        elif isinstance(item, str) and item.strip():
            examples.append(item.strip())
    return examples


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _looks_like_translation_task_request(text: str) -> bool:
    normalized = text.lower()
    if re.search(
        r"先\s*(?:提取|抽取|处理|整理)\s*术语(?:表)?",
        normalized,
    ):
        return False
    strong_translation_markers = (
        "开始翻译",
        "启动翻译",
        "执行翻译",
        "进行翻译",
        "翻译任务",
        "翻译小说",
        "run translation",
        "start translation",
        "translate novel",
        "translate book",
    )
    has_strong_translation_intent = any(
        marker in normalized for marker in strong_translation_markers
    )
    has_task_input_context = bool(_extract_absolute_path_candidates(text)) or any(
        marker in normalized
        for marker in (
            "input",
            "output",
            "输入目录",
            "输出目录",
            "输入路径",
            "输出路径",
            "epub",
            ".epub",
            "txt",
            ".txt",
            "目录",
            "路径",
        )
    )
    has_translation_task_context = (
        "翻译" in normalized
        and has_task_input_context
        and "模型配置" not in normalized
    )
    if (
        _looks_like_glossary_review_request(text)
        and not has_strong_translation_intent
        and not has_translation_task_context
    ):
        return False
    if any(marker in normalized for marker in ("prompt", "提示词", "预设")) and not any(
        marker in normalized
        for marker in strong_translation_markers
    ):
        return False
    if (
        _looks_like_glossary_extraction_request(text)
        and not has_strong_translation_intent
        and not has_translation_task_context
    ):
        return False
    return has_strong_translation_intent or has_translation_task_context


def _looks_like_translation_after_review_followup_request(
    text: str,
    current_state: Mapping[str, object],
) -> bool:
    if not _latest_completed_glossary_review_task_id(current_state):
        return False
    normalized = text.lower()
    if any(marker in normalized for marker in ("prompt", "提示词", "预设", "模型配置")):
        return False
    if any(
        marker in normalized
        for marker in (
            "完成了吗",
            "完成了没",
            "做完了吗",
            "结束了吗",
            "结束了没",
            "是否完成",
            "是否结束",
            "状态",
            "进度",
            "哪里",
            "哪个页面",
            "我该做什么",
            "status",
            "progress",
            "finished?",
            "done?",
        )
    ):
        return False
    has_translation_or_next_intent = any(
        marker in normalized
        for marker in (
            "翻译",
            "继续",
            "下一步",
            "下一阶段",
            "后续",
            "接着",
            "next",
            "continue",
            "translate",
        )
    )
    if not has_translation_or_next_intent:
        return False
    return any(
        marker in normalized
        for marker in (
            "术语表已确认",
            "术语表已经确认",
            "术语表确认",
            "术语表确认好了",
            "术语表确认无误",
            "术语表没问题",
            "术语确认",
            "术语已经确认",
            "术语确认好了",
            "术语没问题",
            "表没问题",
            "确认好的术语",
            "审查好的术语",
            "审核好的术语",
            "校对好的术语",
            "reviewed glossary",
            "confirmed glossary",
        )
    )


def _looks_like_glossary_review_request(text: str) -> bool:
    normalized = text.lower()
    prompt_like = any(marker in normalized for marker in ("prompt", "提示词", "预设"))
    task_like = any(
        marker in normalized
        for marker in (
            "启动",
            "开始",
            "继续",
            "执行",
            "运行",
            "跑",
            "处理",
            "刚才",
            "最新",
            "最近",
            "已提取",
            "提取结果",
            "这个术语表",
            "start",
            "run",
            "continue",
            "task",
            "workflow",
        )
    )
    has_task_id = bool(re.search(r"\bglossary-(?:review-)?[a-z0-9-]+\b", normalized))
    if prompt_like and not (has_task_id or task_like):
        return False
    has_review_marker = (
        any(
            action in normalized
            for action in (
                "审查",
                "审核",
                "校对",
                "复审",
                "审一遍",
                "审一下",
                "审一审",
                "review",
            )
        )
        and "术语" in normalized
    ) or any(
        marker in normalized
        for marker in (
            "术语审查",
            "术语审核",
            "审查术语",
            "审核术语",
            "校对术语",
            "review glossary",
            "glossary review",
        )
    )
    return has_review_marker and (has_task_id or task_like)


def _looks_like_glossary_review_followup_request(
    text: str,
    current_state: Mapping[str, object],
) -> bool:
    if not _latest_completed_glossary_task_id(current_state):
        return False
    normalized = text.lower()
    if any(marker in normalized for marker in ("prompt", "提示词", "预设", "模型配置")):
        return False
    if any(
        marker in normalized
        for marker in (
            "完成了吗",
            "完成了没",
            "做完了吗",
            "结束了吗",
            "结束了没",
            "是否完成",
            "是否结束",
            "状态",
            "进度",
            "哪里",
            "哪个页面",
            "我该做什么",
            "status",
            "progress",
            "finished?",
            "done?",
        )
    ):
        return False
    if _looks_like_translation_task_request(text):
        return False
    has_continue_intent = any(
        marker in normalized
        for marker in (
            "继续",
            "下一步",
            "下一阶段",
            "后续",
            "接着",
            "往后",
            "next",
            "continue",
        )
    )
    if not has_continue_intent:
        return False
    return any(
        marker in normalized
        for marker in (
            "术语",
            "glossary",
            "workflow",
            "流程",
            "任务",
            "刚才",
            "最新",
            "最近",
            "提取完成",
            "提取完",
            "跑完",
            "完成了",
        )
    )


def _extract_glossary_task_id(text: str) -> str | None:
    match = re.search(r"\b(glossary-(?!review-)[a-zA-Z0-9-]+)\b", text)
    if match is None:
        return None
    return match.group(1).strip()


def _extract_glossary_review_task_id(text: str) -> str | None:
    match = re.search(r"\b(glossary-review-[a-zA-Z0-9-]+)\b", text)
    if match is None:
        return None
    return match.group(1).strip()


def _latest_completed_glossary_task_id(current_state: Mapping[str, object]) -> str | None:
    recent_by_kind = current_state.get("recent_task_summaries")
    if not isinstance(recent_by_kind, Mapping):
        return None
    glossary_tasks = recent_by_kind.get("glossary")
    if not isinstance(glossary_tasks, list):
        return None
    for item in glossary_tasks:
        if not isinstance(item, Mapping):
            continue
        task_id = str(item.get("id") or "").strip()
        status = str(item.get("status") or "").strip().lower()
        if task_id and status == "completed" and bool(item.get("artifact_available")):
            return task_id
    return None


def _latest_completed_glossary_review_task_id(
    current_state: Mapping[str, object],
) -> str | None:
    recent_by_kind = current_state.get("recent_task_summaries")
    if not isinstance(recent_by_kind, Mapping):
        return None
    review_tasks = recent_by_kind.get("glossary_review")
    if not isinstance(review_tasks, list):
        return None
    for item in review_tasks:
        if not isinstance(item, Mapping):
            continue
        task_id = str(item.get("id") or "").strip()
        status = str(item.get("status") or "").strip().lower()
        if task_id and (status == "completed" or bool(item.get("artifact_available"))):
            return task_id
    return None


def _asks_for_reviewed_glossary(text: str) -> bool:
    normalized = text.lower()
    return any(
        marker in normalized
        for marker in (
            "审查好的术语",
            "审核好的术语",
            "校对好的术语",
            "确认好的术语",
            "审查后的术语",
            "审核后的术语",
            "校对后的术语",
            "reviewed glossary",
            "confirmed glossary",
        )
    )


def _looks_like_glossary_extraction_request(text: str) -> bool:
    normalized = text.lower()
    if _looks_like_glossary_review_request(text):
        return False
    if any(marker in normalized for marker in ("prompt", "提示词", "预设")) and not re.search(
        r"/|input|output|输入|输出|目录|路径",
        text,
        flags=re.IGNORECASE,
    ):
        return False
    return any(
        marker in normalized
        for marker in (
            "提取术语",
            "术语提取",
            "处理术语",
            "处理术语表",
            "傻瓜术语",
            "傻瓜式术语",
            "一键术语",
            "自动术语",
            "术语流程",
            "术语弄一下",
            "术语搞一下",
            "把术语弄一下",
            "把术语搞一下",
            "弄术语",
            "搞术语",
            "整理术语",
            "整理术语表",
            "跑术语",
            "跑术语表",
            "术语表 workflow",
            "术语 workflow",
            "完整 workflow",
            "整套 workflow",
            "完整流程",
            "整套流程",
            "跑流程",
            "处理小说",
            "处理这本小说",
            "数据表提取",
            "处理数据表",
            "关键词和术语表",
            "抽取术语",
            "extract glossary",
            "glossary extraction",
            "extract terms",
            "term extraction",
        )
    )


def _looks_like_glossary_extraction_continuation(text: str) -> bool:
    has_path = bool(_extract_absolute_path_candidates(text)) or bool(
        re.search(r"input|output|输入|输出|目录|路径", text, flags=re.IGNORECASE)
    )
    has_context = any(
        marker in text
        for marker in (
            "背景",
            "类型",
            "世界观",
            "作品关键词",
            "人物介绍",
            "作品指南",
            "小说介绍",
        )
    )
    return has_path and has_context


def _should_discard_pending_for_new_message(text: str) -> bool:
    normalized = text.lower()
    if _looks_like_translation_task_request(text):
        return True
    if _looks_like_glossary_review_request(text):
        return True
    if _looks_like_glossary_extraction_request(text):
        return True
    if _looks_like_glossary_extraction_continuation(text):
        return True
    if _looks_like_model_profile_copy_request(text):
        return True
    if _looks_like_model_profile_copy_continuation(text):
        return True
    return any(
        marker in normalized
        for marker in (
            "创建",
            "新建",
            "修改",
            "更新",
            "配置",
            "应用",
            "保存",
            "启动",
            "开始",
            "继续",
            "下一步",
            "提取",
            "处理术语",
            "create",
            "update",
            "apply",
            "save",
            "start",
        )
    )


def _extract_glossary_task_dirs(text: str) -> tuple[str, str | None, bool]:
    input_dir = _extract_labeled_path(
        text,
        ("input", "输入目录", "输入路径", "源目录", "原文目录", "小说目录"),
    )
    output_dir = _extract_labeled_path(
        text,
        ("output", "输出目录", "输出路径", "导出目录", "结果目录"),
    )
    candidates = _extract_absolute_path_candidates(text)
    if input_dir is None and candidates:
        input_dir = candidates[0]
    if output_dir is None and len(candidates) > 1:
        output_dir = candidates[1]
    if input_dir is None:
        return "", output_dir, False
    if output_dir is None:
        output_dir = _infer_sibling_output_dir(text, input_dir)
    return input_dir, output_dir, output_dir is None


def _infer_sibling_output_dir(text: str, input_dir: str) -> str | None:
    normalized = text.lower()
    output_markers = (
        "output 里",
        "output里",
        "output 文件夹",
        "output文件夹",
        "output 目录",
        "output目录",
        "输出放 output",
        "输出到 output",
        "结果放 output",
        "结果到 output",
    )
    if not any(marker in normalized for marker in output_markers):
        return None
    path = Path(input_dir).expanduser()
    if path.name.lower() != "input":
        return None
    return str(path.parent / "output")


def _looks_like_same_directory_output(text: str) -> bool:
    normalized = text.lower()
    return any(
        marker in normalized
        for marker in (
            "输出和输入放在同一个",
            "输入和输出放在同一个",
            "输出和输入同一个",
            "输入和输出同一个",
            "输出放同一个",
            "输出到同一个",
            "同一个目录",
            "同一个文件夹",
            "同一目录",
            "同一文件夹",
            "same directory",
            "same folder",
        )
    )


def _safe_translation_output_dir(input_dir: str) -> str:
    path = Path(input_dir).expanduser()
    name = path.name or "translation"
    return str(path.parent / f"{name}-translated")


def _is_output_inside_input(input_dir: str, output_dir: str) -> bool:
    try:
        input_path = Path(input_dir).expanduser().resolve()
        output_path = Path(output_dir).expanduser().resolve()
    except OSError:
        return False
    try:
        output_path.relative_to(input_path)
    except ValueError:
        return False
    return input_path != output_path


def _extract_labeled_path(text: str, labels: tuple[str, ...]) -> str | None:
    label_pattern = "|".join(re.escape(label) for label in labels)
    pattern = rf"(?:{label_pattern})\s*(?:folder|dir|目录|路径)?\s*(?:是|为|=|:|：)?\s*(/[^\n，。；;]+)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
        return None
    value = _trim_path_like_value(match.group(1))
    return value or None


def _extract_absolute_path_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for match in re.finditer(r"(?:^|(?<=[\s:=：]))(/[^\n，。；;]+)", text):
        value = _trim_path_like_value(match.group(1))
        if value and value not in candidates:
            candidates.append(value)
    return candidates


def _trim_path_like_value(value: str) -> str:
    trimmed = value.strip(" \t\r\n`\"'“”")
    guide_match = re.search(
        r"\s+(?:BL\s*)?作品指南\b|\s+(?:小说背景|作品背景|背景/类型|背景|类型|作品关键词|人物介绍|角色介绍)\b",
        trimmed,
    )
    if guide_match is not None:
        trimmed = trimmed[: guide_match.start()]
    stop_markers = (
        " output",
        " input",
        " 输出",
        " 输入",
        "。output",
        "。input",
        "。输出",
        "。输入",
        "，output",
        "，input",
        "，输出",
        "，输入",
        "；output",
        "；input",
        "；输出",
        "；输入",
        "。小说背景",
        "，小说背景",
        "；小说背景",
        "。背景",
        "，背景",
        "；背景",
        "。类型",
        "，类型",
        "；类型",
        "。作品关键词",
        "，作品关键词",
        "；作品关键词",
        "。作品指南",
        "，作品指南",
        "；作品指南",
        "。BL 作品指南",
        "，BL 作品指南",
        "；BL 作品指南",
        "。人物介绍",
        "，人物介绍",
        "；人物介绍",
        " 小说背景",
        " 背景",
        " 类型",
        " 作品关键词",
        " 作品指南",
        " BL 作品指南",
        " 人物介绍",
        "\n",
    )
    for marker in stop_markers:
        index = trimmed.find(marker)
        if index > 0:
            trimmed = trimmed[:index]
    return trimmed.strip(" \t\r\n，。；;`\"'“”")


def _extract_novel_background(text: str) -> str:
    guide_block = _extract_novel_background_guide_block(text)
    if guide_block:
        return guide_block
    match = re.search(
        r"(?:小说背景|作品背景|背景(?:/类型)?|世界观|类型)\s*(?:是|为|=|:|：)?\s*(.+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return ""
    value = match.group(1).strip()
    for marker in (
        " input",
        " output",
        " 输入",
        " 输出",
    ):
        index = value.find(marker)
        if index > 0:
            value = value[:index]
    return _clean_novel_background_block(value)


def _extract_novel_background_guide_block(text: str) -> str:
    marker_match = re.search(r"(?:BL\s*)?作品指南", text)
    if marker_match is not None:
        return _clean_novel_background_block(text[marker_match.start() :])
    background_match = re.search(
        r"(?:小说背景|作品背景|背景(?:/类型)?|世界观|类型)\s*(?:是|为|=|:|：)?",
        text,
        flags=re.IGNORECASE,
    )
    if background_match is None:
        return ""
    block = text[background_match.start() :]
    if not any(marker in block for marker in ("作品关键词", "人物介绍", "角色介绍")):
        return ""
    return _clean_novel_background_block(block)


def _clean_novel_background_block(value: str) -> str:
    lines: list[str] = []
    previous_blank = False
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line:
            if lines and not previous_blank:
                lines.append("")
                previous_blank = True
            continue
        if _looks_like_background_control_line(line):
            continue
        lines.append(line)
        previous_blank = False
    cleaned = "\n".join(lines).strip(" \t\r\n，。；;")
    return cleaned[:_MAX_NOVEL_BACKGROUND_LENGTH].strip(" \t\r\n，。；;")


def _looks_like_background_control_line(line: str) -> bool:
    if re.match(r"^(?:input|output|输入|输出).*(?:/|目录|路径)", line, re.IGNORECASE):
        return True
    if re.match(r"^/[^，。；;]+$", line):
        return True
    return any(
        marker in line
        for marker in (
            "输出和输入放在同一个",
            "输入和输出放在同一个",
            "输出目录默认",
            "默认使用 input",
        )
    )


def _extract_source_language(text: str) -> str:
    normalized = text.lower()
    if any(marker in normalized for marker in ("韩译中", "韩文", "韩语", "korean", " kr", " ko")):
        return "kr"
    if any(marker in normalized for marker in ("日译中", "日文", "日语", "japanese", " ja", " jp")):
        return "ja"
    return ""


def _extract_target_language(text: str) -> str:
    normalized = text.lower()
    if any(marker in normalized for marker in ("繁中", "繁体", "traditional chinese", "zh-hant")):
        return "zh-Hant"
    if any(marker in normalized for marker in ("译中", "中文", "简中", "简体", "chinese", " zh")):
        return "zh"
    return ""


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
    normalized = text.lower()
    has_model = any(marker in normalized for marker in ("模型", "model", "profile"))
    has_copy = any(
        marker in normalized
        for marker in (
            "按照",
            "基于",
            "复制",
            "克隆",
            "参照",
            "照着",
            "一样的",
            "same",
            "copy",
            "clone",
            "duplicate",
        )
    )
    has_create = any(
        marker in normalized
        for marker in (
            "创建",
            "新建",
            "配置一个",
            "新模型",
            "新配置",
            "复制一个",
            "复制一套",
            "add",
            "create",
        )
    )
    return has_model and has_copy and has_create


def _looks_like_model_profile_copy_continuation(text: str) -> bool:
    normalized = text.lower()
    if len(text) > 500:
        return False
    has_confirmation = any(
        marker in normalized
        for marker in (
            "对的",
            "是的",
            "可以",
            "确认",
            "创建一个新的",
            "用这个",
            "不要用相同",
            "改为",
            "改成",
            "model_id",
            "模型 id",
            "模型id",
            "provider format",
            "base_url",
            "接口模型",
        )
    )
    return has_confirmation


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
    has_concrete_connection = any(
        marker in normalized
        for marker in (
            "base_url",
            "api key",
            "apikey",
            "密钥",
            "provider model_id",
            "model_id",
            "模型 id",
            "模型id",
            "接口地址",
        )
    )
    return has_model and has_create_or_config and has_uncertainty and not has_concrete_connection


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
    requested_families = _model_family_tokens_from_text(user_message)
    current_profiles = _current_workspace_profiles(
        state,
        profile_store=profile_store,
    )
    if not requested_families:
        for profile in current_profiles:
            requested_families.update(_model_family_tokens_for_profile(profile))

    comparable_current_profiles = [
        profile
        for profile in current_profiles
        if not requested_families
        or (_model_family_tokens_for_profile(profile) & requested_families)
    ]
    current_best = max(
        (_model_quality_score(profile) for profile in comparable_current_profiles),
        default=-100,
    )
    candidates: list[tuple[int, str, ModelConfig]] = []
    for profile in profile_store.load():
        if not profile.api_keys:
            continue
        families = _model_family_tokens_for_profile(profile)
        if requested_families and not (families & requested_families):
            continue
        score = _model_quality_score(profile)
        if score <= current_best and requested_families:
            continue
        candidates.append((score, profile.display_name.lower(), profile))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_score = candidates[0][0]
    best = [profile for score, _, profile in candidates if score == best_score]
    if len(best) != 1:
        return None
    if best_score < 10:
        return None
    return best[0]


def _current_workspace_profiles(
    state: AgentWorkspaceState,
    *,
    profile_store: ModelProfileStore,
) -> list[ModelConfig]:
    ids = [state.workflow_model_id, *state.stage_model_ids.values()]
    profiles: list[ModelConfig] = []
    seen: set[str] = set()
    for profile_id in ids:
        if not profile_id or profile_id in seen:
            continue
        profile = profile_store.get(profile_id)
        if profile is None:
            continue
        seen.add(profile.id)
        profiles.append(profile)
    return profiles


def _is_same_model_family(left: ModelConfig, right: ModelConfig) -> bool:
    return bool(
        _model_family_tokens_for_profile(left) & _model_family_tokens_for_profile(right)
    )


def _model_family_tokens_for_profile(profile: ModelConfig) -> set[str]:
    return _model_family_tokens_from_text(
        " ".join((profile.id, profile.display_name, profile.model_id))
    )


def _model_family_tokens_from_text(text: str) -> set[str]:
    normalized = text.lower()
    aliases = {
        "deepseek": ("deepseek", "deep seek", "深度求索"),
        "gemini": ("gemini", "google"),
        "claude": ("claude", "anthropic"),
        "gpt": ("gpt", "openai", "chatgpt"),
        "qwen": ("qwen", "通义", "千问"),
        "kimi": ("kimi", "moonshot"),
    }
    result: set[str] = set()
    for family, markers in aliases.items():
        if any(marker in normalized for marker in markers):
            result.add(family)
    return result


def _model_quality_score(profile: ModelConfig) -> int:
    label = f"{profile.id} {profile.display_name} {profile.model_id}".lower()
    score = 0
    if "pro" in label:
        score += 40
    if "reasoner" in label or "reasoning" in label:
        score += 36
    if "opus" in label:
        score += 34
    if "sonnet" in label:
        score += 24
    if "max" in label or "ultra" in label:
        score += 18
    if profile.thinking_level is not ThinkingLevel.OFF:
        score += 8
    if "flash" in label:
        score -= 16
    if any(marker in label for marker in ("mini", "lite", "nano", "haiku")):
        score -= 20
    return score


def _model_profile_copy_payload(
    source: ModelConfig,
    *,
    display_name: str,
    provider_model_id: str | None,
) -> dict[str, object]:
    body = source.to_dict()
    body.pop("id", None)
    body.pop("api_keys", None)
    body["copy_from_profile_id"] = source.id
    body["display_name"] = display_name
    if provider_model_id:
        body["model_id"] = provider_model_id
    return body


def _resolve_model_profile_from_copy_request(
    text: str,
    *,
    profile_store: ModelProfileStore,
) -> ModelConfig | None:
    labels = _extract_model_copy_source_labels(text)
    for label in labels:
        resolved = _resolve_model_profile_by_label(label, profile_store)
        if resolved is not None:
            return resolved
        fuzzy = _resolve_model_profile_by_fuzzy_text(label, profile_store)
        if fuzzy is not None:
            return fuzzy
    return _resolve_model_profile_by_fuzzy_text(text, profile_store)


def _extract_model_copy_source_labels(text: str) -> list[str]:
    labels: list[str] = []
    patterns = (
        r"(?:按照|基于|参照|照着|复制|克隆|从)\s*([A-Za-z0-9_.\-\s]+?)\s*(?:的配置|配置|模型|profile)",
        r"(?:copy|clone|duplicate|from|based on)\s+([A-Za-z0-9_.\-\s]+?)(?:\s+profile|\s+model|$)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            label = match.group(1).strip(" \t\n\r:：，。,.")
            if label:
                labels.append(label)
    return labels


def _resolve_model_profile_by_fuzzy_text(
    text: str,
    profile_store: ModelProfileStore,
) -> ModelConfig | None:
    text_tokens = _lookup_tokens(text)
    if not text_tokens:
        return None
    scored: list[tuple[int, ModelConfig]] = []
    normalized_text = _normalize_lookup_label(text)
    for profile in profile_store.load():
        labels = (profile.id, profile.display_name, profile.model_id)
        score = 0
        for label in labels:
            normalized_label = _normalize_lookup_label(label)
            if normalized_label and normalized_label in normalized_text:
                score += 6
            score += len(text_tokens & _lookup_tokens(label))
        if score:
            scored.append((score, profile))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored[0][0] < 2:
        return None
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]


def _lookup_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", value.lower())
        if token
        not in {
            "api",
            "base",
            "config",
            "format",
            "id",
            "key",
            "model",
            "profile",
            "provider",
            "url",
            "一样",
            "一样的",
            "不同",
            "新的",
            "模型",
            "相同",
            "配置",
        }
    }


def _extract_model_copy_display_name(text: str) -> str:
    patterns = (
        r"(?:模型(?:显示)?(?:名称|名字|名)|display_name|display name)\s*(?:叫做|叫|改为|改成|设为|设置为|为|=|:|：)\s*[「『“\"']?(.+?)[」』”\"']?(?:[，。,.]|$)",
        r"(?:配置|创建|新建)(?:一个|一套)?\s*[「『“\"']?(.+?)[」』”\"']?(?:的)?(?:新)?模型(?:配置)?",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip(" \t\n\r「」『』“”\"'")
            if value:
                return value[:_MAX_TITLE_LENGTH]
    return ""


def _extract_provider_model_id_override(text: str) -> str | None:
    patterns = (
        r"(?:model[_\s-]?id|模型\s*ID|接口模型|provider\s+model(?:\s+id)?)\s*(?:改为|改成|设置为|设为|使用|用|为|=|:|：)\s*[`\"']?([^`\"'，。,\s]+)",
        r"[`\"']([A-Za-z0-9_.:/\-]+)[`\"']\s*(?:这个)?\s*(?:model[_\s-]?id|模型\s*ID|接口模型)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if value:
                return value[:_MAX_TITLE_LENGTH]
    return None


def _infer_provider_model_id_from_copy_request(
    source: ModelConfig,
    *,
    display_name: str,
    text: str,
) -> str | None:
    if not _requests_different_provider_model_id(text):
        return None
    source_model_id = source.model_id.strip()
    if not source_model_id:
        return None
    target = display_name.lower()
    if "pro" in target and re.search(r"(?:^|[-_.])flash(?:$|[-_.])", source_model_id):
        return re.sub(
            r"(?i)(?:^|(?<=[-_.]))flash(?=$|[-_.])",
            "pro",
            source_model_id,
            count=1,
        )[:_MAX_TITLE_LENGTH]
    if "flash" in target and re.search(r"(?:^|[-_.])pro(?:$|[-_.])", source_model_id):
        return re.sub(
            r"(?i)(?:^|(?<=[-_.]))pro(?=$|[-_.])",
            "flash",
            source_model_id,
            count=1,
        )[:_MAX_TITLE_LENGTH]
    return None


def _requests_different_provider_model_id(text: str) -> bool:
    normalized = text.lower()
    return bool(
        re.search(r"(?:不要|不|不能|别).{0,8}(?:相同|一样).{0,8}(?:model[_\s-]?id|模型\s*id)", normalized)
        or re.search(r"(?:model[_\s-]?id|模型\s*id|接口模型).{0,12}(?:新的|不同|不一样|改为|改成|换成)", normalized)
    )


def _salvage_create_prompt_draft(
    draft: AgentActionDraft,
    *,
    text: str,
) -> AgentActionDraft | None:
    try:
        prompt_kind = _coerce_prompt_kind(None, fallback_text=text)
    except BridgeError:
        return None
    name = _extract_quoted_name(text) or _extract_named_value(text)
    if not name:
        return None
    system_prompt = _prompt_body_from_payload(draft.payload)
    if not system_prompt:
        system_prompt = _build_prompt_body_from_request(
            prompt_kind=prompt_kind,
            user_request=text,
        )
    payload = dict(draft.payload)
    payload.update(
        {
            "kind": prompt_kind.value,
            "name": name,
            "description": str(payload.get("description") or draft.summary).strip(),
            "system_prompt": system_prompt,
            "enabled": bool(payload.get("enabled", True)),
        }
    )
    return AgentActionDraft.create(
        kind=draft.kind,
        title=draft.title,
        summary=draft.summary,
        payload=payload,
    )


def _salvage_create_recipe_draft(
    draft: AgentActionDraft,
    *,
    text: str,
    current_state: Mapping[str, object],
) -> AgentActionDraft | None:
    name = _extract_quoted_name(text) or _extract_named_value(text)
    if not name:
        return None
    stage_models = current_state.get("stage_model_ids")
    stage_prompts = current_state.get("stage_prompt_ids")
    payload = _recipe_body_from_payload(draft.payload)
    payload.update(
        {
            "name": name,
            "description": str(payload.get("description") or draft.summary).strip(),
            "stage_model_ids": dict(stage_models)
            if isinstance(stage_models, Mapping)
            else {},
            "stage_prompt_ids": dict(stage_prompts)
            if isinstance(stage_prompts, Mapping)
            else {},
        }
    )
    return AgentActionDraft.create(
        kind=draft.kind,
        title=draft.title,
        summary=draft.summary,
        payload=payload,
    )


def _salvage_compound_draft(
    draft: AgentActionDraft,
    *,
    text: str,
    current_state: Mapping[str, object],
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> AgentActionDraft | None:
    actions: list[dict[str, object]] = []
    concurrency = _extract_model_concurrency_update(text, profile_store=profile_store)
    if concurrency is not None:
        profile_id, limit = concurrency
        actions.append(
            {
                "kind": "update_model_profile",
                "title": "更新模型并发数",
                "summary": f"将模型 {profile_id} 的并发数改为 {limit}。",
                "payload": {
                    "profile_id": profile_id,
                    "patch": {"concurrency_limit": limit},
                },
            }
        )
    prompt_rename = _extract_prompt_rename(text, cache_root=cache_root)
    if prompt_rename is not None:
        preset, old_name, new_name = prompt_rename
        if preset.is_system:
            actions.append(
                {
                    "kind": "create_prompt_preset",
                    "title": "创建 Prompt 自定义副本",
                    "summary": (
                        f"内置 Prompt {old_name} 为只读；创建同内容的自定义副本 "
                        f"{new_name}。"
                    ),
                    "payload": {
                        "kind": preset.kind.value,
                        "name": new_name,
                        "description": f"从内置 Prompt {old_name} 复制创建。",
                        "system_prompt": preset.system_prompt,
                        "enabled": preset.enabled,
                    },
                }
            )
        else:
            actions.append(
                {
                    "kind": "update_prompt_preset",
                    "title": "重命名 Prompt",
                    "summary": f"将 Prompt {old_name} 重命名为 {new_name}。",
                    "payload": {
                        "id": preset.id,
                        "patch": {"name": new_name},
                    },
                }
            )
    recipe_name = _extract_recipe_save_name(text)
    if recipe_name:
        stage_models = current_state.get("stage_model_ids")
        stage_prompts = current_state.get("stage_prompt_ids")
        actions.append(
            {
                "kind": "create_recipe",
                "title": "保存当前阶段配置",
                "summary": f"保存当前阶段配置为 {recipe_name}。",
                "payload": {
                    "name": recipe_name,
                    "description": "由 Agent Lab 根据当前工作区阶段配置创建。",
                    "stage_model_ids": dict(stage_models)
                    if isinstance(stage_models, Mapping)
                    else {},
                    "stage_prompt_ids": dict(stage_prompts)
                    if isinstance(stage_prompts, Mapping)
                    else {},
                },
            }
        )
    if not actions:
        return None
    return AgentActionDraft.create(
        kind=draft.kind,
        title=draft.title,
        summary=draft.summary,
        payload={"actions": actions},
    )


def _build_prompt_body_from_request(
    *,
    prompt_kind: PromptKind,
    user_request: str,
) -> str:
    stage = {
        PromptKind.TRANSLATION: "翻译",
        PromptKind.GLOSSARY: "术语提取",
        PromptKind.GLOSSARY_REVIEW: "术语审核",
    }[prompt_kind]
    return "\n".join(
        (
            f"你是 Transoria {stage}流程中的专业模型。",
            "请严格遵守用户对这套 Prompt 的要求：",
            user_request.strip(),
            "保持输出清晰、稳定，并优先服务于小说翻译质量。",
        )
    )


def _extract_quoted_name(text: str) -> str:
    for pattern in (r"[『「“](.+?)[』」”]", r"['\"](.+?)['\"]"):
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            if value:
                return value[:_MAX_RECIPE_NAME_LENGTH]
    return ""


def _extract_named_value(text: str) -> str:
    match = re.search(r"(?:名字|名称|叫|名为)\s*[:：]?\s*([^\s，。,.]+)", text)
    if not match:
        return ""
    return match.group(1).strip("『』「」“”\"' ")[:_MAX_RECIPE_NAME_LENGTH]


def _extract_model_concurrency_update(
    text: str,
    *,
    profile_store: ModelProfileStore,
) -> tuple[str, int] | None:
    patterns = (
        r"(?:模型|model)\s*[「『“\"']?(.+?)[」』”\"']?\s*的?并发(?:数)?(?:改成|修改为|设置为|设为|=)\s*(\d+)",
        r"[「『“\"']?([A-Za-z0-9_.\- ]+)[」』”\"']?\s*的?并发(?:数)?(?:改成|修改为|设置为|设为|=)\s*(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        label = match.group(1).strip(" \t\r\n「」『』“”\"'")
        profile = _resolve_model_profile_by_label(label, profile_store)
        if profile is None:
            continue
        limit = int(match.group(2))
        if limit < 0:
            continue
        return profile.id, limit
    return None


def _resolve_model_profile_by_label(
    label: str,
    profile_store: ModelProfileStore,
) -> ModelConfig | None:
    normalized = _normalize_lookup_label(label)
    matches = [
        profile
        for profile in profile_store.load()
        if _normalize_lookup_label(profile.id) == normalized
        or _normalize_lookup_label(profile.display_name) == normalized
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _extract_prompt_rename(
    text: str,
    *,
    cache_root: Path,
) -> tuple[PromptPreset, str, str] | None:
    match = re.search(
        r"(?:(翻译|术语提取|术语审核|术语审查)\s*)?Prompt\s*[「『“\"](.+?)[」』”\"]\s*(?:重命名为|改名为|改成|修改为)\s*[「『“\"](.+?)[」』”\"]",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    kind_text, old_name, new_name = match.groups()
    kinds = (
        (_coerce_prompt_kind(kind_text, fallback_text=kind_text),)
        if kind_text
        else tuple(PromptKind)
    )
    preset = _resolve_prompt_by_name(old_name, kinds=kinds, cache_root=cache_root)
    if preset is None:
        return None
    return preset, old_name.strip(), new_name.strip()[:_MAX_TITLE_LENGTH]


def _resolve_prompt_by_name(
    name: str,
    *,
    kinds: tuple[PromptKind, ...],
    cache_root: Path,
) -> PromptPreset | None:
    normalized = _normalize_lookup_label(name)
    matches: list[PromptPreset] = []
    for kind in kinds:
        matches.extend(
            preset
            for preset in _prompt_store_for(cache_root, kind).load()
            if _normalize_lookup_label(preset.id) == normalized
            or _normalize_lookup_label(preset.name) == normalized
        )
    if len(matches) == 1:
        return matches[0]
    custom_matches = [preset for preset in matches if not preset.is_system]
    if len(custom_matches) == 1:
        return custom_matches[0]
    return None


def _extract_recipe_save_name(text: str) -> str:
    matches = re.findall(
        r"(?:保存成|保存为|存成|存为)(?:名为|名字叫|叫)?[「『“\"](.+?)[」』”\"](?:的)?(?:预设|配方)",
        text,
    )
    if not matches:
        return ""
    return matches[-1].strip()[:_MAX_RECIPE_NAME_LENGTH]


def _normalize_lookup_label(value: str) -> str:
    return value.strip().lower().replace(" ", "").replace("·", "")


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
    workflow_model_id = state.workflow_model_id
    if "workflow_model_id" in patch:
        workflow_model_id = _coerce_model_id(
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
        workflow_thinking_level = _coerce_workflow_thinking_level(
            patch.get("workflow_thinking_level"),
            workflow_model_id=workflow_model_id,
            profile_store=profile_store,
        )
    stage_model_ids = dict(state.stage_model_ids)
    if "stage_model_ids" in patch:
        stage_model_ids.update(
            _coerce_model_slots(
                patch.get("stage_model_ids"),
                profile_store=profile_store,
            )
        )
    stage_prompt_ids = dict(state.stage_prompt_ids)
    if "stage_prompt_ids" in patch:
        stage_prompt_ids.update(
            _coerce_prompt_slots(patch.get("stage_prompt_ids"), cache_root=cache_root)
        )
    active_recipe_id = state.active_recipe_id
    stage_selection_touched = "stage_model_ids" in patch or "stage_prompt_ids" in patch
    if "active_recipe_id" in patch:
        active_recipe_id = _coerce_active_recipe_id(
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


def _apply_draft(
    state: AgentWorkspaceState,
    draft: AgentActionDraft,
    *,
    profile_store: ModelProfileStore,
    cache_root: Path,
    settings_store: SettingsStore,
    task_service: TaskService,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    spec = _agent_action_spec(draft.kind)
    spec.validate(
        draft,
        state=state,
        profile_store=profile_store,
        cache_root=cache_root,
        task_service=task_service,
    )
    return spec.apply(
        draft,
        state=state,
        profile_store=profile_store,
        cache_root=cache_root,
        settings_store=settings_store,
        task_service=task_service,
    )


def _applied_draft_message(
    draft: AgentActionDraft,
    result: Mapping[str, object],
    *,
    task_service: TaskService,
) -> str:
    task_summaries = _task_start_summaries_from_result(
        result,
        task_service=task_service,
    )
    if not task_summaries:
        return f"已应用草案：{draft.title}"
    if len(task_summaries) == 1:
        summary = task_summaries[0]
        return (
            f"已应用草案：{draft.title}\n\n"
            f"已启动{summary['label']}任务。\n"
            f"任务 ID：{summary['task_id']}\n"
            f"当前进度阶段：{summary['status']}。\n"
            f"你可以在 {summary['dashboard']} 查看实时进度。任务结束前，Agent Lab 不会再启动其他任务。"
        )
    lines = [f"已应用草案：{draft.title}", "", "已启动以下任务："]
    for summary in task_summaries:
        lines.append(
            f"- {summary['label']}：{summary['task_id']}，当前阶段：{summary['status']}，"
            f"可在 {summary['dashboard']} 查看。"
        )
    lines.append("任务结束前，Agent Lab 不会再启动其他任务。")
    return "\n".join(lines)


def _task_start_summaries_from_result(
    result: Mapping[str, object],
    *,
    task_service: TaskService,
) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    if isinstance(result.get("results"), list):
        for item in result["results"]:  # type: ignore[index]
            if isinstance(item, Mapping):
                summaries.extend(
                    _task_start_summaries_from_result(
                        item,
                        task_service=task_service,
                    )
                )
        return summaries
    task = result.get("task")
    if not isinstance(task, Mapping):
        return []
    task_id = str(task.get("task_id") or "")
    task_kind = str(task.get("kind") or "")
    if not task_id or task_kind not in _AGENT_TASK_KINDS:
        return []
    header = _task_header_or_none(task_service, task_kind, task_id)
    status = (
        _task_status_label(str(header["status"]))
        if header is not None and isinstance(header.get("status"), str)
        else "已提交，等待任务状态刷新"
    )
    summaries.append(
        {
            "task_id": task_id,
            "kind": task_kind,
            "label": _task_kind_label(task_kind),
            "dashboard": _task_dashboard_label(task_kind),
            "status": status,
        }
    )
    return summaries


def _task_kind_label(kind: str) -> str:
    return {
        "translation": "翻译",
        "glossary": "术语提取",
        "glossary_review": "术语审查",
    }.get(kind, kind)


def _task_dashboard_label(kind: str) -> str:
    return {
        "translation": "翻译 dashboard",
        "glossary": "术语提取 dashboard",
        "glossary_review": "术语审查 dashboard",
    }.get(kind, "对应 dashboard")


def _task_status_label(status: str) -> str:
    return {
        "pending": "已提交，等待运行",
        "running": "正在运行",
        "paused": "已暂停",
        "stopping": "正在停止",
        "completed": "已完成",
        "failed": "已失败",
        "cancelled": "已取消",
        "stopped": "已停止",
    }.get(status, status or "已提交，等待任务状态刷新")


def _validate_draft(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    cache_root: Path,
    task_service: TaskService,
) -> None:
    spec = _agent_action_spec(draft.kind)
    spec.validate(
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
    warnings = _model_quality_warnings(
        draft,
        state=state,
        profile_store=profile_store,
    )
    if not warnings:
        return reply
    suffix = "模型风险提示：" + "；".join(warnings)
    if suffix in reply:
        return reply
    return f"{reply}\n\n{suffix}"


def _model_quality_warnings(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
) -> list[str]:
    seen: set[tuple[str, str]] = set()
    warnings: list[str] = []

    def add(slot: str, profile: ModelConfig | None) -> None:
        if profile is None:
            return
        risk = _profile_quality_risk(profile)
        if risk is None:
            return
        key = (slot, profile.id)
        if key in seen:
            return
        seen.add(key)
        stage = _stage_label(slot)
        warnings.append(f"{stage}使用 {profile.display_name}（{profile.model_id}）{risk}")

    if draft.kind == "update_workspace":
        for slot, profile_id in _model_slot_ids_from_payload(draft.payload).items():
            add(slot, profile_store.get(profile_id))
        workflow_id = draft.payload.get("workflow_model_id")
        if isinstance(workflow_id, str) and workflow_id:
            profile = profile_store.get(workflow_id)
            if profile is not None and _profile_quality_risk(profile) is not None:
                add("workflow", profile)
    elif draft.kind in {"create_recipe", "update_recipe"}:
        for slot, profile_id in _model_slot_ids_from_payload(draft.payload).items():
            add(slot, profile_store.get(profile_id))
    elif draft.kind == "apply_recipe":
        recipe = state.get_recipe(
            str(draft.payload.get("recipe_id") or draft.payload.get("id") or "")
        )
        if recipe is not None:
            for slot, profile_id in recipe.stage_model_ids.items():
                if profile_id:
                    add(slot, profile_store.get(profile_id))
    elif draft.kind == "create_model_profile":
        try:
            add(
                "new_profile",
                _model_profile_from_draft_payload(
                    draft.payload,
                    profile_store=profile_store,
                ),
            )
        except BridgeError:
            return warnings
    elif draft.kind == "update_model_profile":
        profile_id, patch, _api_keys = _coerce_model_profile_update_payload(
            draft.payload
        )
        current = profile_store.get(profile_id)
        if current is not None:
            preview = current
            if patch:
                try:
                    preview = replace(
                        current,
                        **_coerce_model_profile_patch(patch),  # type: ignore[arg-type]
                    )
                except BridgeError:
                    preview = current
            add("updated_profile", preview)
    elif draft.kind == "compound_config_update":
        try:
            inner_drafts = _coerce_compound_action_drafts(draft.payload)
        except BridgeError:
            return warnings
        for inner in inner_drafts:
            for warning in _model_quality_warnings(
                inner,
                state=state,
                profile_store=profile_store,
            ):
                if warning not in warnings:
                    warnings.append(warning)

    return warnings


def _model_slot_ids_from_payload(payload: Mapping[str, object]) -> dict[str, str]:
    raw = payload.get("stage_model_ids")
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, str] = {}
    for slot, value in raw.items():
        if isinstance(value, str) and value:
            result[str(slot)] = value
    return result


def _profile_quality_risk(profile: ModelConfig) -> str | None:
    label = f"{profile.display_name} {profile.model_id}".lower()
    weak_markers = (
        "flash",
        "mini",
        "lite",
        "nano",
        "small",
        "fast",
        "cheap",
        "haiku",
    )
    if not any(marker in label for marker in weak_markers):
        return None
    return "看起来是快速或低成本档，质量、术语一致性和复杂上下文理解可能弱于高质量模型"


def _stage_label(slot: str) -> str:
    return {
        "workflow": "工作模型",
        "translation": "翻译阶段",
        "term_extract": "术语提取阶段",
        "term_review": "术语审核阶段",
        "new_profile": "新模型配置",
        "updated_profile": "被更新的模型配置",
    }.get(slot, slot)


def _agent_action_spec(kind: str) -> _AgentActionSpec:
    specs = _agent_action_specs()
    spec = specs.get(kind)
    if spec is None:
        raise BridgeError.invalid_argument(
            f"unsupported draft kind: {kind!r}",
            details={"kind": kind},
        )
    if spec.mutates and not spec.requires_confirmation:  # pragma: no cover
        raise BridgeError.invalid_argument(
            f"mutating action must require confirmation: {kind!r}",
            details={"kind": kind},
        )
    return spec


def _agent_action_specs() -> dict[str, _AgentActionSpec]:
    specs = {
        "update_workspace": _AgentActionSpec(
            kind="update_workspace",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=_validate_workspace_action,
            apply=_apply_workspace_action,
        ),
        "create_prompt_preset": _AgentActionSpec(
            kind="create_prompt_preset",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=_validate_create_prompt_action,
            apply=_apply_create_prompt_action,
        ),
        "update_prompt_preset": _AgentActionSpec(
            kind="update_prompt_preset",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=_validate_update_prompt_action,
            apply=_apply_update_prompt_action,
        ),
        "update_memory": _AgentActionSpec(
            kind="update_memory",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=_validate_update_memory_action,
            apply=_apply_update_memory_action,
        ),
        "add_memory": _AgentActionSpec(
            kind="add_memory",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=_validate_add_memory_action,
            apply=_apply_add_memory_action,
        ),
        "delete_memory": _AgentActionSpec(
            kind="delete_memory",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=_validate_delete_memory_action,
            apply=_apply_delete_memory_action,
        ),
        "create_recipe": _AgentActionSpec(
            kind="create_recipe",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=_validate_create_recipe_action,
            apply=_apply_create_recipe_action,
        ),
        "update_recipe": _AgentActionSpec(
            kind="update_recipe",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=_validate_update_recipe_action,
            apply=_apply_update_recipe_action,
        ),
        "apply_recipe": _AgentActionSpec(
            kind="apply_recipe",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=_validate_apply_recipe_action,
            apply=_apply_apply_recipe_action,
        ),
        "delete_recipe": _AgentActionSpec(
            kind="delete_recipe",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=_validate_delete_recipe_action,
            apply=_apply_delete_recipe_action,
        ),
        "create_model_profile": _AgentActionSpec(
            kind="create_model_profile",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=_validate_create_model_profile_action,
            apply=_apply_create_model_profile_action,
        ),
        "update_model_profile": _AgentActionSpec(
            kind="update_model_profile",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=_validate_update_model_profile_action,
            apply=_apply_update_model_profile_action,
        ),
        "compound_config_update": _AgentActionSpec(
            kind="compound_config_update",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=_validate_compound_config_action,
            apply=_apply_compound_config_action,
        ),
    }
    for kind in START_DRAFT_KINDS:
        specs[kind] = _AgentActionSpec(
            kind=kind,
            mutates=True,
            requires_confirmation=True,
            starts_task=True,
            validate=_validate_start_task_action,
            apply=_apply_start_task_action,
        )
    return specs


def _validate_compound_config_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    cache_root: Path,
    settings_store: SettingsStore | None = None,
    task_service: TaskService,
    **_: object,
) -> None:
    validation_state = state
    validation_profile_store: ModelProfileStore | _PreviewModelProfileStore = (
        profile_store
    )
    for inner in _coerce_compound_action_drafts(draft.payload):
        spec = _agent_action_spec(inner.kind)
        spec.validate(
            inner,
            state=validation_state,
            profile_store=validation_profile_store,  # type: ignore[arg-type]
            cache_root=cache_root,
            settings_store=settings_store,
            task_service=task_service,
        )
        validation_state = _preview_compound_state(
            validation_state,
            inner,
            profile_store=validation_profile_store,  # type: ignore[arg-type]
            cache_root=cache_root,
        )
        validation_profile_store = _preview_compound_profile_store(
            validation_profile_store,
            inner,
        )


def _apply_compound_config_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    cache_root: Path,
    settings_store: SettingsStore,
    task_service: TaskService,
    **_: object,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    _validate_compound_config_action(
        draft,
        state=state,
        profile_store=profile_store,
        cache_root=cache_root,
        settings_store=settings_store,
        task_service=task_service,
    )
    next_state = state
    results: list[dict[str, object]] = []
    for inner in _coerce_compound_action_drafts(draft.payload):
        spec = _agent_action_spec(inner.kind)
        next_state, result = spec.apply(
            inner,
            state=next_state,
            profile_store=profile_store,
            cache_root=cache_root,
            settings_store=settings_store,
            task_service=task_service,
        )
        results.append(dict(result))
    return next_state, {"kind": draft.kind, "results": results}


def _coerce_compound_action_drafts(
    payload: Mapping[str, object],
) -> list[AgentActionDraft]:
    raw_actions = _compound_raw_actions(payload)
    if not isinstance(raw_actions, list):
        raise BridgeError.invalid_argument(
            "compound draft payload.actions must be a list.",
            field="actions",
        )
    if not raw_actions:
        raise BridgeError.invalid_argument(
            "compound draft must contain at least one action.",
            field="actions",
        )
    if len(raw_actions) > _MAX_COMPOUND_ACTIONS:
        raise BridgeError.invalid_argument(
            "compound draft contains too many actions.",
            field="actions",
            details={"max_count": _MAX_COMPOUND_ACTIONS},
        )
    drafts: list[AgentActionDraft] = []
    for index, raw in enumerate(raw_actions):
        if not isinstance(raw, Mapping):
            raise BridgeError.invalid_argument(
                "compound action must be an object.",
                field=f"actions[{index}]",
            )
        kind = str(raw.get("kind") or "")
        if kind == "compound_config_update":
            raise BridgeError.invalid_argument(
                "compound actions cannot be nested.",
                field=f"actions[{index}].kind",
            )
        spec = _agent_action_spec(kind)
        if spec.mutates and not spec.requires_confirmation:  # pragma: no cover
            raise BridgeError.invalid_argument(
                f"mutating action must require confirmation: {kind!r}",
                field=f"actions[{index}].kind",
            )
        action_payload = raw.get("payload")
        if not isinstance(action_payload, Mapping):
            action_payload = {
                key: value
                for key, value in raw.items()
                if key not in {"kind", "title", "summary", "payload"}
            }
        if not isinstance(action_payload, Mapping) or not action_payload:
            raise BridgeError.invalid_argument(
                "compound action payload must be an object.",
                field=f"actions[{index}].payload",
            )
        drafts.append(
            AgentActionDraft.create(
                kind=kind,
                title=str(raw.get("title") or kind),
                summary=str(raw.get("summary") or ""),
                payload=dict(action_payload),
            )
        )
    return drafts


def _compound_raw_actions(payload: Mapping[str, object]) -> object:
    raw_actions = _first_present(
        payload,
        (
            "actions",
            "action",
            "operations",
            "operation",
            "changes",
            "config_changes",
            "configChanges",
            "updates",
            "修改",
            "动作",
        ),
    )
    if isinstance(raw_actions, Mapping):
        if "kind" in raw_actions:
            return [raw_actions]
        mapped: list[dict[str, object]] = []
        for kind in _agent_action_specs():
            value = raw_actions.get(kind)
            if isinstance(value, Mapping):
                mapped.append({"kind": kind, "payload": dict(value)})
        if mapped:
            return mapped
    return raw_actions


def _preview_compound_state(
    state: AgentWorkspaceState,
    draft: AgentActionDraft,
    *,
    profile_store: ModelProfileStore | "_PreviewModelProfileStore",
    cache_root: Path,
) -> AgentWorkspaceState:
    if draft.kind == "update_workspace":
        return _apply_workspace_patch(
            state,
            draft.payload,
            profile_store=profile_store,
            cache_root=cache_root,
        )
    if draft.kind == "update_memory":
        return state.with_memories(_coerce_memories_payload(draft.payload))
    if draft.kind == "add_memory":
        memories = list(state.memories)
        for memory in _coerce_memory_items_payload(draft.payload):
            if memory not in memories:
                memories.append(memory)
        return state.with_memories(tuple(memories))
    if draft.kind == "delete_memory":
        removals = set(_coerce_memory_items_payload(draft.payload))
        return state.with_memories(
            tuple(item for item in state.memories if item not in removals)
        )
    if draft.kind == "create_recipe":
        name, description, stage_models, stage_prompts = _coerce_recipe_payload(
            draft.payload,
            profile_store=profile_store,
            cache_root=cache_root,
        )
        return state.add_recipe(
            AgentRecipe.create(
                name=name,
                description=description,
                stage_model_ids=stage_models,
                stage_prompt_ids=stage_prompts,
            )
        )
    if draft.kind == "update_recipe":
        recipe_id, fields, stage_models, stage_prompts = _coerce_recipe_update_payload(
            draft.payload,
            profile_store=profile_store,
            cache_root=cache_root,
        )
        recipe = state.get_recipe(recipe_id)
        if recipe is None:
            return state
        return state.replace_recipe(
            recipe.with_updates(
                name=fields.get("name"),
                description=fields.get("description"),
                stage_model_ids=stage_models,
                stage_prompt_ids=stage_prompts,
            )
        )
    if draft.kind == "apply_recipe":
        recipe = _require_recipe_from_payload(draft.payload, state=state)
        return _apply_workspace_patch(
            state,
            {
                "active_recipe_id": recipe.id,
                "stage_model_ids": dict(recipe.stage_model_ids),
                "stage_prompt_ids": dict(recipe.stage_prompt_ids),
            },
            profile_store=profile_store,
            cache_root=cache_root,
        )
    if draft.kind == "delete_recipe":
        recipe = _require_recipe_from_payload(draft.payload, state=state)
        return state.remove_recipe(recipe.id)
    return state


class _PreviewModelProfileStore:
    """Validation-only overlay for compound drafts.

    Compound proposals can create or update a model profile and then reference it
    in a later action. Validation needs to see those earlier changes without
    writing them to disk before the user confirms the whole draft.
    """

    def __init__(
        self,
        base: ModelProfileStore | "_PreviewModelProfileStore",
        overlay: Mapping[str, ModelConfig],
    ) -> None:
        self._base = base
        self._overlay = dict(overlay)

    def get(self, profile_id: str) -> ModelConfig | None:
        if profile_id in self._overlay:
            return self._overlay[profile_id]
        return self._base.get(profile_id)

    def load(self) -> tuple[ModelConfig, ...]:
        base_profiles = {
            profile.id: profile
            for profile in self._base.load()
            if profile.id not in self._overlay
        }
        return (*base_profiles.values(), *self._overlay.values())


def _preview_compound_profile_store(
    profile_store: ModelProfileStore | _PreviewModelProfileStore,
    draft: AgentActionDraft,
) -> ModelProfileStore | _PreviewModelProfileStore:
    if draft.kind == "create_model_profile":
        profile = _model_profile_from_draft_payload(
            draft.payload,
            profile_store=profile_store,  # type: ignore[arg-type]
        )
        return _PreviewModelProfileStore(profile_store, {profile.id: profile})
    if draft.kind != "update_model_profile":
        return profile_store

    profile_id, patch, api_keys = _coerce_model_profile_update_payload(draft.payload)
    current = profile_store.get(profile_id)
    if current is None:
        return profile_store
    updated = current
    if patch:
        updated = replace(
            updated,
            **_coerce_model_profile_patch(patch),  # type: ignore[arg-type]
        )
    if api_keys is not None:
        updated = updated.with_api_keys(_coerce_api_keys(api_keys))
    return _PreviewModelProfileStore(profile_store, {profile_id: updated})


def _validate_workspace_action(
    draft: AgentActionDraft,
    *,
    profile_store: ModelProfileStore,
    cache_root: Path,
    **_: object,
) -> None:
    _apply_workspace_patch(
        AgentWorkspaceState.empty(),
        draft.payload,
        profile_store=profile_store,
        cache_root=cache_root,
    )


def _apply_workspace_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    cache_root: Path,
    **_: object,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    next_state = _apply_workspace_patch(
        state,
        draft.payload,
        profile_store=profile_store,
        cache_root=cache_root,
    )
    return next_state, {"kind": draft.kind}


def _validate_create_prompt_action(draft: AgentActionDraft, **_: object) -> None:
    _coerce_prompt_preset_payload(draft.payload)


def _apply_create_prompt_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    cache_root: Path,
    **_: object,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    preset = _create_prompt_preset(draft.payload, cache_root=cache_root)
    return state, {"kind": draft.kind, "preset": _prompt_body(preset)}


def _validate_update_prompt_action(
    draft: AgentActionDraft,
    *,
    cache_root: Path,
    **_: object,
) -> None:
    preset_id, patch = _coerce_prompt_update_payload(draft.payload)
    _resolve_prompt_for_update(preset_id, patch, cache_root=cache_root)


def _apply_update_prompt_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    cache_root: Path,
    **_: object,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    preset = _update_prompt_preset(draft.payload, cache_root=cache_root)
    return state, {"kind": draft.kind, "preset": _prompt_body(preset)}


def _validate_update_memory_action(draft: AgentActionDraft, **_: object) -> None:
    _coerce_memories_payload(draft.payload)


def _apply_update_memory_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    **_: object,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    memories = _coerce_memories_payload(draft.payload)
    return state.with_memories(memories), {
        "kind": draft.kind,
        "memory_count": len(memories),
    }


def _validate_add_memory_action(draft: AgentActionDraft, **_: object) -> None:
    _coerce_memory_items_payload(draft.payload)


def _apply_add_memory_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    **_: object,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    additions = _coerce_memory_items_payload(draft.payload)
    memories = list(state.memories)
    for memory in additions:
        if memory not in memories:
            memories.append(memory)
    next_state = state.with_memories(tuple(memories))
    return next_state, {"kind": draft.kind, "memory_count": len(next_state.memories)}


def _validate_delete_memory_action(draft: AgentActionDraft, **_: object) -> None:
    _coerce_memory_items_payload(draft.payload)


def _apply_delete_memory_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    **_: object,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    removals = set(_coerce_memory_items_payload(draft.payload))
    next_state = state.with_memories(
        tuple(item for item in state.memories if item not in removals)
    )
    return next_state, {"kind": draft.kind, "memory_count": len(next_state.memories)}


def _validate_create_recipe_action(
    draft: AgentActionDraft,
    *,
    profile_store: ModelProfileStore,
    cache_root: Path,
    **_: object,
) -> None:
    _coerce_recipe_payload(
        draft.payload,
        profile_store=profile_store,
        cache_root=cache_root,
    )


def _apply_create_recipe_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    cache_root: Path,
    **_: object,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    if len(state.recipes) >= _MAX_RECIPES:
        raise BridgeError.invalid_argument(
            "too many recipes.",
            field="recipes",
            details={"max_count": _MAX_RECIPES},
        )
    name, description, stage_models, stage_prompts = _coerce_recipe_payload(
        draft.payload,
        profile_store=profile_store,
        cache_root=cache_root,
    )
    recipe = AgentRecipe.create(
        name=name,
        description=description,
        stage_model_ids=stage_models,
        stage_prompt_ids=stage_prompts,
    )
    return state.add_recipe(recipe), {"kind": draft.kind, "recipe": recipe.to_dict()}


def _validate_update_recipe_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    cache_root: Path,
    **_: object,
) -> None:
    recipe_id, _, _, _ = _coerce_recipe_update_payload(
        draft.payload,
        profile_store=profile_store,
        cache_root=cache_root,
    )
    if state.get_recipe(recipe_id) is None:
        raise BridgeError.not_found(
            f"recipe {recipe_id!r} does not exist.",
            details={"recipe_id": recipe_id},
        )


def _apply_update_recipe_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    cache_root: Path,
    **_: object,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    recipe_id, fields, stage_models, stage_prompts = _coerce_recipe_update_payload(
        draft.payload,
        profile_store=profile_store,
        cache_root=cache_root,
    )
    recipe = state.get_recipe(recipe_id)
    if recipe is None:
        raise BridgeError.not_found(
            f"recipe {recipe_id!r} does not exist.",
            details={"recipe_id": recipe_id},
        )
    updated = recipe.with_updates(
        name=fields.get("name"),
        description=fields.get("description"),
        stage_model_ids=stage_models,
        stage_prompt_ids=stage_prompts,
    )
    return state.replace_recipe(updated), {
        "kind": draft.kind,
        "recipe": updated.to_dict(),
    }


def _validate_apply_recipe_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    cache_root: Path,
    **_: object,
) -> None:
    recipe = _require_recipe_from_payload(draft.payload, state=state)
    _apply_workspace_patch(
        state,
        {
            "active_recipe_id": recipe.id,
            "stage_model_ids": dict(recipe.stage_model_ids),
            "stage_prompt_ids": dict(recipe.stage_prompt_ids),
        },
        profile_store=profile_store,
        cache_root=cache_root,
    )


def _apply_apply_recipe_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    cache_root: Path,
    **_: object,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    recipe = _require_recipe_from_payload(draft.payload, state=state)
    next_state = _apply_workspace_patch(
        state,
        {
            "active_recipe_id": recipe.id,
            "stage_model_ids": dict(recipe.stage_model_ids),
            "stage_prompt_ids": dict(recipe.stage_prompt_ids),
        },
        profile_store=profile_store,
        cache_root=cache_root,
    )
    return next_state, {"kind": draft.kind, "recipe": recipe.to_dict()}


def _validate_delete_recipe_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    **_: object,
) -> None:
    _require_recipe_from_payload(draft.payload, state=state)


def _apply_delete_recipe_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    **_: object,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    recipe = _require_recipe_from_payload(draft.payload, state=state)
    return state.remove_recipe(recipe.id), {
        "kind": draft.kind,
        "recipe_id": recipe.id,
    }


def _validate_create_model_profile_action(
    draft: AgentActionDraft,
    *,
    profile_store: ModelProfileStore,
    **_: object,
) -> None:
    profile = _model_profile_from_draft_payload(
        draft.payload,
        profile_store=profile_store,
    )
    if profile_store.get(profile.id) is not None:
        raise BridgeError.conflict(f"profile id already exists: {profile.id!r}")


def _apply_create_model_profile_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    **_: object,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    profile = _model_profile_from_draft_payload(
        draft.payload,
        profile_store=profile_store,
    )
    try:
        stored = profile_store.create(profile)
    except ValueError as exc:
        raise BridgeError.conflict(str(exc)) from exc
    return state, {
        "kind": draft.kind,
        "profile": _model_profile_body(stored, profile_store=profile_store),
    }


def _validate_update_model_profile_action(
    draft: AgentActionDraft,
    *,
    profile_store: ModelProfileStore,
    **_: object,
) -> None:
    profile_id, patch, api_keys = _coerce_model_profile_update_payload(draft.payload)
    if profile_store.get(profile_id) is None:
        raise BridgeError.not_found(
            f"profile {profile_id!r} does not exist.",
            details={"id": profile_id},
        )
    if patch:
        _coerce_model_profile_patch(patch)
    if api_keys is not None:
        _coerce_api_keys(api_keys)


def _apply_update_model_profile_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    **_: object,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    profile_id, patch, api_keys = _coerce_model_profile_update_payload(draft.payload)
    stored = profile_store.get(profile_id)
    if stored is None:
        raise BridgeError.not_found(
            f"profile {profile_id!r} does not exist.",
            details={"id": profile_id},
        )
    if patch:
        try:
            stored = profile_store.update(
                profile_id,
                _coerce_model_profile_patch(patch),
            )
        except ValueError as exc:
            raise BridgeError.invalid_argument(str(exc)) from exc
    if api_keys is not None:
        try:
            stored = profile_store.set_api_keys(profile_id, _coerce_api_keys(api_keys))
        except KeyError as exc:
            raise BridgeError.not_found(
                f"profile {profile_id!r} does not exist.",
                details={"id": profile_id},
            ) from exc
    return state, {
        "kind": draft.kind,
        "profile": _model_profile_body(stored, profile_store=profile_store),
    }


def _validate_start_task_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    task_service: TaskService,
    **_: object,
) -> None:
    _raise_if_active_task_locked(state)
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
    _validate_glossary_task_reference(draft, task_service=task_service)


def _apply_start_task_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    cache_root: Path,
    settings_store: SettingsStore,
    task_service: TaskService,
    **_: object,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    _raise_if_active_task_locked(state)
    conversation = _require_active(state)
    result = start_agent_task(
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


def _load_with_reconciled_active_task(
    project_store: AgentProjectStore,
    task_service: TaskService,
    *,
    reconcile_stale: bool = True,
) -> AgentWorkspaceState:
    state = project_store.load()
    if state.active_task is None:
        return state
    if _active_task_is_terminal(
        state.active_task,
        task_service,
        reconcile_stale=reconcile_stale,
    ):
        return project_store.save(state.clear_active_task())
    return state


def _active_task_is_terminal(
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


def _raise_if_active_task_locked(state: AgentWorkspaceState) -> None:
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


def _validate_glossary_task_reference(
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


def _create_prompt_preset(
    payload: Mapping[str, object],
    *,
    cache_root: Path,
) -> PromptPreset:
    kind, name, description, system_prompt, enabled = _coerce_prompt_preset_payload(
        payload
    )
    store = _prompt_store_for(cache_root, kind)
    existing = list(store.load())
    preset_id = _generate_prompt_id(name, kind, existing)
    preset = PromptPreset(
        id=preset_id,
        name=name,
        kind=kind,
        system_prompt=system_prompt,
        suffix_prompt="",
        thinking_prompt="",
        description=description,
        enabled=enabled,
        is_system=False,
    )
    store.save([*existing, preset])
    return preset


def _update_prompt_preset(
    payload: Mapping[str, object],
    *,
    cache_root: Path,
) -> PromptPreset:
    preset_id, patch = _coerce_prompt_update_payload(payload)
    kind, presets, index = _resolve_prompt_for_update(
        preset_id,
        patch,
        cache_root=cache_root,
    )
    updated = replace(presets[index], **patch)
    presets[index] = updated
    _prompt_store_for(cache_root, kind).save(presets)
    return updated


def _coerce_prompt_update_payload(
    payload: Mapping[str, object],
) -> tuple[str, dict[str, object]]:
    preset_id = str(payload.get("id") or payload.get("preset_id") or "").strip()
    if not preset_id:
        raise BridgeError.invalid_argument("id is required.", field="id")
    raw_patch = payload.get("patch")
    if raw_patch is None:
        raw_patch = {
            key: payload[key]
            for key in ("name", "system_prompt", "description", "enabled")
            if key in payload
        }
    if not isinstance(raw_patch, Mapping):
        raise BridgeError.invalid_argument(
            "patch object is required.",
            field="patch",
        )
    patch = _coerce_prompt_patch(raw_patch)
    if not patch:
        raise BridgeError.invalid_argument(
            "patch must include at least one editable field.",
            field="patch",
        )
    return preset_id, patch


def _coerce_prompt_patch(patch: Mapping[str, object]) -> dict[str, object]:
    valid = {"name", "system_prompt", "description", "enabled"}
    coerced: dict[str, object] = {}
    for key, value in patch.items():
        if key not in valid:
            raise BridgeError.invalid_argument(
                f"field {key!r} cannot be updated.",
                field=key,
            )
        if key == "enabled":
            if not isinstance(value, bool):
                raise BridgeError.invalid_argument(
                    "enabled must be a boolean.",
                    field=key,
                )
            coerced[key] = value
            continue
        if not isinstance(value, str):
            raise BridgeError.invalid_argument(f"{key} must be a string.", field=key)
        text = value.strip() if key in {"name", "system_prompt"} else value
        if key in {"name", "system_prompt"} and not text:
            raise BridgeError.invalid_argument(f"{key} must not be empty.", field=key)
        coerced[key] = text
    return coerced


def _resolve_prompt_for_update(
    preset_id: str,
    patch: Mapping[str, object],
    *,
    cache_root: Path,
) -> tuple[PromptKind, list[PromptPreset], int]:
    for kind in PromptKind:
        store = _prompt_store_for(cache_root, kind)
        presets = list(store.load())
        for index, preset in enumerate(presets):
            if preset.id != preset_id:
                continue
            if preset.is_system:
                raise BridgeError.invalid_argument(
                    "system prompt presets are read-only; duplicate to edit.",
                    details={"reason": "is_system"},
                )
            return kind, presets, index
    raise BridgeError.not_found(f"prompt preset {preset_id!r} does not exist.")


def _coerce_prompt_preset_payload(
    payload: Mapping[str, object],
) -> tuple[PromptKind, str, str, str, bool]:
    kind = _coerce_prompt_kind(
        _first_present(
            payload,
            (
                "kind",
                "type",
                "prompt_kind",
                "promptKind",
                "stage",
                "category",
                "类型",
                "类别",
                "用途",
            ),
        ),
        fallback_text=" ".join(
            str(payload.get(key) or "")
            for key in ("name", "description", "system_prompt", "prompt", "content")
        ),
    )
    name = str(
        _first_present(payload, ("name", "display_name", "displayName", "名称", "名字"))
        or ""
    ).strip()
    system_prompt = _prompt_body_from_payload(payload)
    if not name:
        raise BridgeError.invalid_argument("name is required.", field="name")
    if not system_prompt:
        raise BridgeError.invalid_argument(
            "system_prompt is required.",
            field="system_prompt",
        )
    return (
        kind,
        name,
        str(payload.get("description") or "").strip(),
        system_prompt,
        bool(payload.get("enabled", True)),
    )


def _coerce_prompt_kind(value: object, *, fallback_text: str = "") -> PromptKind:
    raw_kind = str(value or "").strip()
    normalized = (
        (raw_kind or fallback_text).lower()
        .replace(" ", "")
        .replace("-", "_")
        .replace("·", "")
        .replace("：", "")
        .replace(":", "")
    )
    kind = _PROMPT_KIND_ALIASES.get(raw_kind) or _PROMPT_KIND_ALIASES.get(normalized)
    if kind is not None:
        return kind
    if any(marker in normalized for marker in ("review", "audit", "审核", "审查")):
        return PromptKind.GLOSSARY_REVIEW
    if any(marker in normalized for marker in ("translation", "translate", "翻译")):
        return PromptKind.TRANSLATION
    if any(marker in normalized for marker in ("glossary", "term", "术语")):
        return PromptKind.GLOSSARY
    raise BridgeError.invalid_argument(
        "prompt kind must be translation, glossary, or glossary_review.",
        field="kind",
    )


def _recipe_body_from_payload(payload: Mapping[str, object]) -> dict[str, object]:
    raw_recipe = _first_present(
        payload,
        ("recipe", "recipe_config", "recipeConfig", "config", "preset", "预设", "配方"),
    )
    body = dict(raw_recipe) if isinstance(raw_recipe, Mapping) else dict(payload)
    if "name" not in body:
        for key in (
            "recipe_name",
            "recipeName",
            "preset_name",
            "presetName",
            "config_name",
            "configName",
            "configuration_name",
            "configurationName",
            "display_name",
            "displayName",
            "title",
            "名称",
            "名字",
            "配方名称",
            "预设名称",
        ):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                body["name"] = value
                break
    return body


def _first_present(payload: Mapping[str, object], keys: tuple[str, ...]) -> object | None:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _prompt_body_from_payload(payload: Mapping[str, object]) -> str:
    for key in ("system_prompt", "systemPrompt", "prompt", "content", "body"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


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
    body = _recipe_body_from_payload(payload)
    recipe_id = str(body.get("recipe_id") or body.get("id") or "").strip()
    if not recipe_id:
        raise BridgeError.invalid_argument(
            "recipe_id is required.",
            field="recipe_id",
        )
    fields: dict[str, str] = {}
    if "name" in body:
        name = str(body.get("name") or "").strip()
        if not name:
            raise BridgeError.invalid_argument("name is required.", field="name")
        if len(name) > _MAX_RECIPE_NAME_LENGTH:
            raise BridgeError.invalid_argument(
                "name is too long.",
                field="name",
                details={"max_length": _MAX_RECIPE_NAME_LENGTH},
            )
        fields["name"] = name
    if "description" in body:
        description = str(body.get("description") or "").strip()
        if len(description) > _MAX_RECIPE_DESCRIPTION_LENGTH:
            raise BridgeError.invalid_argument(
                "description is too long.",
                field="description",
                details={"max_length": _MAX_RECIPE_DESCRIPTION_LENGTH},
            )
        fields["description"] = description
    stage_models = (
        _coerce_model_slots(body.get("stage_model_ids"), profile_store=profile_store)
        if "stage_model_ids" in body
        else None
    )
    stage_prompts = (
        _coerce_prompt_slots(body.get("stage_prompt_ids"), cache_root=cache_root)
        if "stage_prompt_ids" in body
        else None
    )
    if not fields and stage_models is None and stage_prompts is None:
        raise BridgeError.invalid_argument(
            "recipe update must include at least one field.",
            field="recipe_id",
        )
    return recipe_id, fields, stage_models, stage_prompts


def _require_recipe_from_payload(
    payload: Mapping[str, object],
    *,
    state: AgentWorkspaceState,
) -> AgentRecipe:
    recipe_id = str(payload.get("recipe_id") or payload.get("id") or "").strip()
    if not recipe_id:
        raise BridgeError.invalid_argument(
            "recipe_id is required.",
            field="recipe_id",
        )
    recipe = state.get_recipe(recipe_id)
    if recipe is None:
        raise BridgeError.not_found(
            f"recipe {recipe_id!r} does not exist.",
            details={"recipe_id": recipe_id},
        )
    return recipe


def _coerce_recipe_payload(
    payload: Mapping[str, object],
    *,
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> tuple[str, str, dict[str, str | None], dict[str, str | None]]:
    body = _recipe_body_from_payload(payload)
    name = str(body.get("name") or "").strip()
    if not name:
        raise BridgeError.invalid_argument("name is required.", field="name")
    if len(name) > _MAX_RECIPE_NAME_LENGTH:
        raise BridgeError.invalid_argument(
            "name is too long.",
            field="name",
            details={"max_length": _MAX_RECIPE_NAME_LENGTH},
        )
    description = str(body.get("description") or "").strip()
    if len(description) > _MAX_RECIPE_DESCRIPTION_LENGTH:
        raise BridgeError.invalid_argument(
            "description is too long.",
            field="description",
            details={"max_length": _MAX_RECIPE_DESCRIPTION_LENGTH},
        )
    stage_models = (
        _coerce_model_slots(body.get("stage_model_ids"), profile_store=profile_store)
        if "stage_model_ids" in body
        else {}
    )
    stage_prompts = (
        _coerce_prompt_slots(body.get("stage_prompt_ids"), cache_root=cache_root)
        if "stage_prompt_ids" in body
        else {}
    )
    return name, description, stage_models, stage_prompts


def _coerce_memory_items_payload(payload: Mapping[str, object]) -> tuple[str, ...]:
    if "memories" in payload:
        return _coerce_memories_payload(payload)
    memory = str(payload.get("memory") or "").strip()
    if not memory:
        raise BridgeError.invalid_argument("memory is required.", field="memory")
    return _coerce_memories_payload({"memories": [memory]})


def _coerce_memories_payload(payload: Mapping[str, object]) -> tuple[str, ...]:
    memories_raw = payload.get("memories")
    if not isinstance(memories_raw, list):
        raise BridgeError.invalid_argument(
            "memories must be a list.",
            field="memories",
        )
    memories: list[str] = []
    seen: set[str] = set()
    for raw in memories_raw:
        if not isinstance(raw, str):
            raise BridgeError.invalid_argument(
                "each memory must be a string.",
                field="memories",
            )
        memory = raw.strip()
        if not memory:
            continue
        if len(memory) > 600:
            raise BridgeError.invalid_argument(
                "memory is too long.",
                field="memories",
                details={"max_length": 600},
            )
        if memory not in seen:
            memories.append(memory)
            seen.add(memory)
    if len(memories) > 30:
        raise BridgeError.invalid_argument(
            "too many memories.",
            field="memories",
            details={"max_count": 30},
        )
    return tuple(memories)


def _model_profile_from_draft_payload(
    payload: Mapping[str, object],
    *,
    profile_store: ModelProfileStore | None = None,
) -> ModelConfig:
    raw_profile = payload.get("profile")
    body = dict(raw_profile) if isinstance(raw_profile, Mapping) else dict(payload)
    source_profile_id = str(
        body.pop("copy_from_profile_id", None)
        or body.pop("source_profile_id", None)
        or ""
    ).strip()
    copied_api_keys: tuple[str, ...] = ()
    if source_profile_id:
        if profile_store is None:
            raise BridgeError.invalid_argument(
                "copy_from_profile_id requires profile store access.",
                field="copy_from_profile_id",
            )
        source = profile_store.get(source_profile_id)
        if source is None:
            raise BridgeError.not_found(
                f"profile {source_profile_id!r} does not exist.",
                details={"id": source_profile_id},
            )
        source_body = source.to_dict()
        source_body.pop("id", None)
        source_body.pop("api_keys", None)
        source_body.update(body)
        body = source_body
        copied_api_keys = source.api_keys
    body.setdefault("id", _generate_model_profile_id(body))
    for field in ("display_name", "model_id"):
        if _looks_like_placeholder_value(body.get(field)):
            raise BridgeError.invalid_argument(
                f"{field} contains placeholder text.",
                field=field,
            )
    api_keys = body.pop("api_keys", None)
    try:
        profile = ModelConfig.from_dict(body)
    except (KeyError, TypeError, ValueError) as exc:
        raise BridgeError.invalid_argument(str(exc)) from exc
    keys = (
        copied_api_keys
        if api_keys is None and source_profile_id
        else _coerce_api_keys(api_keys)
    )
    return profile.with_api_keys(keys)


def _coerce_model_profile_update_payload(
    payload: Mapping[str, object],
) -> tuple[str, dict[str, object], object | None]:
    profile_id = str(payload.get("profile_id") or payload.get("id") or "").strip()
    if not profile_id:
        raise BridgeError.invalid_argument("profile_id is required.", field="profile_id")
    raw_patch = payload.get("patch")
    if raw_patch is None:
        raw_patch = {
            key: value
            for key, value in payload.items()
            if key not in {"profile_id", "id"}
        }
    if not isinstance(raw_patch, Mapping):
        raise BridgeError.invalid_argument(
            "patch object is required.",
            field="patch",
        )
    patch = dict(raw_patch)
    api_keys = patch.pop("api_keys", None)
    if not patch and api_keys is None:
        raise BridgeError.invalid_argument(
            "profile update must include at least one field.",
            field="patch",
        )
    return profile_id, patch, api_keys


def _coerce_model_profile_patch(patch: Mapping[str, object]) -> dict[str, object]:
    valid_fields = set(ModelConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    unknown = set(patch) - valid_fields
    if unknown:
        raise BridgeError.invalid_argument(
            f"unknown profile field(s): {sorted(unknown)!r}",
            details={"unknown_fields": sorted(unknown)},
        )
    coerced: dict[str, object] = {}
    for key, value in patch.items():
        if key in {"display_name", "model_id"} and _looks_like_placeholder_value(value):
            raise BridgeError.invalid_argument(
                f"{key} contains placeholder text.",
                field=key,
            )
        if key == "provider_format" and isinstance(value, str):
            try:
                coerced[key] = ProviderFormat(value)
            except ValueError as exc:
                raise BridgeError.invalid_argument(str(exc), field=key) from exc
        elif key == "thinking_level" and isinstance(value, str):
            try:
                coerced[key] = ThinkingLevel(value)
            except ValueError as exc:
                raise BridgeError.invalid_argument(str(exc), field=key) from exc
        elif key == "custom_headers" and isinstance(value, list):
            coerced[key] = tuple(
                (str(pair[0]), str(pair[1]))
                for pair in value
                if isinstance(pair, (list, tuple)) and len(pair) == 2
            )
        else:
            coerced[key] = value
    return coerced


def _looks_like_placeholder_value(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    if not normalized:
        return False
    return any(
        marker in normalized
        for marker in (
            "新的值",
            "例如",
            "比如",
            "待填",
            "占位",
            "placeholder",
            "example",
            "your-model",
            "model-id",
            "<model",
        )
    )


def _coerce_api_keys(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise BridgeError.invalid_argument(
            "api_keys must be a list of strings.",
            field="api_keys",
        )
    keys: list[str] = []
    for raw in value:
        if not isinstance(raw, str):
            raise BridgeError.invalid_argument(
                "api_keys must be a list of strings.",
                field="api_keys",
            )
        key = raw.strip()
        if key:
            keys.append(key)
    return tuple(keys)


def _model_profile_body(
    profile: ModelConfig,
    *,
    profile_store: ModelProfileStore,
) -> dict[str, object]:
    body = profile.to_dict()
    body.pop("api_keys", None)
    body["api_key_configured"] = bool(profile.api_keys)
    body["api_key_status"] = profile_store.api_key_status(profile.id)
    return body


def _generate_model_profile_id(body: Mapping[str, object]) -> str:
    seed = str(body.get("display_name") or body.get("model_id") or "profile")
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in seed).strip("-")
    if not slug:
        slug = "profile"
    return f"{slug}-{token_hex(3)}"


def _coerce_model_slots(
    value: object,
    *,
    profile_store: ModelProfileStore,
) -> dict[str, str | None]:
    if not isinstance(value, Mapping):
        raise BridgeError.invalid_argument(
            "stage_model_ids must be an object.",
            field="stage_model_ids",
        )
    result: dict[str, str | None] = {}
    for slot, raw in value.items():
        slot_name = str(slot)
        if slot_name not in MODEL_SLOTS:
            raise BridgeError.invalid_argument(
                f"unknown model slot: {slot_name!r}",
                field="stage_model_ids",
            )
        result[slot_name] = _coerce_model_id(
            raw,
            profile_store=profile_store,
            field=f"stage_model_ids.{slot_name}",
        )
    return result


def _coerce_prompt_slots(value: object, *, cache_root: Path) -> dict[str, str | None]:
    if not isinstance(value, Mapping):
        raise BridgeError.invalid_argument(
            "stage_prompt_ids must be an object.",
            field="stage_prompt_ids",
        )
    result: dict[str, str | None] = {}
    for slot, raw in value.items():
        slot_name = str(slot)
        if slot_name not in PROMPT_SLOTS:
            raise BridgeError.invalid_argument(
                f"unknown prompt slot: {slot_name!r}",
                field="stage_prompt_ids",
            )
        prompt_id = _optional_str(raw)
        if prompt_id is not None:
            kind = _PROMPT_KIND_BY_SLOT[slot_name]
            store = _prompt_store_for(cache_root, kind)
            if not any(p.id == prompt_id for p in store.load()):
                raise BridgeError.not_found(
                    f"prompt preset {prompt_id!r} does not exist.",
                    details={"id": prompt_id, "slot": slot_name},
                )
        result[slot_name] = prompt_id
    return result


def _coerce_workflow_thinking_level(
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


def _profile_for_workflow_chat(
    profile: ModelConfig,
    workflow_thinking_level: str,
) -> ModelConfig:
    level = (
        ThinkingLevel(workflow_thinking_level)
        if workflow_thinking_level in {item.value for item in ThinkingLevel}
        else ThinkingLevel.OFF
    )
    return replace(profile, thinking_level=level)


def _coerce_model_id(
    value: object,
    *,
    profile_store: ModelProfileStore,
    field: str,
) -> str | None:
    profile_id = _optional_str(value)
    if profile_id is None:
        return None
    if profile_store.get(profile_id) is None:
        raise BridgeError.not_found(
            f"profile {profile_id!r} does not exist.",
            details={"id": profile_id, "field": field},
        )
    return profile_id


def _optional_str(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    if value is None:
        return None
    raise BridgeError.invalid_argument("value must be a string or null.")


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
    return {
        "workspace": _workspace_wire(state),
        "inventory": _inventory(profile_store, cache_root),
    }


def _workspace_wire(state: AgentWorkspaceState) -> dict[str, object]:
    """Wire shape: workspace config plus the active conversation hoisted flat."""
    active = state.active()
    return {
        "workflow_model_id": state.workflow_model_id,
        "workflow_thinking_level": state.workflow_thinking_level,
        "active_recipe_id": _resolved_active_recipe_id(state),
        "stage_model_ids": dict(state.stage_model_ids),
        "stage_prompt_ids": dict(state.stage_prompt_ids),
        "memories": list(state.memories),
        "recipes": [recipe.to_dict() for recipe in state.recipes],
        "active_task": state.active_task.to_dict() if state.active_task else None,
        "active_conversation_id": state.active_conversation_id,
        "conversations": [
            _conversation_summary(conversation) for conversation in state.conversations
        ],
        "messages": [message.to_dict() for message in active.messages]
        if active
        else [],
        "pending_draft": _draft_wire(active.pending_draft)
        if active and active.pending_draft
        else None,
        "draft_history": [_draft_wire(draft) for draft in active.draft_history]
        if active
        else [],
        "updated_at": state.updated_at,
    }


def _draft_wire(draft: AgentActionDraft) -> dict[str, object]:
    return {**draft.to_dict(), "payload": _sanitize_draft_payload(draft)}


def _sanitize_draft_payload(draft: AgentActionDraft) -> dict[str, object]:
    payload = _normalized_draft_payload_for_wire(draft)
    return _sanitize_preview_value(payload)  # type: ignore[return-value]


def _normalized_draft_payload_for_wire(
    draft: AgentActionDraft,
) -> Mapping[str, object]:
    if draft.kind != "compound_config_update":
        return draft.payload
    try:
        actions = _coerce_compound_action_drafts(draft.payload)
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


def _sanitize_preview_value(value: object, *, key: str | None = None) -> object:
    if _is_sensitive_key(key):
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
                sanitized[child_key_str] = _sanitize_custom_headers(child_value)
            else:
                sanitized[child_key_str] = _sanitize_preview_value(
                    child_value,
                    key=child_key_str,
                )
        return sanitized
    if isinstance(value, list):
        return [_sanitize_preview_value(item) for item in value]
    return value


def _sanitize_custom_headers(value: list[object]) -> list[object]:
    sanitized: list[object] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            sanitized.append(item)
            continue
        name = str(item[0])
        header_value = "<masked>" if _is_sensitive_key(name) else item[1]
        sanitized.append([name, header_value])
    return sanitized


def _is_sensitive_key(key: str | None) -> bool:
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


def _conversation_summary(conversation: AgentConversation) -> dict[str, object]:
    return {
        "id": conversation.id,
        "title": conversation.title,
        "message_count": len(conversation.messages),
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
    }


def _resolved_active_recipe_id(state: AgentWorkspaceState) -> str | None:
    if state.active_recipe_id and state.get_recipe(state.active_recipe_id):
        return state.active_recipe_id
    return _active_recipe_id(state)


def _active_recipe_id(state: AgentWorkspaceState) -> str | None:
    for recipe in state.recipes:
        if (
            dict(recipe.stage_model_ids) == dict(state.stage_model_ids)
            and dict(recipe.stage_prompt_ids) == dict(state.stage_prompt_ids)
        ):
            return recipe.id
    return None


def _coerce_active_recipe_id(
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


def _optional_agent_string(
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


def _optional_agent_limit(
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


def _validate_agent_task_kind(kind: str) -> None:
    if kind not in _AGENT_TASK_KINDS:
        raise BridgeError.invalid_argument(
            f"unsupported agent task kind: {kind!r}",
            field="kind",
        )


def _task_header_or_none(
    task_service: TaskService,
    kind: str,
    task_id: str,
) -> dict[str, object] | None:
    try:
        record = task_service.cache.load_record(task_id)
    except (TaskNotFoundError, ValueError, OSError):
        return None
    try:
        expected = TaskKind(kind)
    except ValueError:
        return None
    if record.kind is not expected:
        return None
    return {
        "id": record.id,
        "kind": record.kind.value,
        "status": record.status.value,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def _recent_task_summaries(
    task_service: TaskService,
    *,
    kind: str,
    limit: int,
) -> list[dict[str, object]]:
    listing = task_service.list_recent_tasks(kind=kind, limit=limit)
    tasks = listing.get("tasks")
    if not isinstance(tasks, list):
        return []
    summaries: list[dict[str, object]] = []
    for item in tasks:
        if not isinstance(item, Mapping):
            continue
        task_id = str(item.get("id", "")).strip()
        if not task_id:
            continue
        availability = _artifact_availability(
            task_service,
            kind=kind,
            task_id=task_id,
        )
        summaries.append(
            {
                "id": task_id,
                "kind": str(item.get("kind") or kind),
                "status": str(item.get("status") or ""),
                "created_at": str(item.get("created_at") or ""),
                "updated_at": str(item.get("updated_at") or ""),
                "artifact_available": bool(availability["available"]),
                "artifact_keys": list(availability["artifact_keys"]),
            }
        )
    return summaries


def _task_status_context(task_service: TaskService) -> dict[str, object]:
    return {
        "recent_task_summaries": {
            task_kind: _recent_task_headers(
                task_service,
                kind=task_kind,
                limit=3,
            )
            for task_kind in _AGENT_TASK_KINDS
        }
    }


def _recent_task_headers(
    task_service: TaskService,
    *,
    kind: str,
    limit: int,
) -> list[dict[str, object]]:
    listing = task_service.list_recent_tasks(kind=kind, limit=limit)
    tasks = listing.get("tasks")
    if not isinstance(tasks, list):
        return []
    summaries: list[dict[str, object]] = []
    for item in tasks:
        if not isinstance(item, Mapping):
            continue
        task_id = str(item.get("id", "")).strip()
        if not task_id:
            continue
        availability = _artifact_availability(
            task_service,
            kind=kind,
            task_id=task_id,
        )
        summaries.append(
            {
                "id": task_id,
                "kind": str(item.get("kind") or kind),
                "status": str(item.get("status") or ""),
                "created_at": str(item.get("created_at") or ""),
                "updated_at": str(item.get("updated_at") or ""),
                "artifact_available": bool(availability.get("available")),
            }
        )
    return summaries


def _artifact_availability(
    task_service: TaskService,
    *,
    kind: str,
    task_id: str,
) -> dict[str, object]:
    try:
        payload = task_service.read_artifacts(kind=kind, task_id=task_id)
    except BridgeError as exc:
        return {
            "kind": kind,
            "task_id": task_id,
            "available": False,
            "artifact_keys": [],
            "error": {
                "code": exc.code,
                "message": str(exc),
            },
        }
    return {
        "kind": kind,
        "task_id": task_id,
        "available": True,
        "artifact_keys": sorted(str(key) for key in payload.keys()),
    }


def _llm_context(
    state: AgentWorkspaceState,
    *,
    settings_store: SettingsStore,
    task_service: TaskService,
) -> dict[str, object]:
    active = state.active()
    recent = active.messages[-_MAX_CONTEXT_MESSAGES:] if active else ()
    return {
        "workflow_model_id": state.workflow_model_id,
        "workflow_thinking_level": state.workflow_thinking_level,
        "stage_model_ids": dict(state.stage_model_ids),
        "stage_prompt_ids": dict(state.stage_prompt_ids),
        "active_task": state.active_task.to_dict() if state.active_task else None,
        "start_task_completeness": all_start_draft_completeness(state),
        "recent_task_summaries": {
            task_kind: _recent_task_summaries(
                task_service,
                kind=task_kind,
                limit=3,
            )
            for task_kind in _AGENT_TASK_KINDS
        },
        "settings_defaults": _settings_defaults(settings_store),
        "memories": list(state.memories),
        "recipes": [
            {
                "id": recipe.id,
                "name": recipe.name,
                "description": recipe.description,
                "stage_model_ids": dict(recipe.stage_model_ids),
                "stage_prompt_ids": dict(recipe.stage_prompt_ids),
            }
            for recipe in state.recipes
        ],
        "recent_messages": [
            {"role": message.role, "content": message.content} for message in recent
        ],
    }


def _settings_defaults(settings_store: SettingsStore) -> dict[str, object]:
    settings = settings_store.load_all()
    return {
        "translation": {
            "input_dir": settings.translation.input_folder,
            "output_dir": settings.translation.output_folder,
            "source_language": settings.translation.source_language,
            "target_language": settings.translation.target_language,
        },
        "glossary": {
            "input_dir": settings.glossary.input_folder,
            "output_dir": settings.glossary.output_folder,
            "source_language": settings.glossary.source_language,
            "target_language": settings.glossary.target_language,
            "novel_background": settings.glossary.novel_background,
        },
        "glossary_review": {
            "input_dir": settings.glossary_review.input_folder,
            "novel_background": settings.glossary_review.novel_background,
            "output_filename": settings.glossary_review.output_filename,
        },
    }


def _inventory(profile_store: ModelProfileStore, cache_root: Path) -> dict[str, object]:
    profiles = profile_store.load()
    prompt_groups: dict[str, object] = {}
    for kind in PromptKind:
        prompt_groups[kind.value] = [
            _prompt_summary(preset)
            for preset in _prompt_store_for(cache_root, kind).load()
            if preset.enabled
        ]
    return {
        "profiles": [
            _profile_inventory_summary(profile)
            for profile in profiles
            if not _model_profile_has_placeholder_fields(profile)
        ],
        "excluded_profile_count": sum(
            1 for profile in profiles if _model_profile_has_placeholder_fields(profile)
        ),
        "prompts": prompt_groups,
    }


def _profile_inventory_summary(profile: ModelConfig) -> dict[str, object]:
    return {
        "id": profile.id,
        "display_name": profile.display_name,
        "provider_format": profile.provider_format.value,
        "base_url": profile.base_url,
        "model_id": profile.model_id,
        "api_key_configured": bool(profile.api_keys),
        "thinking_level": profile.thinking_level.value,
        "supports_thinking": profile.thinking_level is not ThinkingLevel.OFF,
        "max_output_tokens": profile.max_output_tokens,
        "input_token_limit": profile.input_token_limit,
        "concurrency_limit": profile.concurrency_limit,
        "rpm_limit": profile.rpm_limit,
        "tpm_limit": profile.tpm_limit,
        "retry_attempts": profile.retry_attempts,
    }


def _model_profile_has_placeholder_fields(profile: ModelConfig) -> bool:
    return _looks_like_placeholder_value(
        profile.display_name
    ) or _looks_like_placeholder_value(profile.model_id)


def _prompt_summary(preset: PromptPreset) -> dict[str, object]:
    return {
        "id": preset.id,
        "name": preset.name,
        "kind": preset.kind.value,
        "description": preset.description,
        "is_system": preset.is_system,
    }


def _prompt_body(preset: PromptPreset) -> dict[str, object]:
    return {
        **_prompt_summary(preset),
        "system_prompt": preset.system_prompt,
        "enabled": preset.enabled,
        "is_default": False,
    }


def _prompt_store_for(cache_root: Path, kind: PromptKind) -> PromptPresetStore:
    return PromptPresetStore(
        path=cache_root / f"prompts.{kind.value}.json",
        kind=kind,
    )


def _generate_prompt_id(
    name: str,
    kind: PromptKind,
    existing: list[PromptPreset],
) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")
    if not slug:
        slug = kind.value
    existing_ids = {preset.id for preset in existing}
    while True:
        candidate = f"agent-{kind.value}-{slug}-{token_hex(3)}"
        if candidate not in existing_ids:
            return candidate


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
