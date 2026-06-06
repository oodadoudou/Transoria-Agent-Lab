"""``agent.*`` bridge handlers for the experimental Agent Lab surface."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from pathlib import Path
from secrets import token_hex
from typing import Callable, Mapping

from transoria.agent.configuration_agent import (
    AGENT_SYSTEM_PROMPT,
    build_user_prompt,
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
from transoria.llm.client import ChatRequest, LlmClient, LlmRequestError
from transoria.llm.config import ModelConfig, ProviderFormat, ThinkingLevel
from transoria.model_profiles import ModelProfileStore
from transoria.prompts import PromptKind, PromptPreset, PromptPresetStore
from transoria.runtime.cache import TaskNotFoundError
from transoria.settings import SettingsStore
from transoria.workflows.agent.task_starts import start_agent_task

LlmClientFactory = Callable[[], LlmClient]

_PROMPT_KIND_BY_SLOT = {
    "translation": PromptKind.TRANSLATION,
    "term_extract": PromptKind.GLOSSARY,
    "term_review": PromptKind.GLOSSARY_REVIEW,
}

_MAX_TITLE_LENGTH = 120
_MAX_CONTEXT_MESSAGES = 20
_MAX_RECIPE_NAME_LENGTH = 120
_MAX_RECIPE_DESCRIPTION_LENGTH = 400
_MAX_RECIPES = 50
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
            "active_recipe_id": _active_recipe_id(state),
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

        current = _load_with_reconciled_active_task(project_store, task_service)
        conversation = _require_active(current)
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
            AgentMessage.create("assistant", f"Applied draft: {draft.title}")
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
            return current.replace_recipe(
                recipe.with_updates(
                    name=name,
                    description=description,
                    stage_model_ids=stage_models,
                    stage_prompt_ids=stage_prompts,
                )
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
    inventory = _inventory(profile_store, cache_root)
    prompt = build_user_prompt(
        user_message=user_message,
        inventory=inventory,
        current_state=_llm_context(
            state,
            settings_store=settings_store,
            task_service=task_service,
        ),
    )
    request = ChatRequest(
        model=profile,
        system_prompt=AGENT_SYSTEM_PROMPT,
        user_prompt=prompt,
        temperature=0.2,
        stream=False,
        log_label="agent configuration chat",
    )
    try:
        response = asyncio.run(llm_client_factory().chat(request))
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
    reply, draft = parse_agent_response(response.content)
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
    return state.with_config(
        workflow_model_id=workflow_model_id,
        stage_model_ids=stage_model_ids,
        stage_prompt_ids=stage_prompt_ids,
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
            add("new_profile", _model_profile_from_draft_payload(draft.payload))
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
    profile = _model_profile_from_draft_payload(draft.payload)
    if profile_store.get(profile.id) is not None:
        raise BridgeError.conflict(f"profile id already exists: {profile.id!r}")


def _apply_create_model_profile_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    **_: object,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    profile = _model_profile_from_draft_payload(draft.payload)
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
) -> AgentWorkspaceState:
    state = project_store.load()
    if state.active_task is None:
        return state
    if _active_task_is_terminal(state.active_task, task_service):
        return project_store.save(state.clear_active_task())
    return state


def _active_task_is_terminal(
    active_task: AgentActiveTask,
    task_service: TaskService,
) -> bool:
    try:
        record = task_service.cache.load_record(active_task.task_id)
    except (TaskNotFoundError, ValueError, OSError):
        return True
    return record.status in {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.STOPPED,
    }


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
    raw_kind = str(payload.get("kind") or "")
    try:
        kind = PromptKind(raw_kind)
    except ValueError as exc:
        raise BridgeError.invalid_argument(
            "prompt kind must be translation, glossary, or glossary_review.",
            field="kind",
        ) from exc
    name = str(payload.get("name") or "").strip()
    system_prompt = str(payload.get("system_prompt") or "").strip()
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
    recipe_id = str(payload.get("recipe_id") or payload.get("id") or "").strip()
    if not recipe_id:
        raise BridgeError.invalid_argument(
            "recipe_id is required.",
            field="recipe_id",
        )
    fields: dict[str, str] = {}
    if "name" in payload:
        name = str(payload.get("name") or "").strip()
        if not name:
            raise BridgeError.invalid_argument("name is required.", field="name")
        if len(name) > _MAX_RECIPE_NAME_LENGTH:
            raise BridgeError.invalid_argument(
                "name is too long.",
                field="name",
                details={"max_length": _MAX_RECIPE_NAME_LENGTH},
            )
        fields["name"] = name
    if "description" in payload:
        description = str(payload.get("description") or "").strip()
        if len(description) > _MAX_RECIPE_DESCRIPTION_LENGTH:
            raise BridgeError.invalid_argument(
                "description is too long.",
                field="description",
                details={"max_length": _MAX_RECIPE_DESCRIPTION_LENGTH},
            )
        fields["description"] = description
    stage_models = (
        _coerce_model_slots(payload.get("stage_model_ids"), profile_store=profile_store)
        if "stage_model_ids" in payload
        else None
    )
    stage_prompts = (
        _coerce_prompt_slots(payload.get("stage_prompt_ids"), cache_root=cache_root)
        if "stage_prompt_ids" in payload
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
    name = str(payload.get("name") or "").strip()
    if not name:
        raise BridgeError.invalid_argument("name is required.", field="name")
    if len(name) > _MAX_RECIPE_NAME_LENGTH:
        raise BridgeError.invalid_argument(
            "name is too long.",
            field="name",
            details={"max_length": _MAX_RECIPE_NAME_LENGTH},
        )
    description = str(payload.get("description") or "").strip()
    if len(description) > _MAX_RECIPE_DESCRIPTION_LENGTH:
        raise BridgeError.invalid_argument(
            "description is too long.",
            field="description",
            details={"max_length": _MAX_RECIPE_DESCRIPTION_LENGTH},
        )
    stage_models = (
        _coerce_model_slots(payload.get("stage_model_ids"), profile_store=profile_store)
        if "stage_model_ids" in payload
        else {}
    )
    stage_prompts = (
        _coerce_prompt_slots(payload.get("stage_prompt_ids"), cache_root=cache_root)
        if "stage_prompt_ids" in payload
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


def _model_profile_from_draft_payload(payload: Mapping[str, object]) -> ModelConfig:
    raw_profile = payload.get("profile")
    body = dict(raw_profile) if isinstance(raw_profile, Mapping) else dict(payload)
    body.setdefault("id", _generate_model_profile_id(body))
    api_keys = body.pop("api_keys", ())
    try:
        profile = ModelConfig.from_dict(body)
    except (KeyError, TypeError, ValueError) as exc:
        raise BridgeError.invalid_argument(str(exc)) from exc
    return profile.with_api_keys(_coerce_api_keys(api_keys))


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
    return _sanitize_preview_value(draft.payload)  # type: ignore[return-value]


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


def _active_recipe_id(state: AgentWorkspaceState) -> str | None:
    for recipe in state.recipes:
        if (
            dict(recipe.stage_model_ids) == dict(state.stage_model_ids)
            and dict(recipe.stage_prompt_ids) == dict(state.stage_prompt_ids)
        ):
            return recipe.id
    return None


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
            {
                "id": profile.id,
                "display_name": profile.display_name,
                "provider_format": profile.provider_format.value,
                "model_id": profile.model_id,
                "api_key_configured": bool(profile.api_keys),
                "thinking_level": profile.thinking_level.value,
                "max_output_tokens": profile.max_output_tokens,
                "input_token_limit": profile.input_token_limit,
                "concurrency_limit": profile.concurrency_limit,
                "rpm_limit": profile.rpm_limit,
                "tpm_limit": profile.tpm_limit,
                "retry_attempts": profile.retry_attempts,
            }
            for profile in profiles
        ],
        "prompts": prompt_groups,
    }


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
