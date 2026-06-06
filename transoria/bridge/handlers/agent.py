"""``agent.*`` bridge handlers for the experimental Agent Lab surface."""

from __future__ import annotations

import asyncio
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
from transoria.domain import TaskStatus
from transoria.llm.client import ChatRequest, LlmClient, LlmRequestError
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
        conversation = conversation.archive_pending("applied")
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
        conversation = conversation.archive_pending("discarded")
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
    _validate_draft(
        draft,
        state=state,
        profile_store=profile_store,
        cache_root=cache_root,
        task_service=task_service,
    )
    if draft.kind == "update_workspace":
        next_state = _apply_workspace_patch(
            state,
            draft.payload,
            profile_store=profile_store,
            cache_root=cache_root,
        )
        return next_state, {"kind": draft.kind}
    if draft.kind == "create_prompt_preset":
        preset = _create_prompt_preset(draft.payload, cache_root=cache_root)
        return state, {
            "kind": draft.kind,
            "preset": _prompt_body(preset),
        }
    if draft.kind == "update_memory":
        memories = _coerce_memories_payload(draft.payload)
        next_state = state.with_memories(memories)
        return next_state, {"kind": draft.kind, "memory_count": len(memories)}
    if draft.kind == "create_recipe":
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
        return state.add_recipe(recipe), {
            "kind": draft.kind,
            "recipe": recipe.to_dict(),
        }
    if draft.kind in START_DRAFT_KINDS:
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
    # Unreachable: _validate_draft above rejects unsupported kinds first.
    raise BridgeError.invalid_argument(  # pragma: no cover
        f"unsupported draft kind: {draft.kind!r}",
        details={"kind": draft.kind},
    )


def _validate_draft(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    cache_root: Path,
    task_service: TaskService,
) -> None:
    if draft.kind == "update_workspace":
        _apply_workspace_patch(
            AgentWorkspaceState.empty(),
            draft.payload,
            profile_store=profile_store,
            cache_root=cache_root,
        )
        return
    if draft.kind == "create_prompt_preset":
        _coerce_prompt_preset_payload(draft.payload)
        return
    if draft.kind == "update_memory":
        _coerce_memories_payload(draft.payload)
        return
    if draft.kind == "create_recipe":
        _coerce_recipe_payload(
            draft.payload,
            profile_store=profile_store,
            cache_root=cache_root,
        )
        return
    if draft.kind in START_DRAFT_KINDS:
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
        return
    raise BridgeError.invalid_argument(
        f"unsupported draft kind: {draft.kind!r}",
        details={"kind": draft.kind},
    )


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
        "pending_draft": active.pending_draft.to_dict()
        if active and active.pending_draft
        else None,
        "draft_history": [draft.to_dict() for draft in active.draft_history]
        if active
        else [],
        "updated_at": state.updated_at,
    }


def _conversation_summary(conversation: AgentConversation) -> dict[str, object]:
    return {
        "id": conversation.id,
        "title": conversation.title,
        "message_count": len(conversation.messages),
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
    }


def _llm_context(
    state: AgentWorkspaceState,
    *,
    settings_store: SettingsStore,
) -> dict[str, object]:
    active = state.active()
    recent = active.messages[-_MAX_CONTEXT_MESSAGES:] if active else ()
    return {
        "workflow_model_id": state.workflow_model_id,
        "stage_model_ids": dict(state.stage_model_ids),
        "stage_prompt_ids": dict(state.stage_prompt_ids),
        "active_task": state.active_task.to_dict() if state.active_task else None,
        "start_task_completeness": all_start_draft_completeness(state),
        "settings_defaults": _settings_defaults(settings_store),
        "memories": list(state.memories),
        "recipes": [
            {"id": recipe.id, "name": recipe.name, "description": recipe.description}
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
