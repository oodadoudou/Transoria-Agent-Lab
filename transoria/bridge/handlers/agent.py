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
from transoria.agent.project_store import AgentProjectStore
from transoria.agent.schemas import (
    MODEL_SLOTS,
    PROMPT_SLOTS,
    AgentActionDraft,
    AgentMessage,
    AgentWorkspaceState,
    with_updates,
)
from transoria.bridge.errors import BridgeError
from transoria.bridge.handlers._utils import expect_string
from transoria.bridge.router import BridgeRouter
from transoria.llm.client import ChatRequest, LlmClient, LlmRequestError
from transoria.model_profiles import ModelProfileStore
from transoria.prompts import PromptKind, PromptPreset, PromptPresetStore

LlmClientFactory = Callable[[], LlmClient]

_PROMPT_KIND_BY_SLOT = {
    "translation": PromptKind.TRANSLATION,
    "term_extract": PromptKind.GLOSSARY,
    "term_review": PromptKind.GLOSSARY_REVIEW,
}


def _build_handlers(
    *,
    cache_root: Path,
    project_store: AgentProjectStore,
    profile_store: ModelProfileStore,
    llm_client_factory: LlmClientFactory,
) -> dict[str, object]:
    def read_workspace(_payload: Mapping[str, object]) -> dict[str, object]:
        return _workspace_response(project_store.load(), profile_store, cache_root)

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
        return _workspace_response(state, profile_store, cache_root)

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

        current = project_store.load()
        user_message = AgentMessage.create("user", content)
        state_with_user = with_updates(
            current,
            messages=_trim_messages((*current.messages, user_message)),
        )

        reply, draft = _generate_reply(
            state_with_user,
            user_message=content,
            profile_store=profile_store,
            cache_root=cache_root,
            llm_client_factory=llm_client_factory,
        )
        assistant_message = AgentMessage.create("assistant", reply)
        final_state = with_updates(
            state_with_user,
            messages=_trim_messages((*state_with_user.messages, assistant_message)),
            pending_draft=draft
            if draft is not None
            else state_with_user.pending_draft,
        )
        project_store.save(final_state)
        return _workspace_response(final_state, profile_store, cache_root)

    def apply_draft(payload: Mapping[str, object]) -> dict[str, object]:
        draft_id = expect_string(payload, "draft_id")
        current = project_store.load()
        draft = current.pending_draft
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
        )
        applied_draft = draft.with_status("applied")
        message = AgentMessage.create(
            "assistant",
            f"Applied draft: {draft.title}",
        )
        final_state = with_updates(
            applied_state,
            messages=_trim_messages((*applied_state.messages, message)),
            pending_draft=None,
            draft_history=(*applied_state.draft_history, applied_draft),
        )
        project_store.save(final_state)
        return {
            **_workspace_response(final_state, profile_store, cache_root),
            "result": result,
        }

    def discard_draft(payload: Mapping[str, object]) -> dict[str, object]:
        draft_id = expect_string(payload, "draft_id")
        current = project_store.load()
        draft = current.pending_draft
        if draft is None or draft.id != draft_id or draft.status != "pending":
            raise BridgeError.not_found(
                f"pending draft {draft_id!r} does not exist.",
                details={"draft_id": draft_id},
            )
        discarded = draft.with_status("discarded")
        final_state = with_updates(
            current,
            pending_draft=None,
            draft_history=(*current.draft_history, discarded),
        )
        project_store.save(final_state)
        return _workspace_response(final_state, profile_store, cache_root)

    return {
        "agent.read_workspace": read_workspace,
        "agent.update_workspace": update_workspace,
        "agent.send_message": send_message,
        "agent.apply_draft": apply_draft,
        "agent.discard_draft": discard_draft,
    }


def _generate_reply(
    state: AgentWorkspaceState,
    *,
    user_message: str,
    profile_store: ModelProfileStore,
    cache_root: Path,
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
        current_state=_workspace_dict(state),
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
        _validate_draft(draft, profile_store=profile_store, cache_root=cache_root)
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
    return with_updates(
        state,
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
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    _validate_draft(draft, profile_store=profile_store, cache_root=cache_root)
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
        next_state = with_updates(state, memories=memories)
        return next_state, {"kind": draft.kind, "memory_count": len(memories)}
    raise BridgeError.invalid_argument(
        f"unsupported draft kind: {draft.kind!r}",
        details={"kind": draft.kind},
    )


def _validate_draft(
    draft: AgentActionDraft,
    *,
    profile_store: ModelProfileStore,
    cache_root: Path,
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
    raise BridgeError.invalid_argument(
        f"unsupported draft kind: {draft.kind!r}",
        details={"kind": draft.kind},
    )


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


def _workspace_response(
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> dict[str, object]:
    return {
        "workspace": _workspace_dict(state),
        "inventory": _inventory(profile_store, cache_root),
    }


def _workspace_dict(state: AgentWorkspaceState) -> dict[str, object]:
    return state.to_dict()


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


def _trim_messages(messages: tuple[AgentMessage, ...]) -> tuple[AgentMessage, ...]:
    return messages[-80:]


def register(
    router: BridgeRouter,
    *,
    cache_root: Path,
    profile_store: ModelProfileStore,
    llm_client_factory: LlmClientFactory,
) -> None:
    handlers = _build_handlers(
        cache_root=cache_root,
        project_store=AgentProjectStore.from_cache_root(cache_root),
        profile_store=profile_store,
        llm_client_factory=llm_client_factory,
    )
    for method, handler in handlers.items():
        router.register(method, handler)  # type: ignore[arg-type]


__all__ = ["register"]
