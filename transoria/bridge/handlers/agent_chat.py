"""Agent Lab chat reply generation."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

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
from transoria.agent.schemas import AgentActionDraft, AgentWorkspaceState
from transoria.bridge.errors import BridgeError
from transoria.bridge.handlers.agent_action_registry import (
    append_model_quality_warnings,
    coerce_compound_action_drafts,
    validate_action_draft,
)
from transoria.bridge.handlers.agent_draft_salvage import salvage_invalid_agent_draft
from transoria.bridge.handlers.agent_intents import INTENT_TASK_START
from transoria.bridge.handlers.agent_inventory import inventory
from transoria.bridge.handlers.agent_response_routing import (
    classify_message_intent,
    direct_response_for_intent,
)
from transoria.bridge.handlers.agent_status_responses import (
    direct_stage_status_response,
)
from transoria.bridge.handlers.agent_task_intents import (
    looks_like_glossary_review_followup_request,
    looks_like_translation_after_review_followup_request,
)
from transoria.bridge.handlers.agent_task_status import task_status_context
from transoria.bridge.handlers.agent_workspace import (
    llm_context,
    profile_for_workflow_chat,
)
from transoria.bridge.handlers.agent_wire import sanitize_draft_payload
from transoria.bridge.task_service import TaskService
from transoria.llm.client import ChatRequest, ChatResponse, LlmClient, LlmRequestError
from transoria.llm.config import ModelConfig
from transoria.model_profiles import ModelProfileStore
from transoria.settings import SettingsStore

LlmClientFactory = Callable[[], LlmClient]
_MAX_CONTEXT_MESSAGES = 20


def generate_reply(
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
    conversation_context = recent_conversation_context(state)
    is_adjust_request = user_message.startswith(
        "The user clicked Adjust on the current pending draft."
    )
    if not is_adjust_request:
        status_state = task_status_context(task_service)
        if not (
            looks_like_glossary_review_followup_request(user_message, status_state)
            or looks_like_translation_after_review_followup_request(
                user_message,
                status_state,
            )
        ):
            direct = direct_stage_status_response(
                user_message=user_message,
                current_state=status_state,
                task_service=task_service,
            )
            if direct is not None:
                return direct
    current_state = llm_context(
        state,
        settings_store=settings_store,
        task_service=task_service,
        max_context_messages=_MAX_CONTEXT_MESSAGES,
    )
    intent = classify_message_intent(user_message)
    if not workflow_model_id:
        if not is_adjust_request and intent.kind == INTENT_TASK_START:
            return (
                "请先选择一个工作模型。启动术语、术语审查或翻译任务时，Agent 需要用工作模型补齐缺失的阶段配置并生成确认草案。",
                None,
            )
        direct = None
        if not is_adjust_request:
            direct = direct_response_for_intent(
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
                    validate_action_draft(
                        draft,
                        state=state,
                        profile_store=profile_store,
                        cache_root=cache_root,
                        task_service=task_service,
                    )
                except BridgeError as exc:
                    return (f"{reply}\n\n（已忽略无法应用的草案：{exc}）".strip(), None)
                reply = append_model_quality_warnings(
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
    agent_inventory = inventory(profile_store, cache_root)
    direct = None
    if not is_adjust_request:
        direct = direct_response_for_intent(
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
                validate_action_draft(
                    draft,
                    state=state,
                    profile_store=profile_store,
                    cache_root=cache_root,
                    task_service=task_service,
                )
            except BridgeError as exc:
                return (f"{reply}\n\n（已忽略无法应用的草案：{exc}）".strip(), None)
            reply = append_model_quality_warnings(
                reply,
                draft=draft,
                state=state,
                profile_store=profile_store,
            )
        return reply, draft
    prompt = build_user_prompt(
        user_message=user_message,
        inventory=agent_inventory,
        current_state=current_state,
        conversation_context=conversation_context,
    )
    model = profile_for_workflow_chat(profile, state.workflow_thinking_level)
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
        response = run_agent_chat_with_schema_fallback(client, request)
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
    reply, draft = parse_or_repair_agent_response(
        client,
        model=model,
        user_message=user_message,
        raw_response=response.content,
    )
    if draft is not None:
        try:
            validate_action_draft(
                draft,
                state=state,
                profile_store=profile_store,
                cache_root=cache_root,
                task_service=task_service,
            )
        except BridgeError as exc:
            repaired = repair_invalid_agent_draft(
                client,
                model=model,
                user_message=user_message,
                reply=reply,
                draft=draft,
                validation_error=str(exc),
                inventory=agent_inventory,
                current_state=current_state,
            )
            if repaired is None:
                salvaged = _salvage_validated_draft(
                    user_message=user_message,
                    reply=reply,
                    draft=draft,
                    current_state=current_state,
                    profile_store=profile_store,
                    cache_root=cache_root,
                    state=state,
                    task_service=task_service,
                )
                if salvaged is not None:
                    draft = salvaged
                    reply = (
                        f"{reply}\n\n我已根据你的原始请求补齐为可确认的配置草案，请检查后再应用。"
                    )
                    reply = append_model_quality_warnings(
                        reply,
                        draft=draft,
                        state=state,
                        profile_store=profile_store,
                    )
                    return reply, draft
                return (f"{reply}\n\n（已忽略无法应用的草案：{exc}）".strip(), None)
            reply, draft = repaired
            try:
                validate_action_draft(
                    draft,
                    state=state,
                    profile_store=profile_store,
                    cache_root=cache_root,
                    task_service=task_service,
                )
            except BridgeError as repaired_exc:
                salvaged = _salvage_validated_draft(
                    user_message=user_message,
                    reply=reply,
                    draft=draft,
                    current_state=current_state,
                    profile_store=profile_store,
                    cache_root=cache_root,
                    state=state,
                    task_service=task_service,
                )
                if salvaged is not None:
                    draft = salvaged
                    reply = (
                        f"{reply}\n\n我已根据你的原始请求补齐为可确认的配置草案，请检查后再应用。"
                    )
                    reply = append_model_quality_warnings(
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
        reply = append_model_quality_warnings(
            reply,
            draft=draft,
            state=state,
            profile_store=profile_store,
        )
    return reply, draft


def run_agent_chat_with_schema_fallback(
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


def parse_or_repair_agent_response(
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
        repaired = run_agent_chat_with_schema_fallback(client, repair_request)
        return parse_agent_response(repaired.content, require_json=True)
    except (AgentResponseParseError, LlmRequestError):
        return (
            "工作模型返回了无法转换为配置草案的内容。请重试，或把这次配置要求拆短一些。",
            None,
        )


def repair_invalid_agent_draft(
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
        repaired = run_agent_chat_with_schema_fallback(client, repair_request)
        repaired_reply, repaired_draft = parse_agent_response(
            repaired.content,
            require_json=True,
        )
    except (AgentResponseParseError, LlmRequestError):
        return None
    if repaired_draft is None:
        return None
    return repaired_reply, repaired_draft


def recent_conversation_context(
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


def draft_revision_prompt(adjustment: str, draft: AgentActionDraft) -> str:
    sanitized = {
        "kind": draft.kind,
        "title": draft.title,
        "summary": draft.summary,
        "payload": sanitize_draft_payload(
            draft,
            normalize_compound_action_drafts=coerce_compound_action_drafts,
        ),
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


def _salvage_validated_draft(
    *,
    user_message: str,
    reply: str,
    draft: AgentActionDraft,
    current_state: Mapping[str, object],
    profile_store: ModelProfileStore,
    cache_root: Path,
    state: AgentWorkspaceState,
    task_service: TaskService,
) -> AgentActionDraft | None:
    salvaged = salvage_invalid_agent_draft(
        user_message=user_message,
        reply=reply,
        draft=draft,
        current_state=current_state,
        profile_store=profile_store,
        cache_root=cache_root,
    )
    if salvaged is None:
        return None
    try:
        validate_action_draft(
            salvaged,
            state=state,
            profile_store=profile_store,
            cache_root=cache_root,
            task_service=task_service,
        )
    except BridgeError:
        return None
    return salvaged
