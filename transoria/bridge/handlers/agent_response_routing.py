"""Direct response and intent routing for Agent Lab chat."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from transoria.agent.schemas import AgentActionDraft, AgentWorkspaceState
from transoria.bridge.handlers.agent_config_responses import (
    direct_compound_config_response,
    looks_like_direct_compound_config_request as looks_like_compound_config_request,
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
    classify_agent_intent as classify_intent_from_signals,
    direct_intent_sequence,
)
from transoria.bridge.handlers.agent_model_intents import (
    looks_like_model_profile_copy_request,
)
from transoria.bridge.handlers.agent_model_responses import (
    direct_model_profile_copy_response,
    direct_model_upgrade_response,
    direct_vague_model_profile_guidance_response,
    looks_like_model_upgrade_request as looks_like_model_upgrade,
    looks_like_vague_model_profile_request,
)
from transoria.bridge.handlers.agent_prompt_intents import (
    direct_prompt_preset_response,
    direct_prompt_quality_response,
    looks_like_direct_prompt_preset_request,
    looks_like_prompt_quality_request,
)
from transoria.bridge.handlers.agent_task_intents import (
    looks_like_glossary_extraction_request,
    looks_like_glossary_review_request,
    looks_like_task_start_request,
    looks_like_translation_task_request,
)
from transoria.bridge.handlers.agent_task_responses import (
    direct_glossary_extraction_response,
    direct_glossary_review_response,
    direct_translation_response,
)
from transoria.model_profiles import ModelProfileStore

IntentSignal = Callable[[str], bool]


@dataclass(frozen=True)
class IntentRouteSpec:
    kind: str
    signal: IntentSignal
    blocked_by: tuple[str, ...] = ()

    def matches(self, text: str) -> bool:
        return self.signal(text) and not _route_is_blocked(self.kind, text)


def _raw_task_start_request(text: str) -> bool:
    return looks_like_task_start_request(text)


def _raw_compound_config_request(text: str) -> bool:
    return looks_like_compound_config_request(text)


def _raw_prompt_preset_request(text: str) -> bool:
    return looks_like_direct_prompt_preset_request(text)


def _raw_model_profile_guidance_request(text: str) -> bool:
    return looks_like_vague_model_profile_request(text)


def _raw_model_upgrade_request(text: str) -> bool:
    return looks_like_model_upgrade(text)


INTENT_ROUTE_SPECS: tuple[IntentRouteSpec, ...] = (
    IntentRouteSpec(INTENT_TASK_START, _raw_task_start_request),
    IntentRouteSpec(
        INTENT_COMPOUND_CONFIG,
        _raw_compound_config_request,
        blocked_by=(INTENT_TASK_START, INTENT_MODEL_PROFILE_COPY),
    ),
    IntentRouteSpec(INTENT_PROMPT_QUALITY, looks_like_prompt_quality_request),
    IntentRouteSpec(
        INTENT_PROMPT_PRESET,
        _raw_prompt_preset_request,
        blocked_by=(INTENT_TASK_START,),
    ),
    IntentRouteSpec(
        INTENT_MODEL_PROFILE_GUIDANCE,
        _raw_model_profile_guidance_request,
        blocked_by=(INTENT_TASK_START, INTENT_PROMPT_PRESET, INTENT_MODEL_UPGRADE),
    ),
    IntentRouteSpec(
        INTENT_MODEL_PROFILE_COPY,
        looks_like_model_profile_copy_request,
    ),
    IntentRouteSpec(
        INTENT_MODEL_UPGRADE,
        _raw_model_upgrade_request,
        blocked_by=(INTENT_TASK_START, INTENT_PROMPT_PRESET),
    ),
)

_INTENT_ROUTE_SPEC_BY_KIND = {spec.kind: spec for spec in INTENT_ROUTE_SPECS}


def _route_is_blocked(kind: str, text: str) -> bool:
    spec = _INTENT_ROUTE_SPEC_BY_KIND[kind]
    for blocker_kind in spec.blocked_by:
        blocker = _INTENT_ROUTE_SPEC_BY_KIND[blocker_kind]
        if blocker.matches(text):
            return True
    return False


def _route_blocker(kind: str) -> IntentSignal:
    return lambda text: _route_is_blocked(kind, text)


def classify_message_intent(text: str) -> AgentIntent:
    signals = {spec.kind: spec.matches(text) for spec in INTENT_ROUTE_SPECS}
    return classify_intent_from_signals(
        AgentIntentSignals(
            task_start=signals[INTENT_TASK_START],
            compound_config=signals[INTENT_COMPOUND_CONFIG],
            prompt_quality=signals[INTENT_PROMPT_QUALITY],
            prompt_preset=signals[INTENT_PROMPT_PRESET],
            model_profile_guidance=signals[INTENT_MODEL_PROFILE_GUIDANCE],
            model_profile_copy=signals[INTENT_MODEL_PROFILE_COPY],
            model_upgrade=signals[INTENT_MODEL_UPGRADE],
        )
    )


def should_discard_pending_for_new_message(text: str) -> bool:
    return classify_message_intent(text).kind != INTENT_FALLBACK


def direct_response_for_intent(
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
        direct = direct_response_for_kind(
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


def direct_response_for_kind(
    *,
    kind: str,
    user_message: str,
    conversation_context: list[Mapping[str, object]],
    state: AgentWorkspaceState,
    current_state: Mapping[str, object],
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> tuple[str, AgentActionDraft | None] | None:
    responder = _INTENT_RESPONDERS.get(kind)
    if responder is None:
        return None
    return responder(
        user_message=user_message,
        conversation_context=conversation_context,
        state=state,
        current_state=current_state,
        profile_store=profile_store,
        cache_root=cache_root,
    )


def direct_task_start_response(
    *,
    user_message: str,
    conversation_context: list[Mapping[str, object]],
    state: AgentWorkspaceState,
    current_state: Mapping[str, object],
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> tuple[str, AgentActionDraft | None] | None:
    direct = direct_translation_response(
        user_message=user_message,
        conversation_context=conversation_context,
        state=state,
        current_state=current_state,
        profile_store=profile_store,
        cache_root=cache_root,
    )
    if direct is None:
        direct = direct_glossary_review_response(
            user_message=user_message,
            conversation_context=conversation_context,
            state=state,
            current_state=current_state,
            profile_store=profile_store,
            cache_root=cache_root,
        )
    if direct is None:
        direct = direct_glossary_extraction_response(
            user_message=user_message,
            conversation_context=conversation_context,
            state=state,
            current_state=current_state,
            profile_store=profile_store,
            cache_root=cache_root,
        )
    return direct


def _direct_compound_config_response(
    *,
    user_message: str,
    conversation_context: list[Mapping[str, object]],
    state: AgentWorkspaceState,
    current_state: Mapping[str, object],
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> tuple[str, AgentActionDraft | None] | None:
    return direct_compound_config_response(
        user_message=user_message,
        current_state=current_state,
        profile_store=profile_store,
        cache_root=cache_root,
        excluded_request=_route_blocker(INTENT_COMPOUND_CONFIG),
    )


def _direct_prompt_quality_response(
    *,
    user_message: str,
    conversation_context: list[Mapping[str, object]],
    state: AgentWorkspaceState,
    current_state: Mapping[str, object],
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> tuple[str, AgentActionDraft | None] | None:
    return direct_prompt_quality_response(
        user_message=user_message,
        state=state,
        cache_root=cache_root,
    )


def _direct_prompt_preset_response(
    *,
    user_message: str,
    conversation_context: list[Mapping[str, object]],
    state: AgentWorkspaceState,
    current_state: Mapping[str, object],
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> tuple[str, AgentActionDraft | None] | None:
    return direct_prompt_preset_response(
        user_message=user_message,
        excluded_request=_route_blocker(INTENT_PROMPT_PRESET),
    )


def _direct_model_profile_guidance_response(
    *,
    user_message: str,
    conversation_context: list[Mapping[str, object]],
    state: AgentWorkspaceState,
    current_state: Mapping[str, object],
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> tuple[str, AgentActionDraft | None] | None:
    return direct_vague_model_profile_guidance_response(
        user_message=user_message,
        profile_store=profile_store,
        excluded_request=_route_blocker(INTENT_MODEL_PROFILE_GUIDANCE),
    )


def _direct_model_profile_copy_response(
    *,
    user_message: str,
    conversation_context: list[Mapping[str, object]],
    state: AgentWorkspaceState,
    current_state: Mapping[str, object],
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> tuple[str, AgentActionDraft | None] | None:
    return direct_model_profile_copy_response(
        user_message=user_message,
        conversation_context=conversation_context,
        profile_store=profile_store,
    )


def _direct_model_upgrade_response(
    *,
    user_message: str,
    conversation_context: list[Mapping[str, object]],
    state: AgentWorkspaceState,
    current_state: Mapping[str, object],
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> tuple[str, AgentActionDraft | None] | None:
    return direct_model_upgrade_response(
        user_message=user_message,
        state=state,
        profile_store=profile_store,
        excluded_request=_route_blocker(INTENT_MODEL_UPGRADE),
    )


IntentResponder = Callable[
    ...,
    tuple[str, AgentActionDraft | None] | None,
]

_INTENT_RESPONDERS: dict[str, IntentResponder] = {
    INTENT_TASK_START: direct_task_start_response,
    INTENT_COMPOUND_CONFIG: _direct_compound_config_response,
    INTENT_PROMPT_QUALITY: _direct_prompt_quality_response,
    INTENT_PROMPT_PRESET: _direct_prompt_preset_response,
    INTENT_MODEL_PROFILE_GUIDANCE: _direct_model_profile_guidance_response,
    INTENT_MODEL_PROFILE_COPY: _direct_model_profile_copy_response,
    INTENT_MODEL_UPGRADE: _direct_model_upgrade_response,
}


def looks_like_direct_prompt_preset_request_for_agent(text: str) -> bool:
    return _INTENT_ROUTE_SPEC_BY_KIND[INTENT_PROMPT_PRESET].matches(text)


def looks_like_direct_compound_config_request(text: str) -> bool:
    return _INTENT_ROUTE_SPEC_BY_KIND[INTENT_COMPOUND_CONFIG].matches(text)


def looks_like_direct_compound_config_excluded_request(text: str) -> bool:
    return _route_is_blocked(INTENT_COMPOUND_CONFIG, text)


def looks_like_vague_model_profile_request_for_agent(text: str) -> bool:
    return _INTENT_ROUTE_SPEC_BY_KIND[INTENT_MODEL_PROFILE_GUIDANCE].matches(text)


def looks_like_model_upgrade_request_for_agent(text: str) -> bool:
    return _INTENT_ROUTE_SPEC_BY_KIND[INTENT_MODEL_UPGRADE].matches(text)


def looks_like_vague_model_response_excluded_request(text: str) -> bool:
    return _route_is_blocked(INTENT_MODEL_PROFILE_GUIDANCE, text)


def looks_like_base_model_response_excluded_request(text: str) -> bool:
    return _route_is_blocked(INTENT_MODEL_UPGRADE, text)
