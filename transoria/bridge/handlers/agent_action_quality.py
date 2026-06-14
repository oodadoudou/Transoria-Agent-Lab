"""Model quality warnings for Agent Lab action drafts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

from transoria.agent.schemas import AgentActionDraft, AgentWorkspaceState
from transoria.bridge.errors import BridgeError
from transoria.bridge.handlers.agent_model_profile_actions import (
    coerce_model_profile_patch,
    coerce_model_profile_update_payload,
    model_profile_from_draft_payload,
)
from transoria.llm.config import ModelConfig
from transoria.model_profiles import ModelProfileStore

CompoundDraftCoercer = Callable[[Mapping[str, object]], list[AgentActionDraft]]


def append_model_quality_warnings(
    reply: str,
    *,
    draft: AgentActionDraft,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    coerce_compound_action_drafts: CompoundDraftCoercer,
) -> str:
    warnings = model_quality_warnings(
        draft,
        state=state,
        profile_store=profile_store,
        coerce_compound_action_drafts=coerce_compound_action_drafts,
    )
    if not warnings:
        return reply
    suffix = "模型风险提示：" + "；".join(warnings)
    if suffix in reply:
        return reply
    return f"{reply}\n\n{suffix}"


def model_quality_warnings(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    coerce_compound_action_drafts: CompoundDraftCoercer,
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
                model_profile_from_draft_payload(
                    draft.payload,
                    profile_store=profile_store,
                ),
            )
        except BridgeError:
            return warnings
    elif draft.kind == "update_model_profile":
        profile_id, patch, _api_keys = coerce_model_profile_update_payload(
            draft.payload
        )
        current = profile_store.get(profile_id)
        if current is not None:
            preview = current
            if patch:
                try:
                    preview = replace(
                        current,
                        **coerce_model_profile_patch(patch),  # type: ignore[arg-type]
                    )
                except BridgeError:
                    preview = current
            add("updated_profile", preview)
    elif draft.kind == "compound_config_update":
        try:
            inner_drafts = coerce_compound_action_drafts(draft.payload)
        except BridgeError:
            return warnings
        for inner in inner_drafts:
            for warning in model_quality_warnings(
                inner,
                state=state,
                profile_store=profile_store,
                coerce_compound_action_drafts=coerce_compound_action_drafts,
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
