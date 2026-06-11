"""Agent Lab action registry, validation, and draft execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

from transoria.agent.completeness import START_DRAFT_KINDS
from transoria.agent.schemas import (
    AgentActionDraft,
    AgentRecipe,
    AgentWorkspaceState,
)
from transoria.bridge.errors import BridgeError
from transoria.bridge.handlers.agent_actions import (
    AgentActionSpec,
    coerce_compound_action_drafts as dispatch_coerce_compound_action_drafts,
    compound_raw_actions as dispatch_compound_raw_actions,
    get_agent_action_spec,
    validate_compound_action_kind as dispatch_validate_compound_action_kind,
    validate_compound_config_action as dispatch_compound_validate,
    validate_draft as dispatch_validate_draft,
)
from transoria.bridge.handlers.agent_inventory import prompt_body
from transoria.bridge.handlers.agent_memory_actions import (
    apply_add_memory_action,
    apply_delete_memory_action,
    apply_update_memory_action,
    coerce_memories_payload,
    coerce_memory_items_payload,
    validate_add_memory_action,
    validate_delete_memory_action,
    validate_update_memory_action,
)
from transoria.bridge.handlers.agent_model_profile_actions import (
    coerce_api_keys,
    coerce_model_profile_patch,
    coerce_model_profile_update_payload,
    model_profile_body,
    model_profile_from_draft_payload,
)
from transoria.bridge.handlers.agent_prompt_actions import (
    coerce_prompt_preset_payload,
    coerce_prompt_update_payload,
    create_prompt_preset,
    resolve_prompt_for_update,
    update_prompt_preset,
)
from transoria.bridge.handlers.agent_recipe_actions import (
    coerce_recipe_payload,
    coerce_recipe_update_payload,
    require_recipe_from_payload,
)
from transoria.bridge.handlers.agent_tasks import (
    apply_start_task_action,
    validate_start_task_action,
)
from transoria.bridge.handlers.agent_workspace import apply_workspace_patch
from transoria.bridge.task_service import TaskService
from transoria.llm.config import ModelConfig
from transoria.model_profiles import ModelProfileStore
from transoria.settings import SettingsStore
from transoria.workflows.agent.task_starts import start_agent_task

TaskStarter = Callable[..., dict[str, object]]

MAX_RECIPES = 50
MAX_COMPOUND_ACTIONS = 8


def apply_action_draft(
    state: AgentWorkspaceState,
    draft: AgentActionDraft,
    *,
    profile_store: ModelProfileStore,
    cache_root: Path,
    settings_store: SettingsStore,
    task_service: TaskService,
    start_task: TaskStarter = start_agent_task,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    spec = action_spec(draft.kind, start_task=start_task)
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


def validate_action_draft(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    cache_root: Path,
    task_service: TaskService,
) -> None:
    dispatch_validate_draft(
        draft,
        specs=action_specs(),
        state=state,
        profile_store=profile_store,
        cache_root=cache_root,
        task_service=task_service,
    )


def append_model_quality_warnings(
    reply: str,
    *,
    draft: AgentActionDraft,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
) -> str:
    warnings = model_quality_warnings(
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


def model_quality_warnings(
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


def action_spec(
    kind: str,
    *,
    start_task: TaskStarter = start_agent_task,
) -> AgentActionSpec:
    return get_agent_action_spec(kind, action_specs(start_task=start_task))


def action_specs(
    *,
    start_task: TaskStarter = start_agent_task,
) -> dict[str, AgentActionSpec]:
    specs = {
        "update_workspace": AgentActionSpec(
            kind="update_workspace",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=_validate_workspace_action,
            apply=_apply_workspace_action,
        ),
        "create_prompt_preset": AgentActionSpec(
            kind="create_prompt_preset",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=_validate_create_prompt_action,
            apply=_apply_create_prompt_action,
        ),
        "update_prompt_preset": AgentActionSpec(
            kind="update_prompt_preset",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=_validate_update_prompt_action,
            apply=_apply_update_prompt_action,
        ),
        "update_memory": AgentActionSpec(
            kind="update_memory",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=_validate_update_memory_action,
            apply=_apply_update_memory_action,
        ),
        "add_memory": AgentActionSpec(
            kind="add_memory",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=_validate_add_memory_action,
            apply=_apply_add_memory_action,
        ),
        "delete_memory": AgentActionSpec(
            kind="delete_memory",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=_validate_delete_memory_action,
            apply=_apply_delete_memory_action,
        ),
        "create_recipe": AgentActionSpec(
            kind="create_recipe",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=_validate_create_recipe_action,
            apply=_apply_create_recipe_action,
        ),
        "update_recipe": AgentActionSpec(
            kind="update_recipe",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=_validate_update_recipe_action,
            apply=_apply_update_recipe_action,
        ),
        "apply_recipe": AgentActionSpec(
            kind="apply_recipe",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=_validate_apply_recipe_action,
            apply=_apply_apply_recipe_action,
        ),
        "delete_recipe": AgentActionSpec(
            kind="delete_recipe",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=_validate_delete_recipe_action,
            apply=_apply_delete_recipe_action,
        ),
        "create_model_profile": AgentActionSpec(
            kind="create_model_profile",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=_validate_create_model_profile_action,
            apply=_apply_create_model_profile_action,
        ),
        "update_model_profile": AgentActionSpec(
            kind="update_model_profile",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=_validate_update_model_profile_action,
            apply=_apply_update_model_profile_action,
        ),
        "compound_config_update": AgentActionSpec(
            kind="compound_config_update",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=_validate_compound_config_action,
            apply=lambda draft, **kwargs: _apply_compound_config_action(
                draft,
                start_task=start_task,
                **kwargs,
            ),
        ),
    }
    for kind in START_DRAFT_KINDS:
        specs[kind] = AgentActionSpec(
            kind=kind,
            mutates=True,
            requires_confirmation=True,
            starts_task=True,
            validate=_validate_start_task_action,
            apply=lambda draft, **kwargs: _apply_start_task_action(
                draft,
                start_task=start_task,
                **kwargs,
            ),
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
    dispatch_compound_validate(
        draft,
        specs=action_specs(),
        max_actions=MAX_COMPOUND_ACTIONS,
        preview_state=_preview_compound_state,
        preview_profile_store=_preview_compound_profile_store,
        state=state,
        profile_store=profile_store,
        cache_root=cache_root,
        settings_store=settings_store,
        task_service=task_service,
    )


def _apply_compound_config_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    cache_root: Path,
    settings_store: SettingsStore,
    task_service: TaskService,
    start_task: TaskStarter = start_agent_task,
    **_: object,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    dispatch_compound_validate(
        draft,
        specs=action_specs(start_task=start_task),
        max_actions=MAX_COMPOUND_ACTIONS,
        preview_state=_preview_compound_state,
        preview_profile_store=_preview_compound_profile_store,
        state=state,
        profile_store=profile_store,
        cache_root=cache_root,
        settings_store=settings_store,
        task_service=task_service,
    )
    next_state = state
    results: list[dict[str, object]] = []
    for inner in coerce_compound_action_drafts(draft.payload):
        spec = action_spec(inner.kind, start_task=start_task)
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


def coerce_compound_action_drafts(
    payload: Mapping[str, object],
) -> list[AgentActionDraft]:
    return dispatch_coerce_compound_action_drafts(
        payload,
        specs=action_specs(),
        max_actions=MAX_COMPOUND_ACTIONS,
    )


def compound_raw_actions(payload: Mapping[str, object]) -> object:
    return dispatch_compound_raw_actions(
        payload,
        specs=action_specs(),
    )


def validate_compound_action_kind(kind: str) -> None:
    dispatch_validate_compound_action_kind(kind, specs=action_specs())


def _preview_compound_state(
    state: AgentWorkspaceState,
    draft: AgentActionDraft,
    *,
    profile_store: ModelProfileStore | "_PreviewModelProfileStore",
    cache_root: Path,
) -> AgentWorkspaceState:
    if draft.kind == "update_workspace":
        return apply_workspace_patch(
            state,
            draft.payload,
            profile_store=profile_store,
            cache_root=cache_root,
        )
    if draft.kind == "update_memory":
        return state.with_memories(coerce_memories_payload(draft.payload))
    if draft.kind == "add_memory":
        memories = list(state.memories)
        for memory in coerce_memory_items_payload(draft.payload):
            if memory not in memories:
                memories.append(memory)
        return state.with_memories(tuple(memories))
    if draft.kind == "delete_memory":
        removals = set(coerce_memory_items_payload(draft.payload))
        return state.with_memories(
            tuple(item for item in state.memories if item not in removals)
        )
    if draft.kind == "create_recipe":
        name, description, stage_models, stage_prompts = coerce_recipe_payload(
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
        recipe_id, fields, stage_models, stage_prompts = coerce_recipe_update_payload(
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
        recipe = require_recipe_from_payload(draft.payload, state=state)
        return apply_workspace_patch(
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
        recipe = require_recipe_from_payload(draft.payload, state=state)
        return state.remove_recipe(recipe.id)
    return state


class _PreviewModelProfileStore:
    """Validation-only overlay for compound drafts."""

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
        profile = model_profile_from_draft_payload(
            draft.payload,
            profile_store=profile_store,  # type: ignore[arg-type]
        )
        return _PreviewModelProfileStore(profile_store, {profile.id: profile})
    if draft.kind != "update_model_profile":
        return profile_store

    profile_id, patch, api_keys = coerce_model_profile_update_payload(draft.payload)
    current = profile_store.get(profile_id)
    if current is None:
        return profile_store
    updated = current
    if patch:
        updated = replace(
            updated,
            **coerce_model_profile_patch(patch),  # type: ignore[arg-type]
        )
    if api_keys is not None:
        updated = updated.with_api_keys(coerce_api_keys(api_keys))
    return _PreviewModelProfileStore(profile_store, {profile_id: updated})


def _validate_workspace_action(
    draft: AgentActionDraft,
    *,
    profile_store: ModelProfileStore,
    cache_root: Path,
    **_: object,
) -> None:
    apply_workspace_patch(
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
    next_state = apply_workspace_patch(
        state,
        draft.payload,
        profile_store=profile_store,
        cache_root=cache_root,
    )
    return next_state, {"kind": draft.kind}


def _validate_create_prompt_action(draft: AgentActionDraft, **_: object) -> None:
    coerce_prompt_preset_payload(draft.payload)


def _apply_create_prompt_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    cache_root: Path,
    **_: object,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    preset = create_prompt_preset(draft.payload, cache_root=cache_root)
    return state, {"kind": draft.kind, "preset": prompt_body(preset)}


def _validate_update_prompt_action(
    draft: AgentActionDraft,
    *,
    cache_root: Path,
    **_: object,
) -> None:
    preset_id, patch = coerce_prompt_update_payload(draft.payload)
    resolve_prompt_for_update(preset_id, patch, cache_root=cache_root)


def _apply_update_prompt_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    cache_root: Path,
    **_: object,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    preset = update_prompt_preset(draft.payload, cache_root=cache_root)
    return state, {"kind": draft.kind, "preset": prompt_body(preset)}


def _validate_update_memory_action(draft: AgentActionDraft, **_: object) -> None:
    validate_update_memory_action(draft)


def _apply_update_memory_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    **_: object,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    return apply_update_memory_action(draft, state=state)


def _validate_add_memory_action(draft: AgentActionDraft, **_: object) -> None:
    validate_add_memory_action(draft)


def _apply_add_memory_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    **_: object,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    return apply_add_memory_action(draft, state=state)


def _validate_delete_memory_action(draft: AgentActionDraft, **_: object) -> None:
    validate_delete_memory_action(draft)


def _apply_delete_memory_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    **_: object,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    return apply_delete_memory_action(draft, state=state)


def _validate_create_recipe_action(
    draft: AgentActionDraft,
    *,
    profile_store: ModelProfileStore,
    cache_root: Path,
    **_: object,
) -> None:
    coerce_recipe_payload(
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
    if len(state.recipes) >= MAX_RECIPES:
        raise BridgeError.invalid_argument(
            "too many recipes.",
            field="recipes",
            details={"max_count": MAX_RECIPES},
        )
    name, description, stage_models, stage_prompts = coerce_recipe_payload(
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
    recipe_id, _, _, _ = coerce_recipe_update_payload(
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
    recipe_id, fields, stage_models, stage_prompts = coerce_recipe_update_payload(
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
    recipe = require_recipe_from_payload(draft.payload, state=state)
    apply_workspace_patch(
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
    recipe = require_recipe_from_payload(draft.payload, state=state)
    next_state = apply_workspace_patch(
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
    require_recipe_from_payload(draft.payload, state=state)


def _apply_delete_recipe_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    **_: object,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    recipe = require_recipe_from_payload(draft.payload, state=state)
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
    profile = model_profile_from_draft_payload(
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
    profile = model_profile_from_draft_payload(
        draft.payload,
        profile_store=profile_store,
    )
    try:
        stored = profile_store.create(profile)
    except ValueError as exc:
        raise BridgeError.conflict(str(exc)) from exc
    return state, {
        "kind": draft.kind,
        "profile": model_profile_body(stored, profile_store=profile_store),
    }


def _validate_update_model_profile_action(
    draft: AgentActionDraft,
    *,
    profile_store: ModelProfileStore,
    **_: object,
) -> None:
    profile_id, patch, api_keys = coerce_model_profile_update_payload(draft.payload)
    if profile_store.get(profile_id) is None:
        raise BridgeError.not_found(
            f"profile {profile_id!r} does not exist.",
            details={"id": profile_id},
        )
    if patch:
        coerce_model_profile_patch(patch)
    if api_keys is not None:
        coerce_api_keys(api_keys)


def _apply_update_model_profile_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    **_: object,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    profile_id, patch, api_keys = coerce_model_profile_update_payload(draft.payload)
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
                coerce_model_profile_patch(patch),
            )
        except ValueError as exc:
            raise BridgeError.invalid_argument(str(exc)) from exc
    if api_keys is not None:
        try:
            stored = profile_store.set_api_keys(profile_id, coerce_api_keys(api_keys))
        except KeyError as exc:
            raise BridgeError.not_found(
                f"profile {profile_id!r} does not exist.",
                details={"id": profile_id},
            ) from exc
    return state, {
        "kind": draft.kind,
        "profile": model_profile_body(stored, profile_store=profile_store),
    }


def _validate_start_task_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    task_service: TaskService,
    **_: object,
) -> None:
    validate_start_task_action(draft, state=state, task_service=task_service)


def _apply_start_task_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    cache_root: Path,
    settings_store: SettingsStore,
    task_service: TaskService,
    start_task: TaskStarter = start_agent_task,
    **_: object,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    return apply_start_task_action(
        draft,
        state=state,
        profile_store=profile_store,
        cache_root=cache_root,
        settings_store=settings_store,
        task_service=task_service,
        start_task=start_task,
    )
