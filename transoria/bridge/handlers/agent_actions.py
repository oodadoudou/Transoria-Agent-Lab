"""Agent Lab action registry dispatch helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from transoria.agent.schemas import AgentActionDraft, AgentWorkspaceState
from transoria.bridge.errors import BridgeError
from transoria.bridge.handlers.agent_compound import (
    coerce_compound_action_drafts as normalize_compound_action_drafts,
    compound_raw_actions as normalized_compound_raw_actions,
)
from transoria.bridge.task_service import TaskService
from transoria.model_profiles import ModelProfileStore
from transoria.settings import SettingsStore


@dataclass(frozen=True)
class AgentActionSpec:
    kind: str
    mutates: bool
    requires_confirmation: bool
    starts_task: bool
    validate: Callable[..., None]
    apply: Callable[..., tuple[AgentWorkspaceState, dict[str, object]]]


ActionSpecs = Mapping[str, AgentActionSpec]
PreviewState = Callable[..., AgentWorkspaceState]
PreviewProfileStore = Callable[..., object]


def get_agent_action_spec(kind: str, specs: ActionSpecs) -> AgentActionSpec:
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


def validate_draft(
    draft: AgentActionDraft,
    *,
    specs: ActionSpecs,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    cache_root: Path,
    task_service: TaskService,
) -> None:
    spec = get_agent_action_spec(draft.kind, specs)
    spec.validate(
        draft,
        state=state,
        profile_store=profile_store,
        cache_root=cache_root,
        task_service=task_service,
    )


def apply_draft(
    state: AgentWorkspaceState,
    draft: AgentActionDraft,
    *,
    specs: ActionSpecs,
    profile_store: ModelProfileStore,
    cache_root: Path,
    settings_store: SettingsStore,
    task_service: TaskService,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    spec = get_agent_action_spec(draft.kind, specs)
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


def coerce_compound_action_drafts(
    payload: Mapping[str, object],
    *,
    specs: ActionSpecs,
    max_actions: int,
) -> list[AgentActionDraft]:
    return normalize_compound_action_drafts(
        payload,
        action_kinds=tuple(specs),
        max_actions=max_actions,
        validate_kind=lambda kind: validate_compound_action_kind(kind, specs=specs),
    )


def compound_raw_actions(
    payload: Mapping[str, object],
    *,
    specs: ActionSpecs,
) -> object:
    return normalized_compound_raw_actions(
        payload,
        action_kinds=tuple(specs),
    )


def validate_compound_action_kind(kind: str, *, specs: ActionSpecs) -> None:
    spec = get_agent_action_spec(kind, specs)
    if spec.mutates and not spec.requires_confirmation:  # pragma: no cover
        raise BridgeError.invalid_argument(
            f"mutating action must require confirmation: {kind!r}",
            details={"kind": kind},
        )


def validate_compound_config_action(
    draft: AgentActionDraft,
    *,
    specs: ActionSpecs,
    max_actions: int,
    preview_state: PreviewState,
    preview_profile_store: PreviewProfileStore,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    cache_root: Path,
    settings_store: SettingsStore | None = None,
    task_service: TaskService,
) -> None:
    validation_state = state
    validation_profile_store: object = profile_store
    for inner in coerce_compound_action_drafts(
        draft.payload,
        specs=specs,
        max_actions=max_actions,
    ):
        spec = get_agent_action_spec(inner.kind, specs)
        spec.validate(
            inner,
            state=validation_state,
            profile_store=validation_profile_store,
            cache_root=cache_root,
            settings_store=settings_store,
            task_service=task_service,
        )
        validation_state = preview_state(
            validation_state,
            inner,
            profile_store=validation_profile_store,
            cache_root=cache_root,
        )
        validation_profile_store = preview_profile_store(
            validation_profile_store,
            inner,
        )


def apply_compound_config_action(
    draft: AgentActionDraft,
    *,
    specs: ActionSpecs,
    max_actions: int,
    preview_state: PreviewState,
    preview_profile_store: PreviewProfileStore,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    cache_root: Path,
    settings_store: SettingsStore,
    task_service: TaskService,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    validate_compound_config_action(
        draft,
        specs=specs,
        max_actions=max_actions,
        preview_state=preview_state,
        preview_profile_store=preview_profile_store,
        state=state,
        profile_store=profile_store,
        cache_root=cache_root,
        settings_store=settings_store,
        task_service=task_service,
    )
    next_state = state
    results: list[dict[str, object]] = []
    for inner in coerce_compound_action_drafts(
        draft.payload,
        specs=specs,
        max_actions=max_actions,
    ):
        spec = get_agent_action_spec(inner.kind, specs)
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
