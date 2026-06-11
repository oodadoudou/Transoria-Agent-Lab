from __future__ import annotations

from pathlib import Path

import pytest

from transoria.agent.schemas import AgentActionDraft, AgentWorkspaceState
from transoria.bridge.errors import BridgeError
from transoria.bridge.handlers.agent_actions import (
    AgentActionSpec,
    apply_compound_config_action,
    apply_draft,
    validate_compound_config_action,
    validate_draft,
)


def test_apply_draft_validates_then_applies() -> None:
    events: list[str] = []

    def validate(draft: AgentActionDraft, **_: object) -> None:
        events.append(f"validate:{draft.kind}")

    def apply(
        draft: AgentActionDraft,
        *,
        state: AgentWorkspaceState,
        **_: object,
    ) -> tuple[AgentWorkspaceState, dict[str, object]]:
        events.append(f"apply:{draft.kind}")
        return state.with_memories(("applied",)), {"kind": draft.kind}

    specs = {
        "update_memory": AgentActionSpec(
            kind="update_memory",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=validate,
            apply=apply,
        )
    }
    draft = AgentActionDraft.create(
        kind="update_memory",
        title="Update memory",
        summary="",
        payload={"memories": ["applied"]},
    )

    next_state, result = apply_draft(
        AgentWorkspaceState.empty(),
        draft,
        specs=specs,
        profile_store=object(),  # type: ignore[arg-type]
        cache_root=Path("."),
        settings_store=object(),  # type: ignore[arg-type]
        task_service=object(),  # type: ignore[arg-type]
    )

    assert events == ["validate:update_memory", "apply:update_memory"]
    assert next_state.memories == ("applied",)
    assert result == {"kind": "update_memory"}


def test_validate_draft_rejects_unknown_action() -> None:
    with pytest.raises(BridgeError):
        validate_draft(
            AgentActionDraft.create(
                kind="unknown",
                title="Unknown",
                summary="",
                payload={"id": "x"},
            ),
            specs={},
            state=AgentWorkspaceState.empty(),
            profile_store=object(),  # type: ignore[arg-type]
            cache_root=Path("."),
            task_service=object(),  # type: ignore[arg-type]
        )


def test_compound_validation_uses_preview_state_between_actions() -> None:
    def validate_set(draft: AgentActionDraft, **_: object) -> None:
        if not draft.payload.get("value"):
            raise BridgeError.invalid_argument("value is required")

    def validate_use(
        draft: AgentActionDraft,
        *,
        state: AgentWorkspaceState,
        **_: object,
    ) -> None:
        if draft.payload.get("value") not in state.memories:
            raise BridgeError.invalid_argument("value must be prepared first")

    def apply_noop(
        draft: AgentActionDraft,
        *,
        state: AgentWorkspaceState,
        **_: object,
    ) -> tuple[AgentWorkspaceState, dict[str, object]]:
        return state, {"kind": draft.kind}

    specs = {
        "set_value": AgentActionSpec(
            kind="set_value",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=validate_set,
            apply=apply_noop,
        ),
        "use_value": AgentActionSpec(
            kind="use_value",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=validate_use,
            apply=apply_noop,
        ),
    }
    draft = AgentActionDraft.create(
        kind="compound_config_update",
        title="Compound",
        summary="",
        payload={
            "actions": [
                {"kind": "set_value", "payload": {"value": "ready"}},
                {"kind": "use_value", "payload": {"value": "ready"}},
            ]
        },
    )

    validate_compound_config_action(
        draft,
        specs=specs,
        max_actions=8,
        preview_state=lambda state, inner, **_: state.with_memories(
            (str(inner.payload["value"]),)
        )
        if inner.kind == "set_value"
        else state,
        preview_profile_store=lambda profile_store, _inner: profile_store,
        state=AgentWorkspaceState.empty(),
        profile_store=object(),  # type: ignore[arg-type]
        cache_root=Path("."),
        task_service=object(),  # type: ignore[arg-type]
    )


def test_apply_compound_returns_ordered_results() -> None:
    def validate(_draft: AgentActionDraft, **_: object) -> None:
        return None

    def apply(
        draft: AgentActionDraft,
        *,
        state: AgentWorkspaceState,
        **_: object,
    ) -> tuple[AgentWorkspaceState, dict[str, object]]:
        return state.with_memories((*state.memories, draft.kind)), {
            "kind": draft.kind
        }

    specs = {
        "first": AgentActionSpec(
            kind="first",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=validate,
            apply=apply,
        ),
        "second": AgentActionSpec(
            kind="second",
            mutates=True,
            requires_confirmation=True,
            starts_task=False,
            validate=validate,
            apply=apply,
        ),
    }
    draft = AgentActionDraft.create(
        kind="compound_config_update",
        title="Compound",
        summary="",
        payload={
            "actions": [
                {"kind": "first", "payload": {"value": 1}},
                {"kind": "second", "payload": {"value": 2}},
            ]
        },
    )

    next_state, result = apply_compound_config_action(
        draft,
        specs=specs,
        max_actions=8,
        preview_state=lambda state, _inner, **_: state,
        preview_profile_store=lambda profile_store, _inner: profile_store,
        state=AgentWorkspaceState.empty(),
        profile_store=object(),  # type: ignore[arg-type]
        cache_root=Path("."),
        settings_store=object(),  # type: ignore[arg-type]
        task_service=object(),  # type: ignore[arg-type]
    )

    assert next_state.memories == ("first", "second")
    assert result == {
        "kind": "compound_config_update",
        "results": [{"kind": "first"}, {"kind": "second"}],
    }
