from __future__ import annotations

import pytest

from transoria.agent.schemas import AgentActionDraft, AgentWorkspaceState
from transoria.bridge.errors import BridgeError
from transoria.bridge.handlers.agent_memory_actions import (
    apply_add_memory_action,
    apply_delete_memory_action,
    apply_update_memory_action,
    coerce_memories_payload,
)


def _draft(kind: str, payload: dict[str, object]) -> AgentActionDraft:
    return AgentActionDraft.create(
        kind=kind,
        title="Memory",
        summary="Update memory",
        payload=payload,
    )


def test_coerce_memories_payload_trims_and_deduplicates() -> None:
    assert coerce_memories_payload({"memories": ["  A  ", "", "A", "B"]}) == (
        "A",
        "B",
    )


def test_coerce_memories_payload_rejects_too_long_memory() -> None:
    with pytest.raises(BridgeError):
        coerce_memories_payload({"memories": ["x" * 601]})


def test_apply_update_memory_replaces_memories() -> None:
    state = AgentWorkspaceState.empty().with_memories(("old",))

    next_state, result = apply_update_memory_action(
        _draft("update_memory", {"memories": ["new"]}),
        state=state,
    )

    assert next_state.memories == ("new",)
    assert result["memory_count"] == 1


def test_apply_add_memory_appends_without_duplicates() -> None:
    state = AgentWorkspaceState.empty().with_memories(("A",))

    next_state, result = apply_add_memory_action(
        _draft("add_memory", {"memories": ["A", "B"]}),
        state=state,
    )

    assert next_state.memories == ("A", "B")
    assert result["memory_count"] == 2


def test_apply_delete_memory_removes_matching_items() -> None:
    state = AgentWorkspaceState.empty().with_memories(("A", "B"))

    next_state, result = apply_delete_memory_action(
        _draft("delete_memory", {"memory": "A"}),
        state=state,
    )

    assert next_state.memories == ("B",)
    assert result["memory_count"] == 1
