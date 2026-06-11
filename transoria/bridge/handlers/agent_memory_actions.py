"""Memory action helpers for Agent Lab bridge handlers."""

from __future__ import annotations

from typing import Mapping

from transoria.agent.schemas import AgentActionDraft, AgentWorkspaceState
from transoria.bridge.errors import BridgeError


def coerce_memory_items_payload(payload: Mapping[str, object]) -> tuple[str, ...]:
    if "memories" in payload:
        return coerce_memories_payload(payload)
    memory = str(payload.get("memory") or "").strip()
    if not memory:
        raise BridgeError.invalid_argument("memory is required.", field="memory")
    return coerce_memories_payload({"memories": [memory]})


def coerce_memories_payload(payload: Mapping[str, object]) -> tuple[str, ...]:
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


def validate_update_memory_action(draft: AgentActionDraft) -> None:
    coerce_memories_payload(draft.payload)


def apply_update_memory_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    memories = coerce_memories_payload(draft.payload)
    return state.with_memories(memories), {
        "kind": draft.kind,
        "memory_count": len(memories),
    }


def validate_add_memory_action(draft: AgentActionDraft) -> None:
    coerce_memory_items_payload(draft.payload)


def apply_add_memory_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    additions = coerce_memory_items_payload(draft.payload)
    memories = list(state.memories)
    for memory in additions:
        if memory not in memories:
            memories.append(memory)
    next_state = state.with_memories(tuple(memories))
    return next_state, {"kind": draft.kind, "memory_count": len(next_state.memories)}


def validate_delete_memory_action(draft: AgentActionDraft) -> None:
    coerce_memory_items_payload(draft.payload)


def apply_delete_memory_action(
    draft: AgentActionDraft,
    *,
    state: AgentWorkspaceState,
) -> tuple[AgentWorkspaceState, dict[str, object]]:
    removals = set(coerce_memory_items_payload(draft.payload))
    next_state = state.with_memories(
        tuple(item for item in state.memories if item not in removals)
    )
    return next_state, {"kind": draft.kind, "memory_count": len(next_state.memories)}
