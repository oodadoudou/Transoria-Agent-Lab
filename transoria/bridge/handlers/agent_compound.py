"""Compound Agent Lab draft normalization helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from transoria.agent.schemas import AgentActionDraft
from transoria.bridge.errors import BridgeError

ValidateActionKind = Callable[[str], None]

COMPOUND_ACTION_KEYS: tuple[str, ...] = (
    "actions",
    "action",
    "steps",
    "step",
    "drafts",
    "items",
    "operations",
    "operation",
    "changes",
    "config_changes",
    "configChanges",
    "updates",
    "修改",
    "动作",
)


def coerce_compound_action_drafts(
    payload: Mapping[str, object],
    *,
    action_kinds: Sequence[str],
    max_actions: int,
    validate_kind: ValidateActionKind,
) -> list[AgentActionDraft]:
    raw_actions = compound_raw_actions(payload, action_kinds=action_kinds)
    if not isinstance(raw_actions, list):
        raise BridgeError.invalid_argument(
            "compound draft payload.actions must be a list.",
            field="actions",
        )
    if not raw_actions:
        raise BridgeError.invalid_argument(
            "compound draft must contain at least one action.",
            field="actions",
        )
    if len(raw_actions) > max_actions:
        raise BridgeError.invalid_argument(
            "compound draft contains too many actions.",
            field="actions",
            details={"max_count": max_actions},
        )
    drafts: list[AgentActionDraft] = []
    for index, raw in enumerate(raw_actions):
        if not isinstance(raw, Mapping):
            raise BridgeError.invalid_argument(
                "compound action must be an object.",
                field=f"actions[{index}]",
            )
        kind = str(raw.get("kind") or "")
        if kind == "compound_config_update":
            raise BridgeError.invalid_argument(
                "compound actions cannot be nested.",
                field=f"actions[{index}].kind",
            )
        validate_kind(kind)
        action_payload = raw.get("payload")
        if not isinstance(action_payload, Mapping):
            action_payload = {
                key: value
                for key, value in raw.items()
                if key not in {"kind", "title", "summary", "payload"}
            }
        if not isinstance(action_payload, Mapping) or not action_payload:
            raise BridgeError.invalid_argument(
                "compound action payload must be an object.",
                field=f"actions[{index}].payload",
            )
        drafts.append(
            AgentActionDraft.create(
                kind=kind,
                title=str(raw.get("title") or kind),
                summary=str(raw.get("summary") or ""),
                payload=dict(action_payload),
            )
        )
    return drafts


def compound_raw_actions(
    payload: Mapping[str, object],
    *,
    action_kinds: Sequence[str],
) -> object:
    raw_actions = first_present(payload, COMPOUND_ACTION_KEYS)
    if raw_actions is None:
        mapped = compound_actions_from_action_map(payload, action_kinds=action_kinds)
        if mapped:
            return mapped
    if isinstance(raw_actions, Mapping):
        if "kind" in raw_actions:
            return [raw_actions]
        mapped = compound_actions_from_action_map(
            raw_actions,
            action_kinds=action_kinds,
        )
        if mapped:
            return mapped
    return raw_actions


def compound_actions_from_action_map(
    raw_actions: Mapping[str, object],
    *,
    action_kinds: Sequence[str],
) -> list[dict[str, object]]:
    mapped: list[dict[str, object]] = []
    for kind in action_kinds:
        if kind == "compound_config_update":
            continue
        value = raw_actions.get(kind)
        if isinstance(value, Mapping):
            mapped.append({"kind": kind, "payload": dict(value)})
            continue
        if isinstance(value, list):
            for item in value:
                if not isinstance(item, Mapping):
                    continue
                if "kind" in item:
                    mapped.append(dict(item))
                else:
                    mapped.append({"kind": kind, "payload": dict(item)})
    return mapped


def first_present(payload: Mapping[str, object], keys: Sequence[str]) -> object:
    for key in keys:
        if key in payload:
            return payload[key]
    return None
