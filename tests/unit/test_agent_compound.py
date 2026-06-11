from __future__ import annotations

import pytest

from transoria.bridge.errors import BridgeError
from transoria.bridge.handlers.agent_compound import coerce_compound_action_drafts


ACTION_KINDS = (
    "update_model_profile",
    "update_prompt_preset",
    "create_recipe",
    "compound_config_update",
)


def _validate_kind(kind: str) -> None:
    if kind not in ACTION_KINDS:
        raise BridgeError.invalid_argument(f"unsupported draft kind: {kind!r}")


def test_coerce_compound_accepts_top_level_action_map() -> None:
    drafts = coerce_compound_action_drafts(
        {
            "update_model_profile": {"profile_id": "deepseek-f", "patch": {}},
            "update_prompt_preset": {"preset_id": "default", "patch": {}},
        },
        action_kinds=ACTION_KINDS,
        max_actions=8,
        validate_kind=_validate_kind,
    )

    assert [draft.kind for draft in drafts] == [
        "update_model_profile",
        "update_prompt_preset",
    ]
    assert drafts[0].payload["profile_id"] == "deepseek-f"


def test_coerce_compound_accepts_steps_alias_and_grouped_action_map() -> None:
    drafts = coerce_compound_action_drafts(
        {
            "steps": {
                "update_model_profile": [
                    {"profile_id": "a", "patch": {}},
                    {"profile_id": "b", "patch": {}},
                ],
                "create_recipe": {"name": "测试复合配置"},
            }
        },
        action_kinds=ACTION_KINDS,
        max_actions=8,
        validate_kind=_validate_kind,
    )

    assert [draft.kind for draft in drafts] == [
        "update_model_profile",
        "update_model_profile",
        "create_recipe",
    ]
    assert drafts[2].payload["name"] == "测试复合配置"


def test_coerce_compound_rejects_nested_compound() -> None:
    with pytest.raises(BridgeError):
        coerce_compound_action_drafts(
            {"actions": [{"kind": "compound_config_update", "payload": {}}]},
            action_kinds=ACTION_KINDS,
            max_actions=8,
            validate_kind=_validate_kind,
        )


def test_coerce_compound_rejects_too_many_actions() -> None:
    with pytest.raises(BridgeError):
        coerce_compound_action_drafts(
            {
                "actions": [
                    {"kind": "update_model_profile", "payload": {"profile_id": str(i)}}
                    for i in range(3)
                ]
            },
            action_kinds=ACTION_KINDS,
            max_actions=2,
            validate_kind=_validate_kind,
        )


def test_coerce_compound_rejects_unknown_action() -> None:
    with pytest.raises(BridgeError):
        coerce_compound_action_drafts(
            {"actions": [{"kind": "unknown", "payload": {"id": "x"}}]},
            action_kinds=ACTION_KINDS,
            max_actions=8,
            validate_kind=_validate_kind,
        )
