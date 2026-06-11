from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from transoria.agent.schemas import (
    AgentActionDraft,
    AgentConversation,
    AgentMessage,
    AgentRecipe,
    AgentWorkspaceState,
)
from transoria.bridge.errors import BridgeError
from transoria.bridge.handlers.agent_wire import (
    resolved_active_recipe_id,
    sanitize_draft_payload,
    sanitize_preview_value,
    workspace_wire,
)


def test_sanitize_preview_value_masks_sensitive_fields() -> None:
    payload = {
        "api_key": "secret",
        "api_keys": ["first", "", None, "second"],
        "custom_headers": [
            ["Authorization", "Bearer secret"],
            ["X-Name", "public"],
            ["x-api-key", "secret"],
        ],
        "nested": {"proxy-authorization": "proxy-secret", "name": "visible"},
    }

    sanitized = sanitize_preview_value(payload)

    assert sanitized == {
        "api_key": "<masked>",
        "api_keys": ["<masked>", "<masked>"],
        "custom_headers": [
            ["Authorization", "<masked>"],
            ["X-Name", "public"],
            ["x-api-key", "<masked>"],
        ],
        "nested": {"proxy-authorization": "<masked>", "name": "visible"},
    }


def test_sanitize_draft_payload_normalizes_compound_actions() -> None:
    draft = AgentActionDraft.create(
        kind="compound_config_update",
        title="批量修改配置",
        summary="",
        payload={"actions": [{"kind": "create_model_profile"}]},
    )

    def normalize(
        payload: Mapping[str, object],
    ) -> list[AgentActionDraft]:
        assert payload == {"actions": [{"kind": "create_model_profile"}]}
        return [
            AgentActionDraft.create(
                kind="create_model_profile",
                title="创建模型",
                summary="复制 DeepSeek-f",
                payload={
                    "profile": {
                        "display_name": "DeepSeek 4 Pro",
                        "api_key": "secret",
                    }
                },
            )
        ]

    sanitized = sanitize_draft_payload(
        draft,
        normalize_compound_action_drafts=normalize,
    )

    assert sanitized == {
        "actions": [
            {
                "kind": "create_model_profile",
                "title": "创建模型",
                "summary": "复制 DeepSeek-f",
                "payload": {
                    "profile": {
                        "display_name": "DeepSeek 4 Pro",
                        "api_key": "<masked>",
                    }
                },
            }
        ]
    }


def test_sanitize_draft_payload_falls_back_when_compound_payload_is_invalid() -> None:
    draft = AgentActionDraft.create(
        kind="compound_config_update",
        title="坏草案",
        summary="",
        payload={"api_key": "secret", "actions": "bad"},
    )

    def normalize(payload: Mapping[str, object]) -> list[AgentActionDraft]:
        raise BridgeError.invalid_argument("compound draft payload.actions must be a list.")

    sanitized = sanitize_draft_payload(
        draft,
        normalize_compound_action_drafts=normalize,
    )

    assert sanitized == {"api_key": "<masked>", "actions": "bad"}


def test_workspace_wire_hoists_active_conversation_and_pending_draft() -> None:
    recipe = AgentRecipe.create(
        name="测试预设",
        stage_model_ids={
            "translation": "model-a",
            "term_extract": None,
            "term_review": None,
        },
        stage_prompt_ids={
            "translation": "prompt-a",
            "term_extract": None,
            "term_review": None,
        },
    )
    draft = AgentActionDraft.create(
        kind="create_prompt_preset",
        title="创建提示词",
        summary="",
        payload={"kind": "translation", "name": "测试", "api_key": "secret"},
    )
    conversation = AgentConversation.create(
        title="对话",
        messages=(AgentMessage.create("user", "hi"),),
    ).with_pending_draft(draft)
    state = AgentWorkspaceState(
        workflow_model_id="workflow-model",
        workflow_thinking_level="high",
        active_recipe_id="missing-recipe",
        stage_model_ids=recipe.stage_model_ids,
        stage_prompt_ids=recipe.stage_prompt_ids,
        memories=("偏好短句",),
        recipes=(recipe,),
        conversations=(conversation,),
        active_conversation_id=conversation.id,
    )

    wire = workspace_wire(
        state,
        normalize_compound_action_drafts=lambda payload: [],
    )

    assert wire["workflow_model_id"] == "workflow-model"
    assert wire["workflow_thinking_level"] == "high"
    assert wire["active_recipe_id"] == recipe.id
    assert wire["memories"] == ["偏好短句"]
    assert wire["messages"] == [conversation.messages[0].to_dict()]
    assert wire["conversations"] == [
        {
            "id": conversation.id,
            "title": "对话",
            "message_count": 1,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
        }
    ]
    assert wire["pending_draft"] == {
        **draft.to_dict(),
        "payload": {
            "kind": "translation",
            "name": "测试",
            "api_key": "<masked>",
        },
    }


def test_resolved_active_recipe_id_uses_explicit_valid_recipe_first() -> None:
    matching = AgentRecipe.create(
        name="匹配预设",
        stage_model_ids={"translation": "model-a"},
        stage_prompt_ids={"translation": "prompt-a"},
    )
    explicit = AgentRecipe.create(name="显式预设")
    state = AgentWorkspaceState(
        active_recipe_id=explicit.id,
        stage_model_ids=matching.stage_model_ids,
        stage_prompt_ids=matching.stage_prompt_ids,
        recipes=(matching, explicit),
    )

    assert resolved_active_recipe_id(state) == explicit.id


def test_resolved_active_recipe_id_returns_none_when_no_recipe_matches() -> None:
    state = replace(
        AgentWorkspaceState.empty(),
        stage_model_ids={"translation": "model-a"},
        stage_prompt_ids={"translation": "prompt-a"},
        recipes=(),
    )

    assert resolved_active_recipe_id(state) is None
