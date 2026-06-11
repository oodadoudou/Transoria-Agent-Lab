from __future__ import annotations

from pathlib import Path

from transoria.agent.schemas import AgentWorkspaceState
from transoria.bridge.handlers.agent_inventory import prompt_store_for
from transoria.bridge.handlers.agent_prompt_intents import (
    direct_prompt_preset_response,
    direct_prompt_quality_response,
    extract_prompt_create_name,
    looks_like_direct_prompt_preset_request,
    quality_issue_markers,
)
from transoria.prompts import PromptKind, PromptPreset, default_preset


def test_direct_prompt_preset_response_builds_confirmable_draft() -> None:
    response = direct_prompt_preset_response(
        user_message=(
            "请新建一套术语审查 Prompt，名字叫「ABO 审查」，"
            "要求重点检查人名、组织名一致性和 ABO 设定继承。"
        )
    )

    assert response is not None
    reply, draft = response
    assert "确认后才会写入" in reply
    assert draft is not None
    assert draft.kind == "create_prompt_preset"
    assert draft.payload["kind"] == "glossary_review"
    assert draft.payload["name"] == "ABO 审查"
    assert "ABO" in str(draft.payload["system_prompt"])


def test_prompt_preset_intent_can_be_excluded_by_task_routing() -> None:
    text = (
        "请处理术语 workflow，输入目录 /tmp/book，输出目录 /tmp/book-out，"
        "背景是现代 BL。"
    )

    assert not looks_like_direct_prompt_preset_request(
        text,
        excluded_request=lambda _text: True,
    )
    assert direct_prompt_preset_response(
        user_message=text,
        excluded_request=lambda _text: True,
    ) is None


def test_prompt_quality_response_updates_selected_custom_prompt(tmp_path: Path) -> None:
    custom = PromptPreset(
        id="custom-translation",
        name="自定义翻译",
        kind=PromptKind.TRANSLATION,
        system_prompt="原始规则。",
        description="",
        enabled=True,
        is_system=False,
    )
    prompt_store_for(tmp_path, PromptKind.TRANSLATION).save(
        [default_preset(PromptKind.TRANSLATION), custom]
    )
    state = AgentWorkspaceState.empty().with_config(
        stage_prompt_ids={"translation": custom.id}
    )

    response = direct_prompt_quality_response(
        user_message="翻译效果不好，人名不一致，还有源文残留，请优化翻译 Prompt。",
        state=state,
        cache_root=tmp_path,
    )

    assert response is not None
    reply, draft = response
    assert "追加质量修正规则" in reply
    assert draft is not None
    assert draft.kind == "update_prompt_preset"
    assert draft.payload["id"] == custom.id
    patch = draft.payload["patch"]
    assert isinstance(patch, dict)
    assert "人名不一致" in str(patch["system_prompt"])
    assert "源文残留" in str(patch["system_prompt"])


def test_prompt_quality_response_creates_editable_copy_for_system_default(
    tmp_path: Path,
) -> None:
    prompt_store_for(tmp_path, PromptKind.TRANSLATION).save(
        [default_preset(PromptKind.TRANSLATION)]
    )
    state = AgentWorkspaceState.empty()

    response = direct_prompt_quality_response(
        user_message="翻译质量不好，文风很直译，请帮我优化 Prompt。",
        state=state,
        cache_root=tmp_path,
    )

    assert response is not None
    reply, draft = response
    assert "创建一套新的" in reply
    assert draft is not None
    assert draft.kind == "create_prompt_preset"
    assert draft.payload["kind"] == "translation"
    assert "文风不自然" in str(draft.payload["system_prompt"])


def test_prompt_name_and_quality_issue_helpers() -> None:
    assert extract_prompt_create_name("新建翻译 Prompt，名字叫「文学翻译」") == "文学翻译"
    assert quality_issue_markers("翻译有源文残留、人名不一致、漏翻") == [
        "源文残留",
        "人名不一致",
        "漏翻",
    ]
