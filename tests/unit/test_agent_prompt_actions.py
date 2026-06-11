from __future__ import annotations

from pathlib import Path

import pytest

from transoria.bridge.errors import BridgeError
from transoria.bridge.handlers.agent_inventory import prompt_store_for
from transoria.bridge.handlers.agent_prompt_actions import (
    coerce_prompt_kind,
    coerce_prompt_patch,
    create_prompt_preset,
    prompt_body_from_payload,
    update_prompt_preset,
)
from transoria.prompts import PromptKind, PromptPreset, default_preset


def test_create_prompt_preset_infers_glossary_review_kind_and_writes_store(
    tmp_path: Path,
) -> None:
    preset = create_prompt_preset(
        {
            "name": "小说术语审查预设",
            "description": "检查人名和组织名一致性。",
            "prompt": "重点检查 ABO 设定继承。",
        },
        cache_root=tmp_path,
    )

    assert preset.kind is PromptKind.GLOSSARY_REVIEW
    assert preset.name == "小说术语审查预设"
    assert preset.system_prompt == "重点检查 ABO 设定继承。"
    stored = prompt_store_for(tmp_path, PromptKind.GLOSSARY_REVIEW).load()
    assert [item.id for item in stored if item.id == preset.id] == [preset.id]


def test_update_prompt_preset_edits_custom_preset(tmp_path: Path) -> None:
    store = prompt_store_for(tmp_path, PromptKind.TRANSLATION)
    store.save(
        [
            default_preset(PromptKind.TRANSLATION),
            PromptPreset(
                id="custom-translation",
                name="Old",
                kind=PromptKind.TRANSLATION,
                system_prompt="old body",
                description="old desc",
                enabled=True,
                is_system=False,
            ),
        ]
    )

    updated = update_prompt_preset(
        {
            "id": "custom-translation",
            "patch": {
                "name": "New",
                "system_prompt": " new body ",
                "description": "new desc",
                "enabled": False,
            },
        },
        cache_root=tmp_path,
    )

    assert updated.name == "New"
    assert updated.system_prompt == "new body"
    assert updated.description == "new desc"
    assert updated.enabled is False
    stored = {preset.id: preset for preset in store.load()}
    assert stored["custom-translation"].name == "New"


def test_update_prompt_preset_rejects_system_preset(tmp_path: Path) -> None:
    system_preset = default_preset(PromptKind.GLOSSARY)
    prompt_store_for(tmp_path, PromptKind.GLOSSARY).save([system_preset])

    with pytest.raises(BridgeError) as exc_info:
        update_prompt_preset(
            {"id": system_preset.id, "patch": {"name": "不能改"}},
            cache_root=tmp_path,
        )

    assert "read-only" in str(exc_info.value)


def test_coerce_prompt_kind_supports_aliases_and_fallback_text() -> None:
    assert coerce_prompt_kind("术语提取 Prompt") is PromptKind.GLOSSARY
    assert coerce_prompt_kind("term_review") is PromptKind.GLOSSARY_REVIEW
    assert (
        coerce_prompt_kind(None, fallback_text="请创建一个文学小说翻译提示词")
        is PromptKind.TRANSLATION
    )


def test_prompt_patch_rejects_unknown_and_invalid_fields() -> None:
    with pytest.raises(BridgeError):
        coerce_prompt_patch({"suffix_prompt": "not editable here"})

    with pytest.raises(BridgeError):
        coerce_prompt_patch({"enabled": "yes"})


def test_prompt_body_from_payload_uses_first_non_empty_body() -> None:
    assert (
        prompt_body_from_payload(
            {"system_prompt": " ", "content": "正文", "body": "fallback"}
        )
        == "正文"
    )
