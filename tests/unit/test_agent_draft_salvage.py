from __future__ import annotations

from pathlib import Path

from transoria.agent.schemas import AgentActionDraft
from transoria.bridge.handlers.agent_draft_salvage import (
    resolve_prompt_by_name,
    salvage_compound_draft,
    salvage_create_prompt_draft,
    salvage_create_recipe_draft,
)
from transoria.bridge.handlers.agent_inventory import prompt_store_for
from transoria.llm.config import ModelConfig, ProviderFormat
from transoria.model_profiles import ModelProfileStore
from transoria.prompts import PromptKind, PromptPreset, default_preset


def _seed_profile(cache_root: Path) -> ModelProfileStore:
    store = ModelProfileStore.from_cache_root(cache_root)
    store.create(
        ModelConfig(
            id="deepseek-f",
            display_name="DeepSeek-f",
            provider_format=ProviderFormat.OPENAI,
            base_url="https://api.deepseek.com/v1",
            model_id="deepseek-v4-flash",
            concurrency_limit=2,
        )
    )
    return store


def test_salvage_create_prompt_draft_fills_kind_name_and_body() -> None:
    draft = AgentActionDraft.create(
        kind="create_prompt_preset",
        title="创建 Prompt",
        summary="创建术语审查提示词。",
        payload={},
    )

    salvaged = salvage_create_prompt_draft(
        draft,
        text="请新建一套术语审查 Prompt「小说术语审查预设」，重点检查人名一致。",
    )

    assert salvaged is not None
    assert salvaged.payload["kind"] == "glossary_review"
    assert salvaged.payload["name"] == "小说术语审查预设"
    assert "人名一致" in str(salvaged.payload["system_prompt"])


def test_salvage_create_recipe_draft_uses_current_stage_selection() -> None:
    draft = AgentActionDraft.create(
        kind="create_recipe",
        title="保存预设",
        summary="保存当前配置。",
        payload={},
    )

    salvaged = salvage_create_recipe_draft(
        draft,
        text="把当前阶段配置保存成「测试复合配置」的预设。",
        current_state={
            "stage_model_ids": {"translation": "deepseek-f"},
            "stage_prompt_ids": {"translation": "translation-prompt"},
        },
    )

    assert salvaged is not None
    assert salvaged.payload["name"] == "测试复合配置"
    assert salvaged.payload["stage_model_ids"] == {"translation": "deepseek-f"}
    assert salvaged.payload["stage_prompt_ids"] == {
        "translation": "translation-prompt"
    }


def test_salvage_compound_draft_builds_multiple_actions(tmp_path: Path) -> None:
    profile_store = _seed_profile(tmp_path)
    prompt_store_for(tmp_path, PromptKind.TRANSLATION).save(
        [default_preset(PromptKind.TRANSLATION)]
    )
    draft = AgentActionDraft.create(
        kind="compound_config_update",
        title="复合配置修改",
        summary="修改模型、Prompt 和预设。",
        payload={},
    )

    salvaged = salvage_compound_draft(
        draft,
        text=(
            "请帮我准备一个配置修改草案：把模型 DeepSeek-f 的并发数改成 4，"
            "把翻译 Prompt「默认」重命名为「标准中文翻译预设」，"
            "并把当前阶段配置保存成「测试复合配置」的预设。"
        ),
        current_state={
            "stage_model_ids": {"translation": "deepseek-f"},
            "stage_prompt_ids": {"translation": "default-translation-zh"},
        },
        profile_store=profile_store,
        cache_root=tmp_path,
    )

    assert salvaged is not None
    actions = salvaged.payload["actions"]
    assert isinstance(actions, list)
    assert [action["kind"] for action in actions] == [
        "update_model_profile",
        "create_prompt_preset",
        "create_recipe",
    ]
    assert actions[0]["payload"] == {
        "profile_id": "deepseek-f",
        "patch": {"concurrency_limit": 4},
    }
    assert actions[1]["payload"]["name"] == "标准中文翻译预设"
    assert actions[2]["payload"]["name"] == "测试复合配置"


def test_resolve_prompt_by_name_prefers_custom_duplicate(tmp_path: Path) -> None:
    custom = PromptPreset(
        id="custom-default",
        name="默认",
        kind=PromptKind.TRANSLATION,
        system_prompt="custom body",
        is_system=False,
    )
    prompt_store_for(tmp_path, PromptKind.TRANSLATION).save(
        [default_preset(PromptKind.TRANSLATION), custom]
    )

    resolved = resolve_prompt_by_name(
        "默认",
        kinds=(PromptKind.TRANSLATION,),
        cache_root=tmp_path,
    )

    assert resolved == custom
