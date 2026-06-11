from __future__ import annotations

from pathlib import Path

from transoria.agent.schemas import AgentWorkspaceState
from transoria.bridge.handlers.agent_model_intents import (
    extract_model_copy_display_name,
    find_existing_upgrade_profile,
    infer_provider_model_id_from_copy_request,
    requested_existing_model_for_task,
    resolve_model_profile_from_copy_request,
)
from transoria.llm.config import ModelConfig, ProviderFormat
from transoria.model_profiles import ModelProfileStore


def _profile(
    profile_id: str,
    *,
    display_name: str,
    model_id: str,
    api_keys: tuple[str, ...] = ("sk-test",),
) -> ModelConfig:
    return ModelConfig(
        id=profile_id,
        display_name=display_name,
        provider_format=ProviderFormat.OPENAI,
        base_url="https://api.deepseek.com/v1",
        model_id=model_id,
        api_keys=api_keys,
    )


def _store(tmp_path: Path, *profiles: ModelConfig) -> ModelProfileStore:
    store = ModelProfileStore.from_cache_root(tmp_path)
    for profile in profiles:
        store.create(profile)
    return store


def test_resolve_model_profile_from_copy_request_matches_display_name(
    tmp_path: Path,
) -> None:
    source = _profile(
        "deepseek-f",
        display_name="DeepSeek-f",
        model_id="deepseek-v4-flash",
    )
    store = _store(tmp_path, source)

    resolved = resolve_model_profile_from_copy_request(
        "按照 DeepSeek-f 的配置复制一个新模型。",
        profile_store=store,
    )

    assert resolved == source


def test_extract_model_copy_display_name_reads_chinese_name() -> None:
    assert (
        extract_model_copy_display_name("然后将模型名称改为 DeepSeek 4 Pro。")
        == "DeepSeek 4 Pro"
    )


def test_infer_provider_model_id_replaces_flash_with_pro_when_requested() -> None:
    source = _profile(
        "deepseek-f",
        display_name="DeepSeek-f",
        model_id="deepseek-v4-flash",
    )

    inferred = infer_provider_model_id_from_copy_request(
        source,
        display_name="DeepSeek 4 Pro",
        text="不要用相同的模型 ID，复制配置后把模型名称改为 DeepSeek 4 Pro。",
    )

    assert inferred == "deepseek-v4-pro"


def test_requested_existing_model_for_task_resolves_task_model_reference(
    tmp_path: Path,
) -> None:
    profile = _profile(
        "deepseek-f",
        display_name="DeepSeek-f",
        model_id="deepseek-v4-flash",
    )
    store = _store(tmp_path, profile)

    resolved = requested_existing_model_for_task(
        "请用我已经配置好的 DeepSeek-f 模型进行术语提取。",
        profile_store=store,
    )

    assert resolved == profile


def test_find_existing_upgrade_profile_prefers_higher_quality_same_family(
    tmp_path: Path,
) -> None:
    flash = _profile(
        "deepseek-f",
        display_name="DeepSeek Flash",
        model_id="deepseek-v4-flash",
    )
    pro = _profile(
        "deepseek-pro",
        display_name="DeepSeek Pro",
        model_id="deepseek-v4-pro",
    )
    store = _store(tmp_path, flash, pro)
    state = AgentWorkspaceState(
        workflow_model_id=flash.id,
        stage_model_ids={
            "translation": flash.id,
            "term_extract": None,
            "term_review": None,
        },
    )

    resolved = find_existing_upgrade_profile(
        "帮我把当前 DeepSeek 模型换成更强一点的模型。",
        state=state,
        profile_store=store,
    )

    assert resolved == pro


def test_find_existing_upgrade_profile_returns_none_on_tie(tmp_path: Path) -> None:
    flash = _profile(
        "deepseek-f",
        display_name="DeepSeek Flash",
        model_id="deepseek-v4-flash",
    )
    pro_a = _profile(
        "deepseek-pro-a",
        display_name="DeepSeek Pro A",
        model_id="deepseek-v4-pro",
    )
    pro_b = _profile(
        "deepseek-pro-b",
        display_name="DeepSeek Pro B",
        model_id="deepseek-v4-pro",
    )
    store = _store(tmp_path, flash, pro_a, pro_b)
    state = AgentWorkspaceState(
        workflow_model_id=flash.id,
        stage_model_ids={
            "translation": flash.id,
            "term_extract": None,
            "term_review": None,
        },
    )

    resolved = find_existing_upgrade_profile(
        "帮我把 DeepSeek 模型升级成更高质量的模型。",
        state=state,
        profile_store=store,
    )

    assert resolved is None
