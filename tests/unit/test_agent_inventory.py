from __future__ import annotations

from pathlib import Path

from transoria.bridge.handlers.agent_inventory import (
    generate_prompt_id,
    inventory,
    looks_like_placeholder_value,
    prompt_body,
    prompt_store_for,
    settings_defaults,
)
from transoria.llm.config import ModelConfig, ProviderFormat
from transoria.model_profiles import ModelProfileStore
from transoria.prompts import PromptKind, PromptPreset, PromptPresetStore
from transoria.settings import SettingsStore


def test_inventory_excludes_placeholder_profiles_and_disabled_prompts(
    tmp_path: Path,
) -> None:
    profile_store = ModelProfileStore.from_cache_root(tmp_path)
    profile_store.create(
        ModelConfig(
            id="valid",
            display_name="Valid",
            provider_format=ProviderFormat.OPENAI,
            base_url="https://example.com/v1",
            model_id="valid-model",
            api_keys=("secret",),
        )
    )
    profile_store.create(
        ModelConfig(
            id="placeholder",
            display_name="DeepSeek Pro",
            provider_format=ProviderFormat.OPENAI,
            base_url="https://example.com/v1",
            model_id="新的值（例如",
            api_keys=("secret",),
        )
    )
    prompt_store_for(tmp_path, PromptKind.TRANSLATION).save(
        (
            PromptPreset(
                id="enabled",
                name="Enabled",
                kind=PromptKind.TRANSLATION,
                system_prompt="prompt",
                enabled=True,
            ),
            PromptPreset(
                id="disabled",
                name="Disabled",
                kind=PromptKind.TRANSLATION,
                system_prompt="prompt",
                enabled=False,
            ),
        )
    )

    result = inventory(profile_store, tmp_path)

    assert [profile["id"] for profile in result["profiles"]] == ["valid"]
    assert result["profiles"][0]["api_key_configured"] is True
    assert result["excluded_profile_count"] == 1
    translation_prompts = result["prompts"]["translation"]  # type: ignore[index]
    prompt_ids = {prompt["id"] for prompt in translation_prompts}  # type: ignore[union-attr]
    assert "enabled" in prompt_ids
    assert "disabled" not in prompt_ids


def test_prompt_body_includes_enabled_and_default_flags() -> None:
    preset = PromptPreset(
        id="preset",
        name="Preset",
        kind=PromptKind.GLOSSARY_REVIEW,
        system_prompt="body",
        description="desc",
        enabled=False,
    )

    assert prompt_body(preset) == {
        "id": "preset",
        "name": "Preset",
        "kind": "glossary_review",
        "description": "desc",
        "is_system": False,
        "system_prompt": "body",
        "enabled": False,
        "is_default": False,
    }


def test_generate_prompt_id_uses_kind_slug_and_avoids_existing_ids() -> None:
    existing = [
        PromptPreset(
            id="agent-translation-my-prompt-deadbe",
            name="Existing",
            kind=PromptKind.TRANSLATION,
            system_prompt="body",
        )
    ]

    generated = generate_prompt_id("My Prompt", PromptKind.TRANSLATION, existing)

    assert generated.startswith("agent-translation-my-prompt-")
    assert generated not in {preset.id for preset in existing}


def test_settings_defaults_exposes_agent_read_only_defaults(tmp_path: Path) -> None:
    defaults = settings_defaults(SettingsStore(tmp_path / "settings.json"))

    assert set(defaults) == {"translation", "glossary", "glossary_review"}
    assert "input_dir" in defaults["translation"]  # type: ignore[operator]
    assert "novel_background" in defaults["glossary"]  # type: ignore[operator]
    assert "output_filename" in defaults["glossary_review"]  # type: ignore[operator]


def test_placeholder_detection_matches_agent_safety_filter() -> None:
    assert looks_like_placeholder_value("新的值（例如 deepseek-v4-pro）")
    assert looks_like_placeholder_value("your-model-id")
    assert not looks_like_placeholder_value("deepseek-v4-flash")
