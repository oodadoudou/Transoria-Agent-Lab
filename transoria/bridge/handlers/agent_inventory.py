"""Read-only inventory helpers for Agent Lab bridge handlers."""

from __future__ import annotations

from pathlib import Path
from secrets import token_hex

from transoria.llm.config import ModelConfig, ThinkingLevel
from transoria.model_profiles import ModelProfileStore
from transoria.prompts import PromptKind, PromptPreset, PromptPresetStore
from transoria.settings import SettingsStore


def settings_defaults(settings_store: SettingsStore) -> dict[str, object]:
    settings = settings_store.load_all()
    return {
        "translation": {
            "input_dir": settings.translation.input_folder,
            "output_dir": settings.translation.output_folder,
            "source_language": settings.translation.source_language,
            "target_language": settings.translation.target_language,
        },
        "glossary": {
            "input_dir": settings.glossary.input_folder,
            "output_dir": settings.glossary.output_folder,
            "source_language": settings.glossary.source_language,
            "target_language": settings.glossary.target_language,
            "novel_background": settings.glossary.novel_background,
        },
        "glossary_review": {
            "input_dir": settings.glossary_review.input_folder,
            "novel_background": settings.glossary_review.novel_background,
            "output_filename": settings.glossary_review.output_filename,
        },
    }


def inventory(profile_store: ModelProfileStore, cache_root: Path) -> dict[str, object]:
    profiles = profile_store.load()
    prompt_groups: dict[str, object] = {}
    for kind in PromptKind:
        prompt_groups[kind.value] = [
            prompt_summary(preset)
            for preset in prompt_store_for(cache_root, kind).load()
            if preset.enabled
        ]
    return {
        "profiles": [
            profile_inventory_summary(profile)
            for profile in profiles
            if not model_profile_has_placeholder_fields(profile)
        ],
        "excluded_profile_count": sum(
            1 for profile in profiles if model_profile_has_placeholder_fields(profile)
        ),
        "prompts": prompt_groups,
    }


def profile_inventory_summary(profile: ModelConfig) -> dict[str, object]:
    return {
        "id": profile.id,
        "display_name": profile.display_name,
        "provider_format": profile.provider_format.value,
        "base_url": profile.base_url,
        "model_id": profile.model_id,
        "api_key_configured": bool(profile.api_keys),
        "thinking_level": profile.thinking_level.value,
        "supports_thinking": profile.thinking_level is not ThinkingLevel.OFF,
        "max_output_tokens": profile.max_output_tokens,
        "input_token_limit": profile.input_token_limit,
        "concurrency_limit": profile.concurrency_limit,
        "rpm_limit": profile.rpm_limit,
        "tpm_limit": profile.tpm_limit,
        "retry_attempts": profile.retry_attempts,
    }


def model_profile_has_placeholder_fields(profile: ModelConfig) -> bool:
    return looks_like_placeholder_value(
        profile.display_name
    ) or looks_like_placeholder_value(profile.model_id)


def looks_like_placeholder_value(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    if not normalized:
        return False
    return any(
        marker in normalized
        for marker in (
            "新的值",
            "例如",
            "比如",
            "待填",
            "占位",
            "placeholder",
            "example",
            "your-model",
            "model-id",
            "<model",
        )
    )


def prompt_summary(preset: PromptPreset) -> dict[str, object]:
    return {
        "id": preset.id,
        "name": preset.name,
        "kind": preset.kind.value,
        "description": preset.description,
        "is_system": preset.is_system,
    }


def prompt_body(preset: PromptPreset) -> dict[str, object]:
    return {
        **prompt_summary(preset),
        "system_prompt": preset.system_prompt,
        "enabled": preset.enabled,
        "is_default": False,
    }


def prompt_store_for(cache_root: Path, kind: PromptKind) -> PromptPresetStore:
    return PromptPresetStore(
        path=cache_root / f"prompts.{kind.value}.json",
        kind=kind,
    )


def generate_prompt_id(
    name: str,
    kind: PromptKind,
    existing: list[PromptPreset],
) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")
    if not slug:
        slug = kind.value
    existing_ids = {preset.id for preset in existing}
    while True:
        candidate = f"agent-{kind.value}-{slug}-{token_hex(3)}"
        if candidate not in existing_ids:
            return candidate
