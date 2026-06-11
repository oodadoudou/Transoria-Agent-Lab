from __future__ import annotations

from pathlib import Path

import pytest

from transoria.bridge.errors import BridgeError
from transoria.bridge.handlers.agent_model_profile_actions import (
    coerce_model_profile_patch,
    coerce_model_profile_update_payload,
    model_profile_body,
    model_profile_from_draft_payload,
)
from transoria.llm.config import ModelConfig, ProviderFormat, ThinkingLevel
from transoria.model_profiles import ModelProfileStore


def _profile(profile_id: str = "deepseek-f") -> ModelConfig:
    return ModelConfig(
        id=profile_id,
        display_name="DeepSeek-f",
        provider_format=ProviderFormat.OPENAI,
        base_url="https://api.deepseek.com/v1",
        model_id="deepseek-v4-flash",
        thinking_level=ThinkingLevel.OFF,
        concurrency_limit=2,
    )


def test_model_profile_from_payload_creates_profile_with_trimmed_keys() -> None:
    profile = model_profile_from_draft_payload(
        {
            "display_name": "DeepSeek Pro",
            "provider_format": "openai",
            "base_url": "https://api.deepseek.com/v1",
            "model_id": "deepseek-v4-pro",
            "api_keys": ["  sk-one  ", ""],
        }
    )

    assert profile.display_name == "DeepSeek Pro"
    assert profile.provider_format is ProviderFormat.OPENAI
    assert profile.api_keys == ("sk-one",)


def test_model_profile_from_payload_copies_source_profile(tmp_path: Path) -> None:
    store = ModelProfileStore.from_cache_root(tmp_path)
    source = store.create(_profile().with_api_keys(("sk-source",)))

    copied = model_profile_from_draft_payload(
        {
            "copy_from_profile_id": source.id,
            "display_name": "DeepSeek 4 Pro",
            "model_id": "deepseek-v4-pro",
        },
        profile_store=store,
    )

    assert copied.id != source.id
    assert copied.display_name == "DeepSeek 4 Pro"
    assert copied.provider_format is source.provider_format
    assert copied.base_url == source.base_url
    assert copied.api_keys == ("sk-source",)


def test_model_profile_from_payload_rejects_placeholder_model_id() -> None:
    with pytest.raises(BridgeError):
        model_profile_from_draft_payload(
            {
                "display_name": "DeepSeek 4 Pro",
                "provider_format": "openai",
                "base_url": "https://api.deepseek.com/v1",
                "model_id": "新的值（例如 deepseek-v4-pro）",
            }
        )


def test_coerce_model_profile_patch_converts_enums_and_headers() -> None:
    patch = coerce_model_profile_patch(
        {
            "provider_format": "openai",
            "thinking_level": "high",
            "custom_headers": [["X-Test", "yes"], ["ignored"]],
        }
    )

    assert patch["provider_format"] is ProviderFormat.OPENAI
    assert patch["thinking_level"] is ThinkingLevel.HIGH
    assert patch["custom_headers"] == (("X-Test", "yes"),)


def test_coerce_model_profile_update_payload_extracts_api_keys() -> None:
    profile_id, patch, api_keys = coerce_model_profile_update_payload(
        {
            "profile_id": "deepseek-f",
            "patch": {"display_name": "DeepSeek", "api_keys": ["sk-one"]},
        }
    )

    assert profile_id == "deepseek-f"
    assert patch == {"display_name": "DeepSeek"}
    assert api_keys == ["sk-one"]


def test_model_profile_body_masks_api_keys(tmp_path: Path) -> None:
    store = ModelProfileStore.from_cache_root(tmp_path)
    stored = store.create(_profile().with_api_keys(("sk-source",)))

    body = model_profile_body(stored, profile_store=store)

    assert "api_keys" not in body
    assert body["api_key_configured"] is True
    assert body["api_key_status"] == "present"
