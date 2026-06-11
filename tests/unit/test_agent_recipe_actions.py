from __future__ import annotations

from pathlib import Path

import pytest

from transoria.agent.schemas import AgentRecipe, AgentWorkspaceState
from transoria.bridge.errors import BridgeError
from transoria.bridge.handlers.agent_inventory import prompt_store_for
from transoria.bridge.handlers.agent_recipe_actions import (
    coerce_model_slots,
    coerce_prompt_slots,
    coerce_recipe_payload,
    coerce_recipe_update_payload,
    recipe_body_from_payload,
    require_recipe_from_payload,
)
from transoria.llm.config import ModelConfig, ProviderFormat
from transoria.model_profiles import ModelProfileStore
from transoria.prompts import PromptKind, PromptPreset


def _seed_profile(cache_root: Path, profile_id: str = "profile-one") -> None:
    ModelProfileStore.from_cache_root(cache_root).create(
        ModelConfig(
            id=profile_id,
            display_name="Profile One",
            provider_format=ProviderFormat.OPENAI,
            base_url="https://example.com/v1",
            model_id="model-one",
        )
    )


def _seed_prompt(
    cache_root: Path,
    kind: PromptKind,
    preset_id: str,
) -> None:
    prompt_store_for(cache_root, kind).save(
        [
            PromptPreset(
                id=preset_id,
                name="Prompt",
                kind=kind,
                system_prompt="body",
            )
        ]
    )


def test_recipe_body_accepts_nested_recipe_and_chinese_name_alias() -> None:
    body = recipe_body_from_payload({"recipe": {"预设名称": "标准流程"}})

    assert body["name"] == "标准流程"


def test_coerce_recipe_payload_validates_stage_slots(tmp_path: Path) -> None:
    _seed_profile(tmp_path)
    _seed_prompt(tmp_path, PromptKind.GLOSSARY, "glossary-prompt")
    profile_store = ModelProfileStore.from_cache_root(tmp_path)

    name, description, stage_models, stage_prompts = coerce_recipe_payload(
        {
            "name": "测试流程",
            "description": "desc",
            "stage_model_ids": {"translation": "profile-one"},
            "stage_prompt_ids": {"term_extract": "glossary-prompt"},
        },
        profile_store=profile_store,
        cache_root=tmp_path,
    )

    assert name == "测试流程"
    assert description == "desc"
    assert stage_models == {"translation": "profile-one"}
    assert stage_prompts == {"term_extract": "glossary-prompt"}


def test_coerce_recipe_update_rejects_empty_update(tmp_path: Path) -> None:
    with pytest.raises(BridgeError):
        coerce_recipe_update_payload(
            {"id": "recipe-one"},
            profile_store=ModelProfileStore.from_cache_root(tmp_path),
            cache_root=tmp_path,
        )


def test_require_recipe_from_payload_resolves_existing_recipe() -> None:
    recipe = AgentRecipe.create(name="流程")
    state = AgentWorkspaceState.empty().add_recipe(recipe)

    assert require_recipe_from_payload({"recipe_id": recipe.id}, state=state) == recipe


def test_coerce_model_slots_rejects_unknown_slot_and_profile(tmp_path: Path) -> None:
    profile_store = ModelProfileStore.from_cache_root(tmp_path)

    with pytest.raises(BridgeError):
        coerce_model_slots({"bad_slot": "missing"}, profile_store=profile_store)

    with pytest.raises(BridgeError):
        coerce_model_slots({"translation": "missing"}, profile_store=profile_store)


def test_coerce_prompt_slots_is_bound_to_slot_kind(tmp_path: Path) -> None:
    _seed_prompt(tmp_path, PromptKind.GLOSSARY, "glossary-prompt")

    assert coerce_prompt_slots({"term_extract": "glossary-prompt"}, cache_root=tmp_path)

    with pytest.raises(BridgeError):
        coerce_prompt_slots({"term_review": "glossary-prompt"}, cache_root=tmp_path)
