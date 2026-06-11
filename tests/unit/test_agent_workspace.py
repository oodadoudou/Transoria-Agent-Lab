from __future__ import annotations

from pathlib import Path

import pytest

from transoria.agent.schemas import AgentRecipe, AgentWorkspaceState
from transoria.bridge.errors import BridgeError
from transoria.bridge.handlers.agent_workspace import (
    apply_workspace_patch,
    coerce_active_recipe_id,
    coerce_workflow_thinking_level,
    optional_agent_limit,
    optional_agent_string,
    profile_for_workflow_chat,
)
from transoria.llm.config import ModelConfig, ProviderFormat, ThinkingLevel
from transoria.model_profiles import ModelProfileStore


def _profile(
    profile_id: str,
    *,
    thinking_level: ThinkingLevel = ThinkingLevel.OFF,
) -> ModelConfig:
    return ModelConfig(
        id=profile_id,
        display_name=profile_id,
        provider_format=ProviderFormat.OPENAI,
        base_url="https://example.com/v1",
        model_id=profile_id,
        thinking_level=thinking_level,
    )


def test_apply_workspace_patch_resets_thinking_to_selected_profile_default(
    tmp_path: Path,
) -> None:
    store = ModelProfileStore.from_cache_root(tmp_path)
    store.create(_profile("flash"))
    store.create(_profile("pro", thinking_level=ThinkingLevel.HIGH))
    state = AgentWorkspaceState.empty().with_config(
        workflow_model_id="flash",
        workflow_thinking_level="off",
    )

    updated = apply_workspace_patch(
        state,
        {"workflow_model_id": "pro"},
        profile_store=store,
        cache_root=tmp_path,
    )

    assert updated.workflow_model_id == "pro"
    assert updated.workflow_thinking_level == "high"


def test_apply_workspace_patch_rejects_thinking_for_non_thinking_model(
    tmp_path: Path,
) -> None:
    store = ModelProfileStore.from_cache_root(tmp_path)
    store.create(_profile("flash"))
    state = AgentWorkspaceState.empty().with_config(workflow_model_id="flash")

    with pytest.raises(BridgeError):
        apply_workspace_patch(
            state,
            {"workflow_thinking_level": "high"},
            profile_store=store,
            cache_root=tmp_path,
        )


def test_apply_workspace_patch_clears_active_recipe_when_stage_selection_changes(
    tmp_path: Path,
) -> None:
    store = ModelProfileStore.from_cache_root(tmp_path)
    store.create(_profile("model-a"))
    store.create(_profile("model-b"))
    recipe = AgentRecipe.create(
        name="Flow",
        stage_model_ids={"translation": "model-a"},
    )
    state = (
        AgentWorkspaceState.empty()
        .add_recipe(recipe)
        .with_config(
            active_recipe_id=recipe.id,
            stage_model_ids=recipe.stage_model_ids,
            stage_prompt_ids=recipe.stage_prompt_ids,
        )
    )

    updated = apply_workspace_patch(
        state,
        {"stage_model_ids": {"translation": "model-b"}},
        profile_store=store,
        cache_root=tmp_path,
    )

    assert updated.stage_model_ids["translation"] == "model-b"
    assert updated.active_recipe_id is None


def test_active_recipe_id_validation_accepts_existing_and_rejects_missing() -> None:
    recipe = AgentRecipe.create(name="Flow")
    state = AgentWorkspaceState.empty().add_recipe(recipe)

    assert coerce_active_recipe_id(recipe.id, state=state) == recipe.id
    assert coerce_active_recipe_id("", state=state) is None
    with pytest.raises(BridgeError):
        coerce_active_recipe_id("missing", state=state)


def test_optional_agent_payload_helpers_validate_types() -> None:
    assert optional_agent_string({"kind": " translation "}, "kind") == "translation"
    assert optional_agent_string({}, "kind") is None
    assert optional_agent_limit({}, default=5) == 5
    assert optional_agent_limit({"limit": "2"}, default=5) == 2

    with pytest.raises(BridgeError):
        optional_agent_string({"kind": 1}, "kind")
    with pytest.raises(BridgeError):
        optional_agent_limit({"limit": -1}, default=5)


def test_workflow_thinking_override_does_not_mutate_profile(tmp_path: Path) -> None:
    store = ModelProfileStore.from_cache_root(tmp_path)
    profile = store.create(_profile("pro", thinking_level=ThinkingLevel.HIGH))

    assert (
        coerce_workflow_thinking_level(
            "low",
            workflow_model_id=profile.id,
            profile_store=store,
        )
        == "low"
    )
    overridden = profile_for_workflow_chat(profile, "low")

    assert profile.thinking_level is ThinkingLevel.HIGH
    assert overridden.thinking_level is ThinkingLevel.LOW
