"""Preview helpers for compound Agent Lab action validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from transoria.agent.schemas import AgentActionDraft, AgentRecipe, AgentWorkspaceState
from transoria.bridge.handlers.agent_memory_actions import (
    coerce_memories_payload,
    coerce_memory_items_payload,
)
from transoria.bridge.handlers.agent_model_profile_actions import (
    coerce_api_keys,
    coerce_model_profile_patch,
    coerce_model_profile_update_payload,
    model_profile_from_draft_payload,
)
from transoria.bridge.handlers.agent_recipe_actions import (
    coerce_recipe_payload,
    coerce_recipe_update_payload,
    require_recipe_from_payload,
)
from transoria.bridge.handlers.agent_workspace import apply_workspace_patch
from transoria.llm.config import ModelConfig
from transoria.model_profiles import ModelProfileStore


def preview_compound_state(
    state: AgentWorkspaceState,
    draft: AgentActionDraft,
    *,
    profile_store: ModelProfileStore | "PreviewModelProfileStore",
    cache_root: Path,
) -> AgentWorkspaceState:
    if draft.kind == "update_workspace":
        return apply_workspace_patch(
            state,
            draft.payload,
            profile_store=profile_store,
            cache_root=cache_root,
        )
    if draft.kind == "update_memory":
        return state.with_memories(coerce_memories_payload(draft.payload))
    if draft.kind == "add_memory":
        memories = list(state.memories)
        for memory in coerce_memory_items_payload(draft.payload):
            if memory not in memories:
                memories.append(memory)
        return state.with_memories(tuple(memories))
    if draft.kind == "delete_memory":
        removals = set(coerce_memory_items_payload(draft.payload))
        return state.with_memories(
            tuple(item for item in state.memories if item not in removals)
        )
    if draft.kind == "create_recipe":
        name, description, stage_models, stage_prompts = coerce_recipe_payload(
            draft.payload,
            profile_store=profile_store,
            cache_root=cache_root,
        )
        return state.add_recipe(
            AgentRecipe.create(
                name=name,
                description=description,
                stage_model_ids=stage_models,
                stage_prompt_ids=stage_prompts,
            )
        )
    if draft.kind == "update_recipe":
        recipe_id, fields, stage_models, stage_prompts = coerce_recipe_update_payload(
            draft.payload,
            profile_store=profile_store,
            cache_root=cache_root,
        )
        recipe = state.get_recipe(recipe_id)
        if recipe is None:
            return state
        return state.replace_recipe(
            recipe.with_updates(
                name=fields.get("name"),
                description=fields.get("description"),
                stage_model_ids=stage_models,
                stage_prompt_ids=stage_prompts,
            )
        )
    if draft.kind == "apply_recipe":
        recipe = require_recipe_from_payload(draft.payload, state=state)
        return apply_workspace_patch(
            state,
            {
                "active_recipe_id": recipe.id,
                "stage_model_ids": dict(recipe.stage_model_ids),
                "stage_prompt_ids": dict(recipe.stage_prompt_ids),
            },
            profile_store=profile_store,
            cache_root=cache_root,
        )
    if draft.kind == "delete_recipe":
        recipe = require_recipe_from_payload(draft.payload, state=state)
        return state.remove_recipe(recipe.id)
    return state


class PreviewModelProfileStore:
    """Validation-only overlay for compound drafts."""

    def __init__(
        self,
        base: ModelProfileStore | "PreviewModelProfileStore",
        overlay: Mapping[str, ModelConfig],
    ) -> None:
        self._base = base
        self._overlay = dict(overlay)

    def get(self, profile_id: str) -> ModelConfig | None:
        if profile_id in self._overlay:
            return self._overlay[profile_id]
        return self._base.get(profile_id)

    def load(self) -> tuple[ModelConfig, ...]:
        base_profiles = {
            profile.id: profile
            for profile in self._base.load()
            if profile.id not in self._overlay
        }
        return (*base_profiles.values(), *self._overlay.values())


def preview_compound_profile_store(
    profile_store: ModelProfileStore | PreviewModelProfileStore,
    draft: AgentActionDraft,
) -> ModelProfileStore | PreviewModelProfileStore:
    if draft.kind == "create_model_profile":
        profile = model_profile_from_draft_payload(
            draft.payload,
            profile_store=profile_store,  # type: ignore[arg-type]
        )
        return PreviewModelProfileStore(profile_store, {profile.id: profile})
    if draft.kind != "update_model_profile":
        return profile_store

    profile_id, patch, api_keys = coerce_model_profile_update_payload(draft.payload)
    current = profile_store.get(profile_id)
    if current is None:
        return profile_store
    updated = current
    if patch:
        updated = replace(
            updated,
            **coerce_model_profile_patch(patch),  # type: ignore[arg-type]
        )
    if api_keys is not None:
        updated = updated.with_api_keys(coerce_api_keys(api_keys))
    return PreviewModelProfileStore(profile_store, {profile_id: updated})
