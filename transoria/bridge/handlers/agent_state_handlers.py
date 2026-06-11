"""Workspace, conversation, memory, and recipe bridge handlers for Agent Lab."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from transoria.agent.project_store import AgentProjectStore
from transoria.agent.schemas import (
    AgentConversation,
    AgentRecipe,
    AgentWorkspaceState,
)
from transoria.bridge.errors import BridgeError
from transoria.bridge.handlers._utils import expect_string
from transoria.bridge.handlers.agent_action_registry import (
    MAX_RECIPES,
    coerce_compound_action_drafts,
)
from transoria.bridge.handlers.agent_memory_actions import coerce_memories_payload
from transoria.bridge.handlers.agent_recipe_actions import coerce_recipe_payload
from transoria.bridge.handlers.agent_wire import resolved_active_recipe_id
from transoria.bridge.handlers.agent_workspace import (
    apply_workspace_patch,
    workspace_response,
)
from transoria.model_profiles import ModelProfileStore

_MAX_TITLE_LENGTH = 120


def build_state_handlers(
    *,
    cache_root: Path,
    project_store: AgentProjectStore,
    profile_store: ModelProfileStore,
) -> dict[str, object]:
    def respond(state: AgentWorkspaceState) -> dict[str, object]:
        return workspace_response(
            state,
            profile_store,
            cache_root,
            normalize_compound_action_drafts=coerce_compound_action_drafts,
        )

    def update_workspace(payload: Mapping[str, object]) -> dict[str, object]:
        patch = payload.get("patch")
        if not isinstance(patch, Mapping):
            raise BridgeError.invalid_argument(
                "patch object is required.",
                field="patch",
            )
        state = project_store.update(
            lambda current: apply_workspace_patch(
                current,
                patch,
                profile_store=profile_store,
                cache_root=cache_root,
            )
        )
        return respond(state)

    def create_conversation(payload: Mapping[str, object]) -> dict[str, object]:
        raw_title = payload.get("title")
        title = raw_title.strip() if isinstance(raw_title, str) else ""
        if len(title) > _MAX_TITLE_LENGTH:
            raise BridgeError.invalid_argument(
                "title is too long.",
                field="title",
                details={"max_length": _MAX_TITLE_LENGTH},
            )
        conversation = AgentConversation.seeded()
        if title:
            conversation = conversation.with_title(title)
        state = project_store.update(
            lambda current: current.add_conversation(conversation)
        )
        return respond(state)

    def switch_conversation(payload: Mapping[str, object]) -> dict[str, object]:
        conversation_id = expect_string(payload, "conversation_id")
        current = project_store.load()
        if all(conv.id != conversation_id for conv in current.conversations):
            raise BridgeError.not_found(
                f"conversation {conversation_id!r} does not exist.",
                details={"conversation_id": conversation_id},
            )
        state = project_store.save(current.set_active(conversation_id))
        return respond(state)

    def rename_conversation(payload: Mapping[str, object]) -> dict[str, object]:
        conversation_id = expect_string(payload, "conversation_id")
        title = expect_string(payload, "title").strip()
        if not title:
            raise BridgeError.invalid_argument("title must not be empty.", field="title")
        if len(title) > _MAX_TITLE_LENGTH:
            raise BridgeError.invalid_argument(
                "title is too long.",
                field="title",
                details={"max_length": _MAX_TITLE_LENGTH},
            )
        current = project_store.load()
        target = next(
            (conv for conv in current.conversations if conv.id == conversation_id),
            None,
        )
        if target is None:
            raise BridgeError.not_found(
                f"conversation {conversation_id!r} does not exist.",
                details={"conversation_id": conversation_id},
            )
        state = project_store.save(
            current.replace_conversation(target.with_title(title))
        )
        return respond(state)

    def delete_conversation(payload: Mapping[str, object]) -> dict[str, object]:
        conversation_id = expect_string(payload, "conversation_id")
        current = project_store.load()
        if all(conv.id != conversation_id for conv in current.conversations):
            raise BridgeError.not_found(
                f"conversation {conversation_id!r} does not exist.",
                details={"conversation_id": conversation_id},
            )
        next_state = current.remove_conversation(conversation_id)
        if not next_state.conversations:
            next_state = next_state.add_conversation(AgentConversation.seeded())
        state = project_store.save(next_state)
        return respond(state)

    def update_memory(payload: Mapping[str, object]) -> dict[str, object]:
        memories = coerce_memories_payload(payload)
        state = project_store.update(lambda current: current.with_memories(memories))
        return respond(state)

    def delete_memory(payload: Mapping[str, object]) -> dict[str, object]:
        memory = expect_string(payload, "memory").strip()
        state = project_store.update(
            lambda current: current.with_memories(
                tuple(item for item in current.memories if item != memory)
            )
        )
        return respond(state)

    def create_recipe(payload: Mapping[str, object]) -> dict[str, object]:
        name, description, stage_models, stage_prompts = coerce_recipe_payload(
            payload,
            profile_store=profile_store,
            cache_root=cache_root,
        )

        def updater(current: AgentWorkspaceState) -> AgentWorkspaceState:
            if len(current.recipes) >= MAX_RECIPES:
                raise BridgeError.invalid_argument(
                    "too many recipes.",
                    field="recipes",
                    details={"max_count": MAX_RECIPES},
                )
            recipe = AgentRecipe.create(
                name=name,
                description=description,
                stage_model_ids=stage_models,
                stage_prompt_ids=stage_prompts,
            )
            return current.add_recipe(recipe)

        return respond(project_store.update(updater))

    def update_recipe(payload: Mapping[str, object]) -> dict[str, object]:
        recipe_id = expect_string(payload, "recipe_id")
        name, description, stage_models, stage_prompts = coerce_recipe_payload(
            payload,
            profile_store=profile_store,
            cache_root=cache_root,
        )

        def updater(current: AgentWorkspaceState) -> AgentWorkspaceState:
            recipe = current.get_recipe(recipe_id)
            if recipe is None:
                raise BridgeError.not_found(
                    f"recipe {recipe_id!r} does not exist.",
                    details={"recipe_id": recipe_id},
                )
            updated = recipe.with_updates(
                name=name,
                description=description,
                stage_model_ids=stage_models,
                stage_prompt_ids=stage_prompts,
            )
            next_state = current.replace_recipe(updated)
            if resolved_active_recipe_id(current) != recipe_id:
                return next_state
            return next_state.with_config(
                stage_model_ids=dict(updated.stage_model_ids),
                stage_prompt_ids=dict(updated.stage_prompt_ids),
                active_recipe_id=updated.id,
            )

        return respond(project_store.update(updater))

    def delete_recipe(payload: Mapping[str, object]) -> dict[str, object]:
        recipe_id = expect_string(payload, "recipe_id")

        def updater(current: AgentWorkspaceState) -> AgentWorkspaceState:
            if current.get_recipe(recipe_id) is None:
                raise BridgeError.not_found(
                    f"recipe {recipe_id!r} does not exist.",
                    details={"recipe_id": recipe_id},
                )
            return current.remove_recipe(recipe_id)

        return respond(project_store.update(updater))

    def apply_recipe(payload: Mapping[str, object]) -> dict[str, object]:
        recipe_id = expect_string(payload, "recipe_id")

        def updater(current: AgentWorkspaceState) -> AgentWorkspaceState:
            recipe = current.get_recipe(recipe_id)
            if recipe is None:
                raise BridgeError.not_found(
                    f"recipe {recipe_id!r} does not exist.",
                    details={"recipe_id": recipe_id},
                )
            return apply_workspace_patch(
                current,
                {
                    "active_recipe_id": recipe.id,
                    "stage_model_ids": dict(recipe.stage_model_ids),
                    "stage_prompt_ids": dict(recipe.stage_prompt_ids),
                },
                profile_store=profile_store,
                cache_root=cache_root,
            )

        return respond(project_store.update(updater))

    return {
        "agent.update_workspace": update_workspace,
        "agent.create_conversation": create_conversation,
        "agent.switch_conversation": switch_conversation,
        "agent.rename_conversation": rename_conversation,
        "agent.delete_conversation": delete_conversation,
        "agent.update_memory": update_memory,
        "agent.delete_memory": delete_memory,
        "agent.create_recipe": create_recipe,
        "agent.update_recipe": update_recipe,
        "agent.delete_recipe": delete_recipe,
        "agent.apply_recipe": apply_recipe,
    }


__all__ = ["build_state_handlers"]
