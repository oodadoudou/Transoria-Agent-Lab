"""Read-only bridge handlers for Agent Lab workspace state."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from transoria.agent.project_store import AgentProjectStore
from transoria.bridge.errors import BridgeError
from transoria.bridge.handlers._utils import expect_string
from transoria.bridge.handlers.agent_action_registry import (
    coerce_compound_action_drafts,
)
from transoria.bridge.handlers.agent_inventory import inventory
from transoria.bridge.handlers.agent_task_status import (
    AGENT_TASK_KINDS,
    artifact_availability,
    recent_task_summaries,
    task_header_or_none,
    validate_agent_task_kind,
)
from transoria.bridge.handlers.agent_tasks import load_with_reconciled_active_task
from transoria.bridge.handlers.agent_wire import resolved_active_recipe_id
from transoria.bridge.handlers.agent_workspace import (
    optional_agent_limit,
    optional_agent_string,
    workspace_response,
)
from transoria.bridge.task_service import TaskService
from transoria.model_profiles import ModelProfileStore
from transoria.prompts import PromptKind


def build_read_handlers(
    *,
    cache_root: Path,
    project_store: AgentProjectStore,
    profile_store: ModelProfileStore,
    task_service: TaskService,
) -> dict[str, object]:
    def respond_workspace() -> dict[str, object]:
        state = load_with_reconciled_active_task(project_store, task_service)
        return workspace_response(
            state,
            profile_store,
            cache_root,
            normalize_compound_action_drafts=coerce_compound_action_drafts,
        )

    def read_workspace(_payload: Mapping[str, object]) -> dict[str, object]:
        return respond_workspace()

    def list_model_profiles(_payload: Mapping[str, object]) -> dict[str, object]:
        return {"profiles": inventory(profile_store, cache_root)["profiles"]}

    def list_prompt_presets(payload: Mapping[str, object]) -> dict[str, object]:
        prompts = inventory(profile_store, cache_root)["prompts"]
        kind = optional_agent_string(payload, "kind")
        if kind is None:
            return {"prompts": prompts}
        if kind not in {prompt_kind.value for prompt_kind in PromptKind}:
            raise BridgeError.invalid_argument(
                f"unsupported prompt kind: {kind!r}",
                field="kind",
            )
        return {"kind": kind, "presets": prompts[kind]}  # type: ignore[index]

    def list_recipes(_payload: Mapping[str, object]) -> dict[str, object]:
        state = load_with_reconciled_active_task(project_store, task_service)
        return {
            "recipes": [recipe.to_dict() for recipe in state.recipes],
            "active_recipe_id": resolved_active_recipe_id(state),
        }

    def get_active_task(_payload: Mapping[str, object]) -> dict[str, object]:
        state = load_with_reconciled_active_task(project_store, task_service)
        active = state.active_task
        if active is None:
            return {"active_task": None, "task": None}
        return {
            "active_task": active.to_dict(),
            "task": task_header_or_none(task_service, active.kind, active.task_id),
        }

    def list_recent_task_summaries(payload: Mapping[str, object]) -> dict[str, object]:
        kind = optional_agent_string(payload, "kind")
        limit = optional_agent_limit(payload, default=5)
        if kind is not None:
            validate_agent_task_kind(kind)
            return {
                "kind": kind,
                "tasks": recent_task_summaries(
                    task_service,
                    kind=kind,
                    limit=limit,
                ),
            }
        return {
            "tasks_by_kind": {
                task_kind: recent_task_summaries(
                    task_service,
                    kind=task_kind,
                    limit=limit,
                )
                for task_kind in AGENT_TASK_KINDS
            }
        }

    def get_artifact_availability(payload: Mapping[str, object]) -> dict[str, object]:
        kind = expect_string(payload, "kind").strip()
        validate_agent_task_kind(kind)
        task_id = expect_string(payload, "task_id").strip()
        if not task_id:
            raise BridgeError.invalid_argument(
                "task_id must not be empty.",
                field="task_id",
            )
        return artifact_availability(task_service, kind=kind, task_id=task_id)

    return {
        "agent.read_workspace": read_workspace,
        "agent.list_model_profiles": list_model_profiles,
        "agent.list_prompt_presets": list_prompt_presets,
        "agent.list_recipes": list_recipes,
        "agent.get_active_task": get_active_task,
        "agent.list_recent_task_summaries": list_recent_task_summaries,
        "agent.get_artifact_availability": get_artifact_availability,
    }


__all__ = ["build_read_handlers"]
