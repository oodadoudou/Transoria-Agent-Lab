"""``agent.*`` bridge handlers for the experimental Agent Lab surface."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from transoria.agent.project_store import AgentProjectStore
from transoria.bridge.handlers.agent_chat_handlers import (
    build_chat_handlers as build_agent_chat_handlers,
)
from transoria.bridge.handlers.agent_read_handlers import (
    build_read_handlers as build_agent_read_handlers,
)
from transoria.bridge.handlers.agent_state_handlers import (
    build_state_handlers as build_agent_state_handlers,
)
from transoria.bridge.router import BridgeRouter
from transoria.bridge.task_service import TaskService
from transoria.llm.client import LlmClient
from transoria.model_profiles import ModelProfileStore
from transoria.settings import SettingsStore
from transoria.workflows.agent.task_starts import start_agent_task

LlmClientFactory = Callable[[], LlmClient]


def _build_handlers(
    *,
    cache_root: Path,
    project_store: AgentProjectStore,
    profile_store: ModelProfileStore,
    settings_store: SettingsStore,
    task_service: TaskService,
    llm_client_factory: LlmClientFactory,
) -> dict[str, object]:
    return {
        **build_agent_read_handlers(
            cache_root=cache_root,
            project_store=project_store,
            profile_store=profile_store,
            task_service=task_service,
        ),
        **build_agent_state_handlers(
            cache_root=cache_root,
            project_store=project_store,
            profile_store=profile_store,
        ),
        **build_agent_chat_handlers(
            cache_root=cache_root,
            project_store=project_store,
            profile_store=profile_store,
            settings_store=settings_store,
            task_service=task_service,
            llm_client_factory=llm_client_factory,
            start_task=start_agent_task,
        ),
    }


def register(
    router: BridgeRouter,
    *,
    cache_root: Path,
    profile_store: ModelProfileStore,
    settings_store: SettingsStore,
    task_service: TaskService,
    llm_client_factory: LlmClientFactory,
) -> None:
    handlers = _build_handlers(
        cache_root=cache_root,
        project_store=AgentProjectStore.from_cache_root(cache_root),
        profile_store=profile_store,
        settings_store=settings_store,
        task_service=task_service,
        llm_client_factory=llm_client_factory,
    )
    for method, handler in handlers.items():
        router.register(method, handler)  # type: ignore[arg-type]


__all__ = ["register"]
