"""Project-local persistence for the experimental Agent Lab workspace."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from transoria.agent.schemas import AgentProject, AgentWorkspaceState

_WORKSPACE_FILENAME = "workspace.json"
_PROJECT_FILENAME = "project.json"
_PROJECTS_DIRNAME = "agent_projects"


@dataclass(frozen=True)
class AgentProjectStore:
    root: Path

    @classmethod
    def from_cache_root(cls, cache_root: Path) -> "AgentProjectStore":
        return cls(root=cache_root / "agent_lab")

    @property
    def workspace_path(self) -> Path:
        return self.root / _WORKSPACE_FILENAME

    def load(self) -> AgentWorkspaceState:
        if not self.workspace_path.exists():
            return AgentWorkspaceState.empty()
        raw = self.workspace_path.read_text(encoding="utf-8")
        if not raw.strip():
            return AgentWorkspaceState.empty()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Agent workspace must contain a JSON object: {self.workspace_path}"
            )
        state = AgentWorkspaceState.from_dict(payload)
        if not state.conversations:
            return AgentWorkspaceState.empty()
        return state

    def save(self, state: AgentWorkspaceState) -> AgentWorkspaceState:
        self.root.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(self.workspace_path, state.to_dict())
        return state

    def update(
        self, updater: Callable[[AgentWorkspaceState], AgentWorkspaceState]
    ) -> AgentWorkspaceState:
        return self.save(updater(self.load()))


class ProjectNotFoundError(KeyError):
    """Raised when a project_id has no record on disk."""


@dataclass(frozen=True)
class ProjectRecordStore:
    """Per-project JSON files under ``<cache_root>/agent_projects/<id>/``."""

    root: Path

    @classmethod
    def from_cache_root(cls, cache_root: Path) -> "ProjectRecordStore":
        return cls(root=cache_root / _PROJECTS_DIRNAME)

    def project_dir(self, project_id: str) -> Path:
        return self.root / project_id

    def project_path(self, project_id: str) -> Path:
        return self.project_dir(project_id) / _PROJECT_FILENAME

    def exists(self, project_id: str) -> bool:
        return self.project_path(project_id).exists()

    def load(self, project_id: str) -> AgentProject:
        path = self.project_path(project_id)
        if not path.exists():
            raise ProjectNotFoundError(project_id)
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            raise ProjectNotFoundError(project_id)
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Agent project must contain a JSON object: {path}"
            )
        return AgentProject.from_dict(payload)

    def save(self, project: AgentProject) -> AgentProject:
        self.project_dir(project.id).mkdir(parents=True, exist_ok=True)
        _atomic_write_json(self.project_path(project.id), project.to_dict())
        return project


def _atomic_write_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)
