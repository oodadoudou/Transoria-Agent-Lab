"""Project-local persistence for the experimental Agent Lab workspace."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from transoria.agent.schemas import AgentWorkspaceState

_WORKSPACE_FILENAME = "workspace.json"


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


def _atomic_write_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)
