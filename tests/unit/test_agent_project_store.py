from __future__ import annotations

import json
from pathlib import Path

import pytest

from transoria.agent.project_store import AgentProjectStore
from transoria.agent.schemas import AgentWorkspaceState


def test_load_returns_empty_when_file_missing(tmp_path: Path) -> None:
    store = AgentProjectStore.from_cache_root(tmp_path)
    state = store.load()
    assert len(state.conversations) == 1


def test_load_returns_empty_when_file_blank(tmp_path: Path) -> None:
    store = AgentProjectStore.from_cache_root(tmp_path)
    store.workspace_path.parent.mkdir(parents=True, exist_ok=True)
    store.workspace_path.write_text("   \n", encoding="utf-8")
    state = store.load()
    assert len(state.conversations) == 1


def test_load_rejects_non_object_payload(tmp_path: Path) -> None:
    store = AgentProjectStore.from_cache_root(tmp_path)
    store.workspace_path.parent.mkdir(parents=True, exist_ok=True)
    store.workspace_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError):
        store.load()


def test_load_reseeds_when_no_conversations(tmp_path: Path) -> None:
    store = AgentProjectStore.from_cache_root(tmp_path)
    store.workspace_path.parent.mkdir(parents=True, exist_ok=True)
    store.workspace_path.write_text(
        json.dumps({"workflow_model_id": None, "conversations": []}),
        encoding="utf-8",
    )
    state = store.load()
    assert len(state.conversations) == 1


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    store = AgentProjectStore.from_cache_root(tmp_path)
    saved = store.save(AgentWorkspaceState.empty().with_memories(("a",)))
    reloaded = store.load()
    assert reloaded.memories == ("a",)
    assert reloaded.active_conversation_id == saved.active_conversation_id


# -------- ProjectRecordStore (Phase A scaffolding) --------

import pytest as _pytest  # noqa: PT013, F401  (already imported above)

from transoria.agent.project_store import ProjectNotFoundError, ProjectRecordStore  # noqa: E402
from transoria.agent.schemas import AgentProject  # noqa: E402


def test_record_store_save_then_load_round_trips(tmp_path: Path) -> None:
    store = ProjectRecordStore.from_cache_root(tmp_path)
    project = AgentProject.create(name="N", input_dir="/abs", source_language="kr")
    saved = store.save(project)
    reloaded = store.load(saved.id)
    assert reloaded == saved
    assert store.exists(saved.id) is True


def test_record_store_load_missing_raises(tmp_path: Path) -> None:
    store = ProjectRecordStore.from_cache_root(tmp_path)
    assert store.exists("missing") is False
    with pytest.raises(ProjectNotFoundError):
        store.load("missing")


def test_record_store_load_empty_file_raises(tmp_path: Path) -> None:
    store = ProjectRecordStore.from_cache_root(tmp_path)
    target = store.project_path("proj-blank")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("   ", encoding="utf-8")
    with pytest.raises(ProjectNotFoundError):
        store.load("proj-blank")


def test_record_store_rejects_non_object_payload(tmp_path: Path) -> None:
    store = ProjectRecordStore.from_cache_root(tmp_path)
    target = store.project_path("proj-bad")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError):
        store.load("proj-bad")
