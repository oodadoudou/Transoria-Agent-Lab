from __future__ import annotations

from pathlib import Path

from transoria.agent.schemas import AgentWorkspaceState
from transoria.bridge.handlers.settings import default_store
from transoria.domain import TaskKind, TaskStatus
from transoria.llm.config import ModelConfig, ProviderFormat
from transoria.model_profiles import ModelProfileStore
from transoria.prompts import DEFAULT_GLOSSARY_REVIEW_PRESET_ID
from transoria.runtime.task_record import TaskRecord
from transoria.workflows.agent.task_starts import start_agent_task
from transoria.workflows.glossary_review.config import GlossaryReviewConfig


class _FakeTaskCache:
    def load_record(self, task_id: str) -> TaskRecord:
        return TaskRecord(
            id=task_id,
            kind=TaskKind.GLOSSARY,
            status=TaskStatus.COMPLETED,
        )


class _FakeTaskService:
    cache = _FakeTaskCache()

    def __init__(self, artifact: dict[str, object]) -> None:
        self.artifact = artifact
        self.started_config: GlossaryReviewConfig | None = None

    def read_artifacts(self, *, kind: str, task_id: str) -> dict[str, object]:
        assert kind == "glossary"
        assert task_id == "glossary-1"
        return {"combined_artifact": self.artifact}

    def start_glossary_review_with_config(
        self,
        config: GlossaryReviewConfig,
        *,
        request_id: str,
    ) -> dict[str, object]:
        self.started_config = config
        return {
            "task_id": "glossary-review-1",
            "started_at": "2026-01-01T00:00:00+00:00",
        }


def test_start_glossary_review_prepares_cache_local_input_from_task_id(
    tmp_path: Path,
) -> None:
    profile_store = ModelProfileStore.from_cache_root(tmp_path)
    profile_store.create(
        ModelConfig(
            id="profile-review",
            display_name="Review",
            provider_format=ProviderFormat.OPENAI,
            base_url="https://example.com/v1",
            model_id="review-model",
            api_keys=("key",),
        )
    )
    source_dir = tmp_path / "glossary-output"
    source_dir.mkdir()
    xlsx_path = source_dir / "Novel-glossary.xlsx"
    refs_path = source_dir / "Novel-glossary-references.txt"
    xlsx_path.write_bytes(b"placeholder")
    refs_path.write_text("reference snippets", encoding="utf-8")
    service = _FakeTaskService(
        {
            "xlsx_path": str(xlsx_path),
            "references_path": str(refs_path),
        }
    )
    state = AgentWorkspaceState.empty().with_config(
        stage_model_ids={"term_review": "profile-review"},
        stage_prompt_ids={"term_review": DEFAULT_GLOSSARY_REVIEW_PRESET_ID},
    )

    result = start_agent_task(
        draft_kind="start_glossary_review_task",
        payload={"glossary_task_id": "glossary-1"},
        state=state,
        task_service=service,  # type: ignore[arg-type]
        settings_store=default_store(tmp_path),
        profile_store=profile_store,
        cache_root=tmp_path,
        request_id="draft-1",
    )

    assert result["task_id"] == "glossary-review-1"
    assert service.started_config is not None
    config = service.started_config
    assert config.input_dir == tmp_path / "agent_lab" / "review_inputs" / (
        "glossary-1-draft-1"
    )
    assert config.selected_xlsx_path == config.input_dir / xlsx_path.name
    assert config.selected_xlsx_path.read_bytes() == b"placeholder"
    assert config.selected_reference_paths == (config.input_dir / refs_path.name,)
    assert config.selected_reference_paths[0].read_text(
        encoding="utf-8"
    ) == "reference snippets"
