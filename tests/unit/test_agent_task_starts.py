from __future__ import annotations

from pathlib import Path

from transoria.agent.schemas import AgentWorkspaceState
from transoria.bridge.handlers.settings import default_store
from transoria.domain import TaskKind, TaskStatus
from transoria.llm.config import ModelConfig, ProviderFormat
from transoria.model_profiles import ModelProfileStore
from transoria.prompts import (
    DEFAULT_GLOSSARY_PRESET_ID,
    DEFAULT_GLOSSARY_REVIEW_PRESET_ID,
    DEFAULT_TRANSLATION_PRESET_ID,
)
from transoria.runtime.task_record import TaskRecord
from transoria.workflows.agent.task_starts import start_agent_task
from transoria.workflows.glossary.config import GlossaryConfig
from transoria.workflows.glossary_review.config import GlossaryReviewConfig
from transoria.workflows.translation.config import TranslationConfig


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


class _FakeStartOnlyTaskService:
    cache = _FakeTaskCache()

    def __init__(self) -> None:
        self.started_translation_config: TranslationConfig | None = None
        self.started_glossary_config: GlossaryConfig | None = None

    def start_translation_with_config(
        self,
        config: TranslationConfig,
        *,
        request_id: str,
    ) -> dict[str, object]:
        self.started_translation_config = config
        return {
            "task_id": "translation-1",
            "started_at": "2026-01-01T00:00:00+00:00",
        }

    def start_glossary_with_config(
        self,
        config: GlossaryConfig,
        *,
        request_id: str,
    ) -> dict[str, object]:
        self.started_glossary_config = config
        return {
            "task_id": "glossary-1",
            "started_at": "2026-01-01T00:00:00+00:00",
        }


def _seed_profile(tmp_path: Path, profile_id: str = "profile-agent") -> ModelConfig:
    store = ModelProfileStore.from_cache_root(tmp_path)
    return store.create(
        ModelConfig(
            id=profile_id,
            display_name="Agent stage model",
            provider_format=ProviderFormat.OPENAI,
            base_url="https://example.com/v1",
            model_id="stage-model",
            api_keys=("key",),
        )
    )


def test_start_translation_uses_chat_payload_without_writing_settings(
    tmp_path: Path,
) -> None:
    _seed_profile(tmp_path)
    input_dir = tmp_path / "novel"
    output_dir = tmp_path / "translated"
    input_dir.mkdir()
    (input_dir / "book.txt").write_text("source", encoding="utf-8")
    service = _FakeStartOnlyTaskService()
    state = AgentWorkspaceState.empty().with_config(
        stage_model_ids={"translation": "profile-agent"},
        stage_prompt_ids={"translation": DEFAULT_TRANSLATION_PRESET_ID},
    )

    result = start_agent_task(
        draft_kind="start_translation_task",
        payload={
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "source_language": "ja",
            "target_language": "zh",
        },
        state=state,
        task_service=service,  # type: ignore[arg-type]
        settings_store=default_store(tmp_path),
        profile_store=ModelProfileStore.from_cache_root(tmp_path),
        cache_root=tmp_path,
        request_id="draft-translation",
    )

    assert result["task_id"] == "translation-1"
    assert service.started_translation_config is not None
    config = service.started_translation_config
    assert config.input_dir == input_dir
    assert config.output_dir == output_dir
    assert config.source_language.value == "ja"
    assert config.target_language.value == "zh"
    settings = default_store(tmp_path).load_all()
    assert settings.translation.input_folder == ""
    assert settings.translation.output_folder == ""


def test_start_glossary_uses_chat_payload_without_writing_settings(
    tmp_path: Path,
) -> None:
    _seed_profile(tmp_path)
    input_dir = tmp_path / "novel"
    output_dir = tmp_path / "glossary"
    input_dir.mkdir()
    (input_dir / "book.txt").write_text("source", encoding="utf-8")
    service = _FakeStartOnlyTaskService()
    state = AgentWorkspaceState.empty().with_config(
        stage_model_ids={"term_extract": "profile-agent"},
        stage_prompt_ids={"term_extract": DEFAULT_GLOSSARY_PRESET_ID},
    )

    result = start_agent_task(
        draft_kind="start_glossary_task",
        payload={
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "source_language": "kr",
            "target_language": "zh",
            "novel_background": "宫廷权谋，术语需保持一致。",
        },
        state=state,
        task_service=service,  # type: ignore[arg-type]
        settings_store=default_store(tmp_path),
        profile_store=ModelProfileStore.from_cache_root(tmp_path),
        cache_root=tmp_path,
        request_id="draft-glossary",
    )

    assert result["task_id"] == "glossary-1"
    assert service.started_glossary_config is not None
    config = service.started_glossary_config
    assert config.input_dir == input_dir
    assert config.output_dir == output_dir
    assert config.source_language.value == "kr"
    assert config.target_language.value == "zh"
    assert config.novel_background == "宫廷权谋，术语需保持一致。"
    settings = default_store(tmp_path).load_all()
    assert settings.glossary.input_folder == ""
    assert settings.glossary.output_folder == ""
    assert settings.glossary.novel_background == ""


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
