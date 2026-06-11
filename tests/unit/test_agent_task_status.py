from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from transoria.agent.schemas import AgentActionDraft
from transoria.bridge.errors import BridgeError
from transoria.bridge.handlers.agent_task_status import (
    applied_draft_message,
    artifact_availability,
    direct_glossary_stage_status_response,
    direct_translation_status_response,
    extract_translation_task_id,
    format_glossary_status_response,
    latest_status_kind,
    looks_like_translation_status_query,
    recent_task_summaries,
    validate_agent_task_kind,
)
from transoria.domain import TaskKind, TaskStatus


@dataclass(frozen=True)
class _Record:
    id: str
    kind: TaskKind
    status: TaskStatus
    created_at: str = "2026-01-01T00:00:00Z"
    updated_at: str = "2026-01-01T00:01:00Z"


class _Cache:
    def __init__(self, records: dict[str, _Record]) -> None:
        self._records = records

    def load_record(self, task_id: str) -> _Record:
        record = self._records.get(task_id)
        if record is None:
            raise ValueError(task_id)
        return record


class _TaskService:
    def __init__(
        self,
        *,
        records: dict[str, _Record] | None = None,
        tasks: list[dict[str, object]] | None = None,
        artifacts: dict[str, dict[str, object]] | None = None,
    ) -> None:
        self.cache = _Cache(records or {})
        self._tasks = tasks or []
        self._artifacts = artifacts or {}

    def list_recent_tasks(self, *, kind: str, limit: int) -> dict[str, object]:
        return {
            "tasks": [
                item
                for item in self._tasks
                if str(item.get("kind") or kind) == kind
            ][:limit]
        }

    def read_artifacts(self, *, kind: str, task_id: str) -> dict[str, object]:
        if task_id not in self._artifacts:
            raise BridgeError.not_found(f"{kind} task {task_id!r} has no artifacts.")
        return self._artifacts[task_id]


def test_applied_draft_message_includes_started_task_dashboard() -> None:
    draft = AgentActionDraft.create(
        kind="start_glossary_task",
        title="启动术语提取",
        summary="",
        payload={},
    )
    task_service = _TaskService(
        records={
            "task-1": _Record(
                id="task-1",
                kind=TaskKind.GLOSSARY,
                status=TaskStatus.RUNNING,
            )
        }
    )

    message = applied_draft_message(
        draft,
        {"task": {"task_id": "task-1", "kind": "glossary"}},
        task_service=task_service,  # type: ignore[arg-type]
    )

    assert "已应用草案：启动术语提取" in message
    assert "已启动术语提取任务" in message
    assert "任务 ID：task-1" in message
    assert "当前进度阶段：正在运行" in message
    assert "术语提取 dashboard" in message


def test_applied_draft_message_flattens_nested_compound_task_results() -> None:
    draft = AgentActionDraft.create(
        kind="compound_config_update",
        title="启动多个任务",
        summary="",
        payload={},
    )
    task_service = _TaskService(
        records={
            "task-g": _Record(
                id="task-g",
                kind=TaskKind.GLOSSARY,
                status=TaskStatus.PENDING,
            ),
            "task-t": _Record(
                id="task-t",
                kind=TaskKind.TRANSLATION,
                status=TaskStatus.RUNNING,
            ),
        }
    )

    message = applied_draft_message(
        draft,
        {
            "results": [
                {"task": {"task_id": "task-g", "kind": "glossary"}},
                {"ignored": True},
                {
                    "results": [
                        {"task": {"task_id": "task-t", "kind": "translation"}}
                    ]
                },
            ]
        },
        task_service=task_service,  # type: ignore[arg-type]
    )

    assert "已启动以下任务" in message
    assert "术语提取：task-g" in message
    assert "翻译：task-t" in message


def test_recent_task_summaries_include_artifact_keys() -> None:
    task_service = _TaskService(
        tasks=[
            {
                "id": "task-1",
                "kind": "translation",
                "status": "completed",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:02:00Z",
            }
        ],
        artifacts={"task-1": {"report": {}, "output": {}}},
    )

    summaries = recent_task_summaries(
        task_service,  # type: ignore[arg-type]
        kind="translation",
        limit=5,
    )

    assert summaries == [
        {
            "id": "task-1",
            "kind": "translation",
            "status": "completed",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:02:00Z",
            "artifact_available": True,
            "artifact_keys": ["output", "report"],
        }
    ]


def test_artifact_availability_returns_error_for_missing_task() -> None:
    availability = artifact_availability(
        _TaskService(),  # type: ignore[arg-type]
        kind="translation",
        task_id="missing",
    )

    assert availability["available"] is False
    assert availability["artifact_keys"] == []
    assert availability["error"] == {
        "code": "bridge.not_found",
        "message": "translation task 'missing' has no artifacts.",
    }


def test_validate_agent_task_kind_rejects_unknown_kind() -> None:
    with pytest.raises(BridgeError) as exc_info:
        validate_agent_task_kind("replacement")

    assert exc_info.value.code == "bridge.invalid_argument"
    assert "unsupported agent task kind" in str(exc_info.value)


def test_format_glossary_status_response_reads_statistics_artifact(
    tmp_path: Path,
) -> None:
    statistics_path = tmp_path / "statistics.json"
    statistics_path.write_text(
        json.dumps(
            {
                "candidate_count": 11,
                "final_entry_count": 7,
                "processed_files": ["a.epub", "b.epub"],
            }
        ),
        encoding="utf-8",
    )

    message = format_glossary_status_response(
        task_id="glossary-1",
        status="completed",
        artifacts={
            "statistics_json_path": str(statistics_path),
            "combined_artifact": {"xlsx_path": "/tmp/glossary.xlsx"},
        },
    )

    assert "最近的术语提取任务 glossary-1 当前状态：已完成" in message
    assert "处理文件：2 个" in message
    assert "候选术语：11 条" in message
    assert "最终术语：7 条" in message
    assert "合并术语表：/tmp/glossary.xlsx" in message
    assert "下一步建议启动「术语审查」任务" in message


def test_direct_glossary_stage_status_response_uses_latest_task() -> None:
    task_service = _TaskService(
        records={
            "glossary-1": _Record(
                id="glossary-1",
                kind=TaskKind.GLOSSARY,
                status=TaskStatus.RUNNING,
            )
        },
        artifacts={
            "glossary-1": {
                "per_novel_artifacts": [{"xlsx_path": "/tmp/a.xlsx"}],
            }
        },
    )

    response = direct_glossary_stage_status_response(
        user_message="术语提取现在到哪了？",
        current_state={
            "recent_task_summaries": {
                "glossary": [
                    {
                        "id": "glossary-1",
                        "updated_at": "2026-01-01T00:01:00Z",
                    }
                ]
            }
        },
        task_service=task_service,  # type: ignore[arg-type]
        extract_glossary_task_id=lambda text: None,
        extract_glossary_review_task_id=lambda text: None,
    )

    assert response is not None
    assert response[1] is None
    assert "glossary-1 当前状态：正在运行" in response[0]
    assert "术语提取 dashboard" in response[0]


def test_direct_translation_status_response_reads_low_confidence_stats(
    tmp_path: Path,
) -> None:
    statistics_path = tmp_path / "translation_stats.json"
    statistics_path.write_text(
        json.dumps(
            {
                "completed_segments": 8,
                "total_segments": 10,
                "failed_subtasks": 1,
                "low_confidence_segments": [
                    {"segment_id": "seg-1"},
                    {"segment_id": "seg-2"},
                    {"segment_id": "seg-3"},
                    {"segment_id": "seg-4"},
                ],
                "translated_outputs": ["/tmp/out.epub"],
            }
        ),
        encoding="utf-8",
    )
    task_service = _TaskService(
        records={
            "translation-1": _Record(
                id="translation-1",
                kind=TaskKind.TRANSLATION,
                status=TaskStatus.COMPLETED,
            )
        },
        artifacts={
            "translation-1": {"statistics_json_path": str(statistics_path)},
        },
    )

    response = direct_translation_status_response(
        user_message="translation-1 翻译完成了吗？低置信在哪里？",
        current_state={},
        task_service=task_service,  # type: ignore[arg-type]
    )

    assert response is not None
    assert response[1] is None
    assert "最近的翻译任务 translation-1 当前状态：已完成" in response[0]
    assert "进度：8/10 段" in response[0]
    assert "失败子任务：1" in response[0]
    assert "低置信：4 条（示例：seg-1, seg-2, seg-3）" in response[0]
    assert "输出文件：/tmp/out.epub" in response[0]
    assert "翻译 > 校对" in response[0]


def test_translation_status_query_helpers() -> None:
    assert looks_like_translation_status_query("翻译任务现在进度怎么样？")
    assert not looks_like_translation_status_query("帮我创建翻译 prompt")
    assert extract_translation_task_id("看看 translation-abc-123 的状态") == (
        "translation-abc-123"
    )


def test_latest_status_kind_uses_most_recent_updated_task() -> None:
    assert (
        latest_status_kind(
            {
                "recent_task_summaries": {
                    "glossary": [
                        {
                            "id": "glossary-1",
                            "updated_at": "2026-01-01T00:02:00Z",
                        }
                    ],
                    "glossary_review": [
                        {
                            "id": "glossary-review-1",
                            "updated_at": "2026-01-01T00:03:00Z",
                        }
                    ],
                }
            },
            kinds=("glossary_review", "glossary"),
        )
        == "glossary_review"
    )
