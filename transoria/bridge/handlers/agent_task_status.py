"""Task status helpers for the Agent Lab bridge surface."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Callable

from transoria.agent.schemas import AgentActionDraft
from transoria.bridge.errors import BridgeError
from transoria.bridge.task_service import TaskService
from transoria.domain import TaskKind, TaskStatus
from transoria.runtime.cache import TaskNotFoundError

AGENT_TASK_KINDS: tuple[str, ...] = (
    "translation",
    "glossary",
    "glossary_review",
)


def validate_agent_task_kind(kind: str) -> None:
    if kind not in AGENT_TASK_KINDS:
        raise BridgeError.invalid_argument(
            f"unsupported agent task kind: {kind!r}",
            field="kind",
        )


def applied_draft_message(
    draft: AgentActionDraft,
    result: Mapping[str, object],
    *,
    task_service: TaskService,
) -> str:
    task_summaries = task_start_summaries_from_result(
        result,
        task_service=task_service,
    )
    if not task_summaries:
        return f"已应用草案：{draft.title}"
    if len(task_summaries) == 1:
        summary = task_summaries[0]
        return (
            f"已应用草案：{draft.title}\n\n"
            f"已启动{summary['label']}任务。\n"
            f"任务 ID：{summary['task_id']}\n"
            f"当前进度阶段：{summary['status']}。\n"
            f"你可以在 {summary['dashboard']} 查看实时进度。任务结束前，Agent Lab 不会再启动其他任务。"
        )
    lines = [f"已应用草案：{draft.title}", "", "已启动以下任务："]
    for summary in task_summaries:
        lines.append(
            f"- {summary['label']}：{summary['task_id']}，当前阶段：{summary['status']}，"
            f"可在 {summary['dashboard']} 查看。"
        )
    lines.append("任务结束前，Agent Lab 不会再启动其他任务。")
    return "\n".join(lines)


def task_start_summaries_from_result(
    result: Mapping[str, object],
    *,
    task_service: TaskService,
) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    if isinstance(result.get("results"), list):
        for item in result["results"]:  # type: ignore[index]
            if isinstance(item, Mapping):
                summaries.extend(
                    task_start_summaries_from_result(
                        item,
                        task_service=task_service,
                    )
                )
        return summaries
    task = result.get("task")
    if not isinstance(task, Mapping):
        return []
    task_id = str(task.get("task_id") or "")
    task_kind = str(task.get("kind") or "")
    if not task_id or task_kind not in AGENT_TASK_KINDS:
        return []
    header = task_header_or_none(task_service, task_kind, task_id)
    status = (
        task_status_label(str(header["status"]))
        if header is not None and isinstance(header.get("status"), str)
        else "已提交，等待任务状态刷新"
    )
    summaries.append(
        {
            "task_id": task_id,
            "kind": task_kind,
            "label": task_kind_label(task_kind),
            "dashboard": task_dashboard_label(task_kind),
            "status": status,
        }
    )
    return summaries


def task_kind_label(kind: str) -> str:
    return {
        "translation": "翻译",
        "glossary": "术语提取",
        "glossary_review": "术语审查",
    }.get(kind, kind)


def task_dashboard_label(kind: str) -> str:
    return {
        "translation": "翻译 dashboard",
        "glossary": "术语提取 dashboard",
        "glossary_review": "术语审查 dashboard",
    }.get(kind, "对应 dashboard")


def task_status_label(status: str) -> str:
    return {
        "pending": "已提交，等待运行",
        "running": "正在运行",
        "paused": "已暂停",
        "stopping": "正在停止",
        "completed": "已完成",
        "failed": "已失败",
        "cancelled": "已取消",
        "stopped": "已停止",
    }.get(status, status or "已提交，等待任务状态刷新")


def task_header_or_none(
    task_service: TaskService,
    kind: str,
    task_id: str,
) -> dict[str, object] | None:
    try:
        record = task_service.cache.load_record(task_id)
    except (TaskNotFoundError, ValueError, OSError):
        return None
    try:
        expected = TaskKind(kind)
    except ValueError:
        return None
    if record.kind is not expected:
        return None
    return {
        "id": record.id,
        "kind": record.kind.value,
        "status": record.status.value,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def recent_task_summaries(
    task_service: TaskService,
    *,
    kind: str,
    limit: int,
) -> list[dict[str, object]]:
    listing = task_service.list_recent_tasks(kind=kind, limit=limit)
    tasks = listing.get("tasks")
    if not isinstance(tasks, list):
        return []
    summaries: list[dict[str, object]] = []
    for item in tasks:
        if not isinstance(item, Mapping):
            continue
        task_id = str(item.get("id", "")).strip()
        if not task_id:
            continue
        availability = artifact_availability(
            task_service,
            kind=kind,
            task_id=task_id,
        )
        summaries.append(
            {
                "id": task_id,
                "kind": str(item.get("kind") or kind),
                "status": str(item.get("status") or ""),
                "created_at": str(item.get("created_at") or ""),
                "updated_at": str(item.get("updated_at") or ""),
                "artifact_available": bool(availability["available"]),
                "artifact_keys": list(availability["artifact_keys"]),
            }
        )
    return summaries


def task_status_context(task_service: TaskService) -> dict[str, object]:
    return {
        "recent_task_summaries": {
            task_kind: recent_task_headers(
                task_service,
                kind=task_kind,
                limit=3,
            )
            for task_kind in AGENT_TASK_KINDS
        }
    }


def recent_task_headers(
    task_service: TaskService,
    *,
    kind: str,
    limit: int,
) -> list[dict[str, object]]:
    listing = task_service.list_recent_tasks(kind=kind, limit=limit)
    tasks = listing.get("tasks")
    if not isinstance(tasks, list):
        return []
    summaries: list[dict[str, object]] = []
    for item in tasks:
        if not isinstance(item, Mapping):
            continue
        task_id = str(item.get("id", "")).strip()
        if not task_id:
            continue
        availability = artifact_availability(
            task_service,
            kind=kind,
            task_id=task_id,
        )
        summaries.append(
            {
                "id": task_id,
                "kind": str(item.get("kind") or kind),
                "status": str(item.get("status") or ""),
                "created_at": str(item.get("created_at") or ""),
                "updated_at": str(item.get("updated_at") or ""),
                "artifact_available": bool(availability.get("available")),
            }
        )
    return summaries


def artifact_availability(
    task_service: TaskService,
    *,
    kind: str,
    task_id: str,
) -> dict[str, object]:
    try:
        payload = task_service.read_artifacts(kind=kind, task_id=task_id)
    except BridgeError as exc:
        return {
            "kind": kind,
            "task_id": task_id,
            "available": False,
            "artifact_keys": [],
            "error": {
                "code": exc.code,
                "message": str(exc),
            },
        }
    return {
        "kind": kind,
        "task_id": task_id,
        "available": True,
        "artifact_keys": sorted(str(key) for key in payload.keys()),
    }


def direct_stage_status_response(
    *,
    user_message: str,
    current_state: Mapping[str, object],
    task_service: TaskService,
    extract_glossary_task_id: Callable[[str], str | None],
    extract_glossary_review_task_id: Callable[[str], str | None],
) -> tuple[str, None] | None:
    direct = direct_glossary_stage_status_response(
        user_message=user_message,
        current_state=current_state,
        task_service=task_service,
        extract_glossary_task_id=extract_glossary_task_id,
        extract_glossary_review_task_id=extract_glossary_review_task_id,
    )
    if direct is not None:
        return direct
    return direct_translation_status_response(
        user_message=user_message,
        current_state=current_state,
        task_service=task_service,
    )


def direct_glossary_stage_status_response(
    *,
    user_message: str,
    current_state: Mapping[str, object],
    task_service: TaskService,
    extract_glossary_task_id: Callable[[str], str | None],
    extract_glossary_review_task_id: Callable[[str], str | None],
) -> tuple[str, None] | None:
    requested_kind = requested_glossary_status_kind(
        user_message,
        current_state,
        extract_glossary_task_id=extract_glossary_task_id,
        extract_glossary_review_task_id=extract_glossary_review_task_id,
    )
    if requested_kind is None:
        return None

    if requested_kind == "glossary_review":
        task_id = extract_glossary_review_task_id(user_message) or latest_task_id(
            current_state,
            kind="glossary_review",
        )
        label = "术语审查"
        missing = "我还没有找到最近的术语审查任务记录。请提供 glossary-review- 开头的任务 ID，或先启动术语审查。"
        task_kind = TaskKind.GLOSSARY_REVIEW
    else:
        task_id = extract_glossary_task_id(user_message) or latest_task_id(
            current_state,
            kind="glossary",
        )
        label = "术语提取"
        missing = "我还没有找到最近的术语提取任务记录。请提供 glossary- 开头的任务 ID，或先启动术语提取。"
        task_kind = TaskKind.GLOSSARY
    if not task_id:
        return missing, None

    try:
        record = task_service.cache.load_record(task_id)
    except (TaskNotFoundError, ValueError, OSError):
        return (f"我没有找到{label}任务 {task_id}。请确认任务 ID 是否正确。", None)
    if record.kind is not task_kind:
        return (f"{task_id} 不是{label}任务，不能按{label}进度读取。", None)
    status = record.status.value

    artifacts: Mapping[str, object] = {}
    try:
        raw_artifacts = task_service.read_artifacts(kind=requested_kind, task_id=task_id)
        if isinstance(raw_artifacts, Mapping):
            artifacts = raw_artifacts
    except BridgeError:
        artifacts = {}

    if requested_kind == "glossary_review":
        return format_glossary_review_status_response(
            task_id=task_id,
            status=status,
            artifacts=artifacts,
        ), None
    return format_glossary_status_response(
        task_id=task_id,
        status=status,
        artifacts=artifacts,
    ), None


def direct_translation_status_response(
    *,
    user_message: str,
    current_state: Mapping[str, object],
    task_service: TaskService,
) -> tuple[str, None] | None:
    if not looks_like_translation_status_query(user_message):
        return None
    task_id = extract_translation_task_id(user_message) or latest_translation_task_id(
        current_state
    )
    if not task_id:
        return (
            "我还没有找到最近的翻译任务记录。请提供 translation- 开头的任务 ID，或先启动一次翻译任务。",
            None,
        )
    try:
        record = task_service.cache.load_record(task_id)
    except (TaskNotFoundError, ValueError, OSError):
        return (f"我没有找到翻译任务 {task_id}。请确认任务 ID 是否正确。", None)
    if record.kind is not TaskKind.TRANSLATION:
        return (f"{task_id} 不是翻译任务，不能按翻译进度读取。", None)

    artifacts: Mapping[str, object] = {}
    try:
        raw_artifacts = task_service.read_artifacts(kind="translation", task_id=task_id)
        if isinstance(raw_artifacts, Mapping):
            artifacts = raw_artifacts
    except BridgeError:
        artifacts = {}
    statistics = read_translation_statistics(artifacts)
    completed_segments = coerce_int(
        statistics.get("completed_segments")
        if statistics
        else artifacts.get("completed_segments")
    )
    total_segments = coerce_int(
        statistics.get("total_segments") if statistics else artifacts.get("total_segments")
    )
    failed_subtasks = coerce_int(statistics.get("failed_subtasks") if statistics else None)
    low_confidence_raw = statistics.get("low_confidence_segments") if statistics else None
    low_confidence_total = low_confidence_count(low_confidence_raw)
    low_confidence_sample = low_confidence_examples(low_confidence_raw)
    translated_files = string_list(
        artifacts.get("translated_files")
        or artifacts.get("output_files")
        or (statistics.get("translated_outputs") if statistics else None)
    )

    status = record.status.value
    lines = [f"最近的翻译任务 {task_id} 当前状态：{task_status_label(status)}。"]
    if completed_segments is not None and total_segments is not None:
        lines.append(f"进度：{completed_segments}/{total_segments} 段。")
    elif completed_segments is not None:
        lines.append(f"已完成段数：{completed_segments}。")
    if failed_subtasks is not None:
        lines.append(f"失败子任务：{failed_subtasks}。")
    if low_confidence_total is not None:
        detail = f"低置信：{low_confidence_total} 条"
        if low_confidence_sample:
            detail += f"（示例：{', '.join(low_confidence_sample)}）"
        lines.append(detail + "。")
    else:
        lines.append("我没有在该任务的统计文件里读到低置信数量。")
    if translated_files:
        lines.append(f"输出文件：{translated_files[0]}")
    if status == TaskStatus.COMPLETED.value:
        lines.append(
            "下一步建议进入「翻译 > 校对」页面，选择这个任务，优先处理低置信、原文残留和疑似重复问题。"
        )
    elif status in {TaskStatus.RUNNING.value, TaskStatus.PENDING.value}:
        lines.append("你可以在翻译 dashboard 查看实时进度；任务结束前 Agent Lab 不会再启动其他任务。")
    else:
        lines.append("如果需要继续处理，请先在对应 dashboard 查看错误或产物状态。")
    return "\n".join(lines), None


def requested_glossary_status_kind(
    text: str,
    current_state: Mapping[str, object],
    *,
    extract_glossary_task_id: Callable[[str], str | None],
    extract_glossary_review_task_id: Callable[[str], str | None],
) -> str | None:
    normalized = text.lower()
    if not looks_like_stage_status_query(normalized):
        return None
    if extract_glossary_review_task_id(text):
        return "glossary_review"
    if extract_glossary_task_id(text):
        return "glossary"
    has_review_context = any(
        marker in normalized
        for marker in (
            "术语审查",
            "术语审核",
            "术语复审",
            "审核术语",
            "审查术语",
            "复审术语",
            "glossary review",
            "review glossary",
            "glossary-review",
        )
    )
    has_extract_context = any(
        marker in normalized
        for marker in (
            "术语提取",
            "术语抽取",
            "提取术语",
            "抽取术语",
            "处理术语",
            "整理术语",
            "glossary extraction",
            "glossary task",
        )
    )
    if has_review_context:
        return "glossary_review"
    if has_extract_context:
        return "glossary"
    if "术语" not in normalized and "glossary" not in normalized:
        return None
    return latest_status_kind(current_state, kinds=("glossary_review", "glossary"))


def looks_like_stage_status_query(normalized: str) -> bool:
    return any(
        marker in normalized
        for marker in (
            "完成了吗",
            "完成了没",
            "做完了吗",
            "结束了吗",
            "结束了没",
            "是否完成",
            "是否结束",
            "状态",
            "进度",
            "下一步",
            "现在到哪",
            "进行到哪",
            "做到哪",
            "产物",
            "结果",
            "报告",
            "哪里看",
            "去哪个页面",
            "去哪",
            "status",
            "progress",
            "next",
            "done?",
            "finished?",
        )
    )


def latest_status_kind(
    current_state: Mapping[str, object],
    *,
    kinds: tuple[str, ...],
) -> str | None:
    recent_by_kind = current_state.get("recent_task_summaries")
    if not isinstance(recent_by_kind, Mapping):
        return None
    best_kind: str | None = None
    best_updated = ""
    for kind in kinds:
        tasks = recent_by_kind.get(kind)
        if not isinstance(tasks, list) or not tasks:
            continue
        item = tasks[0]
        if not isinstance(item, Mapping):
            continue
        task_id = str(item.get("id") or "").strip()
        if not task_id:
            continue
        updated_at = str(item.get("updated_at") or item.get("created_at") or "")
        if best_kind is None or updated_at >= best_updated:
            best_kind = kind
            best_updated = updated_at
    return best_kind


def latest_task_id(current_state: Mapping[str, object], *, kind: str) -> str | None:
    recent_by_kind = current_state.get("recent_task_summaries")
    if not isinstance(recent_by_kind, Mapping):
        return None
    tasks = recent_by_kind.get(kind)
    if not isinstance(tasks, list):
        return None
    for item in tasks:
        if not isinstance(item, Mapping):
            continue
        task_id = str(item.get("id") or "").strip()
        if task_id:
            return task_id
    return None


def format_glossary_status_response(
    *,
    task_id: str,
    status: str,
    artifacts: Mapping[str, object],
) -> str:
    statistics = read_json_artifact(artifacts, "statistics_json_path")
    candidate_count = coerce_int(statistics.get("candidate_count"))
    final_entry_count = coerce_int(statistics.get("final_entry_count"))
    processed_files = string_list(statistics.get("processed_files"))
    per_novel = artifacts.get("per_novel_artifacts")
    artifact_count = len(per_novel) if isinstance(per_novel, list) else None
    combined = artifacts.get("combined_artifact")
    combined_xlsx = None
    if isinstance(combined, Mapping):
        raw_path = combined.get("xlsx_path")
        if isinstance(raw_path, str) and raw_path.strip():
            combined_xlsx = raw_path.strip()

    lines = [f"最近的术语提取任务 {task_id} 当前状态：{task_status_label(status)}。"]
    if processed_files:
        lines.append(f"处理文件：{len(processed_files)} 个。")
    if candidate_count is not None:
        lines.append(f"候选术语：{candidate_count} 条。")
    if final_entry_count is not None:
        lines.append(f"最终术语：{final_entry_count} 条。")
    elif artifact_count is not None:
        lines.append(f"已生成术语产物：{artifact_count} 份。")
    if combined_xlsx:
        lines.append(f"合并术语表：{combined_xlsx}")
    if status == TaskStatus.COMPLETED.value:
        lines.append(
            "下一步建议启动「术语审查」任务。我可以基于这个 glossary task ID 生成术语审查草案，仍然需要你确认后才会执行。"
        )
    elif status in {TaskStatus.RUNNING.value, TaskStatus.PENDING.value}:
        lines.append("你可以在术语提取 dashboard 查看实时进度；任务结束前 Agent Lab 不会再启动其他任务。")
    else:
        lines.append("如果需要继续处理，请先在术语提取 dashboard 查看错误和产物状态。")
    return "\n".join(lines)


def format_glossary_review_status_response(
    *,
    task_id: str,
    status: str,
    artifacts: Mapping[str, object],
) -> str:
    changed_count = coerce_int(artifacts.get("changed_count"))
    output_path = artifacts.get("output_path")
    report_path = artifacts.get("report_path")
    lines = [f"最近的术语审查任务 {task_id} 当前状态：{task_status_label(status)}。"]
    if changed_count is not None:
        lines.append(f"模型审查修改：{changed_count} 条。")
    if isinstance(output_path, str) and output_path.strip():
        lines.append(f"确认术语表：{output_path.strip()}")
    if isinstance(report_path, str) and report_path.strip():
        lines.append(f"审查报告：{report_path.strip()}")
    if status == TaskStatus.COMPLETED.value:
        lines.append(
            "下一步建议进入「术语审查」页面的数据表校对区，人工确认术语表。确认无误后，可以在聊天里说“术语表已确认，用它翻译”，我再生成翻译任务草案。"
        )
    elif status in {TaskStatus.RUNNING.value, TaskStatus.PENDING.value}:
        lines.append("你可以在术语审查 dashboard 查看实时进度；任务结束前 Agent Lab 不会再启动其他任务。")
    else:
        lines.append("如果需要继续处理，请先在术语审查 dashboard 查看错误、报告或最终表状态。")
    return "\n".join(lines)


def looks_like_translation_status_query(text: str) -> bool:
    normalized = text.lower()
    has_translation_context = (
        "翻译" in normalized
        or "translation-" in normalized
        or "低置信" in normalized
        or "原文残留" in normalized
        or "校对" in normalized
        or "proofread" in normalized
    )
    if not has_translation_context:
        return False
    return any(
        marker in normalized
        for marker in (
            "完成",
            "结束",
            "状态",
            "进度",
            "低置信",
            "原文残留",
            "下一步",
            "哪里校对",
            "去哪校对",
            "校对页",
            "proofread",
            "status",
            "progress",
        )
    )


def extract_translation_task_id(text: str) -> str | None:
    match = re.search(r"\b(translation-[a-zA-Z0-9-]+)\b", text)
    if match is None:
        return None
    return match.group(1).strip()


def latest_translation_task_id(current_state: Mapping[str, object]) -> str | None:
    return latest_task_id(current_state, kind="translation")


def read_translation_statistics(
    artifacts: Mapping[str, object],
) -> Mapping[str, object]:
    return read_json_artifact(artifacts, "statistics_json_path")


def read_json_artifact(
    artifacts: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    statistics_path = artifacts.get(key)
    if not isinstance(statistics_path, str) or not statistics_path.strip():
        return {}
    path = Path(statistics_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, Mapping) else {}


def low_confidence_count(value: object) -> int | None:
    if isinstance(value, list):
        return len(value)
    return coerce_int(value)


def low_confidence_examples(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    examples: list[str] = []
    for item in value[:3]:
        if isinstance(item, Mapping):
            segment_id = str(item.get("segment_id") or "").strip()
            if segment_id:
                examples.append(segment_id)
        elif isinstance(item, str) and item.strip():
            examples.append(item.strip())
    return examples


def coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]
