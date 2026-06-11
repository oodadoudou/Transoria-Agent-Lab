"""Task-start text intent helpers for the Agent Lab bridge handler."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

_MAX_NOVEL_BACKGROUND_LENGTH = 6000


def looks_like_translation_task_request(text: str) -> bool:
    normalized = text.lower()
    if re.search(
        r"先\s*(?:提取|抽取|处理|整理)\s*术语(?:表)?",
        normalized,
    ):
        return False
    strong_translation_markers = (
        "开始翻译",
        "启动翻译",
        "执行翻译",
        "进行翻译",
        "翻译任务",
        "翻译小说",
        "run translation",
        "start translation",
        "translate novel",
        "translate book",
    )
    has_strong_translation_intent = any(
        marker in normalized for marker in strong_translation_markers
    )
    has_task_input_context = bool(extract_absolute_path_candidates(text)) or any(
        marker in normalized
        for marker in (
            "input",
            "output",
            "输入目录",
            "输出目录",
            "输入路径",
            "输出路径",
            "epub",
            ".epub",
            "txt",
            ".txt",
            "目录",
            "路径",
        )
    )
    has_translation_task_context = (
        "翻译" in normalized
        and has_task_input_context
        and "模型配置" not in normalized
    )
    if (
        looks_like_glossary_review_request(text)
        and not has_strong_translation_intent
        and not has_translation_task_context
    ):
        return False
    if any(marker in normalized for marker in ("prompt", "提示词", "预设")) and not any(
        marker in normalized for marker in strong_translation_markers
    ):
        return False
    if (
        looks_like_glossary_extraction_request(text)
        and not has_strong_translation_intent
        and not has_translation_task_context
    ):
        return False
    return has_strong_translation_intent or has_translation_task_context


def looks_like_task_start_request(text: str) -> bool:
    return (
        looks_like_translation_task_request(text)
        or looks_like_glossary_review_request(text)
        or looks_like_glossary_extraction_request(text)
    )


def looks_like_translation_after_review_followup_request(
    text: str,
    current_state: Mapping[str, object],
) -> bool:
    if not latest_completed_glossary_review_task_id(current_state):
        return False
    normalized = text.lower()
    if any(marker in normalized for marker in ("prompt", "提示词", "预设", "模型配置")):
        return False
    if any(
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
            "哪里",
            "哪个页面",
            "我该做什么",
            "status",
            "progress",
            "finished?",
            "done?",
        )
    ):
        return False
    has_translation_or_next_intent = any(
        marker in normalized
        for marker in (
            "翻译",
            "继续",
            "下一步",
            "下一阶段",
            "后续",
            "接着",
            "next",
            "continue",
            "translate",
        )
    )
    if not has_translation_or_next_intent:
        return False
    return any(
        marker in normalized
        for marker in (
            "术语表已确认",
            "术语表已经确认",
            "术语表确认",
            "术语表确认好了",
            "术语表确认无误",
            "术语表没问题",
            "术语确认",
            "术语已经确认",
            "术语确认好了",
            "术语没问题",
            "表没问题",
            "确认好的术语",
            "审查好的术语",
            "审核好的术语",
            "校对好的术语",
            "reviewed glossary",
            "confirmed glossary",
        )
    )


def looks_like_glossary_review_request(text: str) -> bool:
    normalized = text.lower()
    prompt_like = any(marker in normalized for marker in ("prompt", "提示词", "预设"))
    task_like = any(
        marker in normalized
        for marker in (
            "启动",
            "开始",
            "继续",
            "执行",
            "运行",
            "跑",
            "处理",
            "刚才",
            "最新",
            "最近",
            "已提取",
            "提取结果",
            "这个术语表",
            "start",
            "run",
            "continue",
            "task",
            "workflow",
        )
    )
    has_task_id = bool(re.search(r"\bglossary-(?:review-)?[a-z0-9-]+\b", normalized))
    if prompt_like and not (has_task_id or task_like):
        return False
    has_review_marker = (
        any(
            action in normalized
            for action in (
                "审查",
                "审核",
                "校对",
                "复审",
                "审一遍",
                "审一下",
                "审一审",
                "review",
            )
        )
        and "术语" in normalized
    ) or any(
        marker in normalized
        for marker in (
            "术语审查",
            "术语审核",
            "审查术语",
            "审核术语",
            "校对术语",
            "review glossary",
            "glossary review",
        )
    )
    return has_review_marker and (has_task_id or task_like)


def looks_like_glossary_review_followup_request(
    text: str,
    current_state: Mapping[str, object],
) -> bool:
    if not latest_completed_glossary_task_id(current_state):
        return False
    normalized = text.lower()
    if any(marker in normalized for marker in ("prompt", "提示词", "预设", "模型配置")):
        return False
    if any(
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
            "哪里",
            "哪个页面",
            "我该做什么",
            "status",
            "progress",
            "finished?",
            "done?",
        )
    ):
        return False
    if looks_like_translation_task_request(text):
        return False
    has_continue_intent = any(
        marker in normalized
        for marker in (
            "继续",
            "下一步",
            "下一阶段",
            "后续",
            "接着",
            "往后",
            "next",
            "continue",
        )
    )
    if not has_continue_intent:
        return False
    return any(
        marker in normalized
        for marker in (
            "术语",
            "glossary",
            "workflow",
            "流程",
            "任务",
            "刚才",
            "最新",
            "最近",
            "提取完成",
            "提取完",
            "跑完",
            "完成了",
        )
    )


def extract_glossary_task_id(text: str) -> str | None:
    match = re.search(r"\b(glossary-(?!review-)[a-zA-Z0-9-]+)\b", text)
    if match is None:
        return None
    return match.group(1).strip()


def extract_glossary_review_task_id(text: str) -> str | None:
    match = re.search(r"\b(glossary-review-[a-zA-Z0-9-]+)\b", text)
    if match is None:
        return None
    return match.group(1).strip()


def latest_completed_glossary_task_id(
    current_state: Mapping[str, object],
) -> str | None:
    recent_by_kind = current_state.get("recent_task_summaries")
    if not isinstance(recent_by_kind, Mapping):
        return None
    glossary_tasks = recent_by_kind.get("glossary")
    if not isinstance(glossary_tasks, list):
        return None
    for item in glossary_tasks:
        if not isinstance(item, Mapping):
            continue
        task_id = str(item.get("id") or "").strip()
        status = str(item.get("status") or "").strip().lower()
        if task_id and status == "completed" and bool(item.get("artifact_available")):
            return task_id
    return None


def latest_completed_glossary_review_task_id(
    current_state: Mapping[str, object],
) -> str | None:
    recent_by_kind = current_state.get("recent_task_summaries")
    if not isinstance(recent_by_kind, Mapping):
        return None
    review_tasks = recent_by_kind.get("glossary_review")
    if not isinstance(review_tasks, list):
        return None
    for item in review_tasks:
        if not isinstance(item, Mapping):
            continue
        task_id = str(item.get("id") or "").strip()
        status = str(item.get("status") or "").strip().lower()
        if task_id and (status == "completed" or bool(item.get("artifact_available"))):
            return task_id
    return None


def asks_for_reviewed_glossary(text: str) -> bool:
    normalized = text.lower()
    return any(
        marker in normalized
        for marker in (
            "审查好的术语",
            "审核好的术语",
            "校对好的术语",
            "确认好的术语",
            "审查后的术语",
            "审核后的术语",
            "校对后的术语",
            "reviewed glossary",
            "confirmed glossary",
        )
    )


def looks_like_glossary_extraction_request(text: str) -> bool:
    normalized = text.lower()
    if looks_like_glossary_review_request(text):
        return False
    if any(marker in normalized for marker in ("prompt", "提示词", "预设")) and not re.search(
        r"/|input|output|输入|输出|目录|路径",
        text,
        flags=re.IGNORECASE,
    ):
        return False
    if any(
        marker in normalized
        for marker in (
            "提取术语",
            "术语提取",
            "处理术语",
            "处理术语表",
            "傻瓜术语",
            "傻瓜式术语",
            "一键术语",
            "自动术语",
            "术语流程",
            "术语弄一下",
            "术语搞一下",
            "把术语弄一下",
            "把术语搞一下",
            "弄术语",
            "搞术语",
            "整理术语",
            "整理术语表",
            "跑术语",
            "跑术语表",
            "术语表 workflow",
            "术语 workflow",
            "完整 workflow",
            "整套 workflow",
            "完整流程",
            "整套流程",
            "跑流程",
            "处理小说",
            "处理这本小说",
            "数据表提取",
            "处理数据表",
            "关键词和术语表",
            "抽取术语",
            "extract glossary",
            "glossary extraction",
            "extract terms",
            "term extraction",
        )
    ):
        return True
    return looks_like_dumb_initial_workflow_request(text)


def looks_like_dumb_initial_workflow_request(text: str) -> bool:
    normalized = text.lower()
    if any(
        marker in normalized
        for marker in (
            "开始翻译",
            "启动翻译",
            "执行翻译",
            "进行翻译",
            "翻译任务",
            "翻译小说",
            "run translation",
            "start translation",
            "translate novel",
            "translate book",
        )
    ):
        return False
    if any(marker in normalized for marker in ("prompt", "提示词", "预设")):
        return False
    if "模型配置" in normalized or "model profile" in normalized:
        return False
    has_path = bool(extract_absolute_path_candidates(text)) or bool(
        re.search(
            r"input|output|输入|输出|目录|路径|文件夹|epub|\.epub|txt|\.txt",
            text,
            flags=re.IGNORECASE,
        )
    )
    has_novel_context = any(
        marker in text
        for marker in (
            "背景",
            "类型",
            "世界观",
            "作品关键词",
            "人物介绍",
            "作品指南",
            "小说介绍",
        )
    )
    if not (has_path or has_novel_context):
        return False
    if has_path and has_novel_context:
        return True
    has_process_intent = any(
        marker in normalized
        for marker in (
            "处理",
            "跑一下",
            "跑流程",
            "跑这个",
            "跑一遍",
            "弄一下",
            "搞一下",
            "帮我弄",
            "帮我搞",
            "帮我跑",
            "帮我处理",
            "开始处理",
            "一键",
            "傻瓜",
            "自动走",
            "自动跑",
            "workflow",
            "流程",
        )
    )
    has_novel_target = any(
        marker in normalized
        for marker in (
            "小说",
            "这本",
            "这个目录",
            "该目录",
            "这个文件夹",
            "这套文件",
            "这些文件",
            "这份",
            "epub",
            ".epub",
            "txt",
            ".txt",
        )
    )
    return has_process_intent and has_novel_target


def looks_like_glossary_extraction_continuation(text: str) -> bool:
    has_path = bool(extract_absolute_path_candidates(text)) or bool(
        re.search(r"input|output|输入|输出|目录|路径", text, flags=re.IGNORECASE)
    )
    has_context = any(
        marker in text
        for marker in (
            "背景",
            "类型",
            "世界观",
            "作品关键词",
            "人物介绍",
            "作品指南",
            "小说介绍",
        )
    )
    return has_path and has_context


def extract_glossary_task_dirs(text: str) -> tuple[str, str | None, bool]:
    input_dir = extract_labeled_path(
        text,
        ("input", "输入目录", "输入路径", "源目录", "原文目录", "小说目录"),
    )
    output_dir = extract_labeled_path(
        text,
        ("output", "输出目录", "输出路径", "导出目录", "结果目录"),
    )
    candidates = extract_absolute_path_candidates(text)
    if input_dir is None and candidates:
        input_dir = candidates[0]
    if output_dir is None and len(candidates) > 1:
        output_dir = candidates[1]
    if input_dir is None:
        return "", output_dir, False
    if output_dir is None:
        output_dir = infer_sibling_output_dir(text, input_dir)
    return input_dir, output_dir, output_dir is None


def infer_sibling_output_dir(text: str, input_dir: str) -> str | None:
    normalized = text.lower()
    output_markers = (
        "output 里",
        "output里",
        "output 文件夹",
        "output文件夹",
        "output 目录",
        "output目录",
        "输出放 output",
        "输出到 output",
        "结果放 output",
        "结果到 output",
    )
    if not any(marker in normalized for marker in output_markers):
        return None
    path = Path(input_dir).expanduser()
    if path.name.lower() != "input":
        return None
    return str(path.parent / "output")


def looks_like_same_directory_output(text: str) -> bool:
    normalized = text.lower()
    return any(
        marker in normalized
        for marker in (
            "输出和输入放在同一个",
            "输入和输出放在同一个",
            "输出和输入同一个",
            "输入和输出同一个",
            "输出放同一个",
            "输出到同一个",
            "同一个目录",
            "同一个文件夹",
            "同一目录",
            "同一文件夹",
            "same directory",
            "same folder",
        )
    )


def safe_translation_output_dir(input_dir: str) -> str:
    path = Path(input_dir).expanduser()
    name = path.name or "translation"
    return str(path.parent / f"{name}-translated")


def is_output_inside_input(input_dir: str, output_dir: str) -> bool:
    try:
        input_path = Path(input_dir).expanduser().resolve()
        output_path = Path(output_dir).expanduser().resolve()
    except OSError:
        return False
    try:
        output_path.relative_to(input_path)
    except ValueError:
        return False
    return input_path != output_path


def extract_labeled_path(text: str, labels: tuple[str, ...]) -> str | None:
    label_pattern = "|".join(re.escape(label) for label in labels)
    pattern = rf"(?:{label_pattern})\s*(?:folder|dir|目录|路径)?\s*(?:是|为|=|:|：)?\s*(/[^\n，。；;]+)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
        return None
    value = trim_path_like_value(match.group(1))
    return value or None


def extract_absolute_path_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for match in re.finditer(r"(?:^|(?<=[\s:=：]))(/[^\n，。；;]+)", text):
        value = trim_path_like_value(match.group(1))
        if value and value not in candidates:
            candidates.append(value)
    return candidates


def trim_path_like_value(value: str) -> str:
    trimmed = value.strip(" \t\r\n`\"'“”")
    guide_match = re.search(
        r"\s+(?:BL\s*)?作品指南\b|\s+(?:小说背景|作品背景|背景/类型|背景|类型|作品关键词|人物介绍|角色介绍)\b",
        trimmed,
    )
    if guide_match is not None:
        trimmed = trimmed[: guide_match.start()]
    stop_markers = (
        " output",
        " input",
        " 输出",
        " 输入",
        "。output",
        "。input",
        "。输出",
        "。输入",
        "，output",
        "，input",
        "，输出",
        "，输入",
        "；output",
        "；input",
        "；输出",
        "；输入",
        "。小说背景",
        "，小说背景",
        "；小说背景",
        "。背景",
        "，背景",
        "；背景",
        "。类型",
        "，类型",
        "；类型",
        "。作品关键词",
        "，作品关键词",
        "；作品关键词",
        "。作品指南",
        "，作品指南",
        "；作品指南",
        "。BL 作品指南",
        "，BL 作品指南",
        "；BL 作品指南",
        "。人物介绍",
        "，人物介绍",
        "；人物介绍",
        " 小说背景",
        " 背景",
        " 类型",
        " 作品关键词",
        " 作品指南",
        " BL 作品指南",
        " 人物介绍",
        "\n",
    )
    for marker in stop_markers:
        index = trimmed.find(marker)
        if index > 0:
            trimmed = trimmed[:index]
    return trimmed.strip(" \t\r\n，。；;`\"'“”")


def extract_novel_background(text: str) -> str:
    guide_block = extract_novel_background_guide_block(text)
    if guide_block:
        return guide_block
    match = re.search(
        r"(?:小说背景|作品背景|背景(?:/类型)?|世界观|类型)\s*(?:是|为|=|:|：)?\s*(.+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return ""
    value = match.group(1).strip()
    for marker in (
        " input",
        " output",
        " 输入",
        " 输出",
    ):
        index = value.find(marker)
        if index > 0:
            value = value[:index]
    return clean_novel_background_block(value)


def extract_novel_background_guide_block(text: str) -> str:
    marker_match = re.search(r"(?:BL\s*)?作品指南", text)
    if marker_match is not None:
        return clean_novel_background_block(text[marker_match.start() :])
    background_match = re.search(
        r"(?:小说背景|作品背景|背景(?:/类型)?|世界观|类型)\s*(?:是|为|=|:|：)?",
        text,
        flags=re.IGNORECASE,
    )
    if background_match is None:
        return ""
    block = text[background_match.start() :]
    if not any(marker in block for marker in ("作品关键词", "人物介绍", "角色介绍")):
        return ""
    return clean_novel_background_block(block)


def clean_novel_background_block(value: str) -> str:
    lines: list[str] = []
    previous_blank = False
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line:
            if lines and not previous_blank:
                lines.append("")
                previous_blank = True
            continue
        if looks_like_background_control_line(line):
            continue
        lines.append(line)
        previous_blank = False
    cleaned = "\n".join(lines).strip(" \t\r\n，。；;")
    return cleaned[:_MAX_NOVEL_BACKGROUND_LENGTH].strip(" \t\r\n，。；;")


def looks_like_background_control_line(line: str) -> bool:
    if re.match(r"^(?:input|output|输入|输出).*(?:/|目录|路径)", line, re.IGNORECASE):
        return True
    if re.match(r"^/[^，。；;]+$", line):
        return True
    return any(
        marker in line
        for marker in (
            "输出和输入放在同一个",
            "输入和输出放在同一个",
            "输出目录默认",
            "默认使用 input",
        )
    )


def extract_source_language(text: str) -> str:
    normalized = text.lower()
    if any(marker in normalized for marker in ("韩译中", "韩文", "韩语", "korean", " kr", " ko")):
        return "kr"
    if any(marker in normalized for marker in ("日译中", "日文", "日语", "japanese", " ja", " jp")):
        return "ja"
    return ""


def extract_target_language(text: str) -> str:
    normalized = text.lower()
    if any(marker in normalized for marker in ("繁中", "繁体", "traditional chinese", "zh-hant")):
        return "zh-Hant"
    if any(marker in normalized for marker in ("译中", "中文", "简中", "简体", "chinese", " zh")):
        return "zh"
    return ""


__all__ = [
    "asks_for_reviewed_glossary",
    "extract_absolute_path_candidates",
    "extract_glossary_review_task_id",
    "extract_glossary_task_dirs",
    "extract_glossary_task_id",
    "extract_labeled_path",
    "extract_novel_background",
    "extract_source_language",
    "extract_target_language",
    "infer_sibling_output_dir",
    "is_output_inside_input",
    "latest_completed_glossary_review_task_id",
    "latest_completed_glossary_task_id",
    "looks_like_background_control_line",
    "looks_like_dumb_initial_workflow_request",
    "looks_like_glossary_extraction_continuation",
    "looks_like_glossary_extraction_request",
    "looks_like_glossary_review_followup_request",
    "looks_like_glossary_review_request",
    "looks_like_same_directory_output",
    "looks_like_task_start_request",
    "looks_like_translation_after_review_followup_request",
    "looks_like_translation_task_request",
    "safe_translation_output_dir",
    "trim_path_like_value",
]
