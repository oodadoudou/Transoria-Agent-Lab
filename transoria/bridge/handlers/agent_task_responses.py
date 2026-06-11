"""Direct task-start response builders for Agent Lab chat."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from transoria.agent.completeness import assess_start_draft
from transoria.agent.schemas import AgentActionDraft, AgentWorkspaceState
from transoria.bridge.handlers.agent_model_intents import (
    looks_like_model_profile_copy_request,
    requested_existing_model_for_task,
)
from transoria.bridge.handlers.agent_task_drafts import (
    build_start_task_draft,
    build_start_with_default_stage_config,
)
from transoria.bridge.handlers.agent_task_intents import (
    asks_for_reviewed_glossary,
    extract_absolute_path_candidates,
    extract_glossary_review_task_id,
    extract_glossary_task_dirs,
    extract_glossary_task_id,
    extract_novel_background,
    extract_source_language,
    extract_target_language,
    is_output_inside_input,
    latest_completed_glossary_review_task_id,
    latest_completed_glossary_task_id,
    looks_like_glossary_extraction_continuation,
    looks_like_glossary_extraction_request,
    looks_like_glossary_review_followup_request,
    looks_like_glossary_review_request,
    looks_like_same_directory_output,
    looks_like_translation_after_review_followup_request,
    looks_like_translation_task_request,
    safe_translation_output_dir,
)
from transoria.model_profiles import ModelProfileStore


def direct_glossary_extraction_response(
    *,
    user_message: str,
    conversation_context: list[Mapping[str, object]],
    state: AgentWorkspaceState,
    current_state: Mapping[str, object],
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> tuple[str, AgentActionDraft | None] | None:
    if looks_like_glossary_review_request(user_message):
        return None
    context_text = "\n".join(
        str(item.get("content") or "") for item in conversation_context
    )
    current_request = looks_like_glossary_extraction_request(user_message)
    implicit_current_request = looks_like_glossary_extraction_continuation(
        user_message
    ) and not looks_like_model_profile_copy_request(user_message)
    contextual_continuation = (
        looks_like_glossary_extraction_request(context_text)
        and looks_like_glossary_extraction_continuation(user_message)
    )
    if (
        not current_request
        and not contextual_continuation
        and not implicit_current_request
    ):
        return None
    combined_text = "\n".join((context_text, user_message))
    extraction_text = combined_text if contextual_continuation else user_message
    input_dir, output_dir, output_defaulted = extract_glossary_task_dirs(user_message)
    if not input_dir and contextual_continuation:
        input_dir, output_dir, output_defaulted = extract_glossary_task_dirs(
            extraction_text
        )
    if not input_dir:
        return (
            "我理解你想提取术语。请提供 input 目录；如果不提供 output 目录，我会默认使用 input 目录作为输出目录。",
            None,
        )
    novel_background = extract_novel_background(user_message)
    if not novel_background and contextual_continuation:
        novel_background = _latest_user_novel_background(
            conversation_context,
            current_message=user_message,
        )
    if not novel_background:
        return (
            "我理解你想提取术语。请再提供小说背景，这会进入本次任务配置，不会写入手动设置页。",
            None,
        )
    source_language = extract_source_language(user_message) or _settings_default_value(
        current_state,
        "glossary",
        "source_language",
    )
    target_language = extract_target_language(user_message) or _settings_default_value(
        current_state,
        "glossary",
        "target_language",
    )
    payload = {
        "input_dir": input_dir,
        "output_dir": output_dir or input_dir,
        "source_language": source_language,
        "target_language": target_language,
        "novel_background": novel_background,
    }
    completeness = assess_start_draft(
        draft_kind="start_glossary_task",
        state=state,
        payload=payload,
    )
    requested_stage_model = requested_existing_model_for_task(
        user_message,
        profile_store=profile_store,
    )
    if not completeness.complete:
        dumb_start = build_start_with_default_stage_config(
            draft_kind="start_glossary_task",
            payload=payload,
            state=state,
            completeness_missing=completeness.missing,
            cache_root=cache_root,
            title="补齐术语提取配置并启动任务",
            start_title="启动术语提取",
            start_summary="使用补齐后的术语提取模型和 Prompt，从指定目录提取术语表。",
            reply=(
                "我可以按傻瓜模式处理：先用当前工作模型补齐术语提取模型，"
                "用内置默认术语提取 Prompt 补齐缺失项，然后再启动术语提取任务。"
                "下面是一个组合草案，只有点击「应用」后才会写入工作区配置并启动任务。"
            ),
            requested_stage_model=requested_stage_model,
        )
        if dumb_start is not None:
            return dumb_start
        missing = "、".join(completeness.missing)
        return (
            f"我理解你想提取术语，但当前术语提取任务配置还不完整：{missing}。请先补齐对应模型、Prompt 或语言设置。",
            None,
        )
    note = "未提供 output 目录，草案会默认输出到 input 目录。" if output_defaulted else ""
    draft = build_start_task_draft(
        draft_kind="start_glossary_task",
        title="启动术语提取",
        summary="使用当前术语提取模型和 Prompt，从指定目录提取术语表。",
        payload=payload,
    )
    return (
        (
            "我已准备好术语提取任务草案。"
            f"{note}请确认后再启动；本次目录和背景只作为任务覆盖项，不会写入手动设置。"
        ),
        draft,
    )


def direct_glossary_review_response(
    *,
    user_message: str,
    conversation_context: list[Mapping[str, object]],
    state: AgentWorkspaceState,
    current_state: Mapping[str, object],
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> tuple[str, AgentActionDraft | None] | None:
    if not (
        looks_like_glossary_review_request(user_message)
        or looks_like_glossary_review_followup_request(user_message, current_state)
    ):
        return None
    context_text = _conversation_context_text(conversation_context)
    task_id = (
        extract_glossary_task_id(user_message)
        or extract_glossary_task_id(context_text)
        or latest_completed_glossary_task_id(current_state)
    )
    if not task_id:
        return (
            "我理解你想启动术语审查。请提供已完成的 glossary task ID，或先完成一次术语提取任务。",
            None,
        )
    novel_background = extract_novel_background(
        user_message
    ) or _latest_user_novel_background(
        conversation_context,
        current_message=user_message,
    )
    payload: dict[str, object] = {
        "glossary_task_id": task_id,
    }
    if novel_background:
        payload["novel_background"] = novel_background
    completeness = assess_start_draft(
        draft_kind="start_glossary_review_task",
        state=state,
        payload=payload,
    )
    requested_stage_model = requested_existing_model_for_task(
        user_message,
        profile_store=profile_store,
    )
    if not completeness.complete:
        dumb_start = build_start_with_default_stage_config(
            draft_kind="start_glossary_review_task",
            payload=payload,
            state=state,
            completeness_missing=completeness.missing,
            cache_root=cache_root,
            title="补齐术语审查配置并启动任务",
            start_title="启动术语审查",
            start_summary=f"使用术语提取任务 {task_id} 的术语表和参考文本启动术语审查。",
            reply=(
                "我可以先用当前工作模型补齐术语审查模型，"
                "用内置默认术语审查 Prompt 补齐缺失项，然后再启动术语审查任务。"
                "下面是一个组合草案，只有点击「应用」后才会写入工作区配置并启动任务。"
            ),
            requested_stage_model=requested_stage_model,
        )
        if dumb_start is not None:
            return dumb_start
        missing = "、".join(completeness.missing)
        return (
            f"我理解你想审查术语，但当前术语审查任务配置还不完整：{missing}。请先补齐对应模型、Prompt 或语言设置。",
            None,
        )
    draft = build_start_task_draft(
        draft_kind="start_glossary_review_task",
        title="启动术语审查",
        summary=f"使用术语提取任务 {task_id} 的术语表和参考文本启动术语审查。",
        payload=payload,
    )
    return (
        (
            f"我已准备好术语审查任务草案，会读取术语提取任务 {task_id} 的输出产物。"
            "请确认后再启动；本次背景只作为任务覆盖项，不会写入手动设置。"
        ),
        draft,
    )


def direct_translation_response(
    *,
    user_message: str,
    conversation_context: list[Mapping[str, object]],
    state: AgentWorkspaceState,
    current_state: Mapping[str, object],
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> tuple[str, AgentActionDraft | None] | None:
    followup_request = looks_like_translation_after_review_followup_request(
        user_message,
        current_state,
    )
    if not looks_like_translation_task_request(user_message) and not followup_request:
        return None
    context_text = "\n".join(
        str(item.get("content") or "") for item in conversation_context
    )
    combined_text = "\n".join((context_text, user_message))
    input_dir, output_dir, _ = extract_glossary_task_dirs(user_message)
    if not input_dir:
        input_dir, output_dir, _ = extract_glossary_task_dirs(combined_text)
    same_dir_requested = looks_like_same_directory_output(
        user_message
    ) or looks_like_same_directory_output(combined_text)
    if not input_dir:
        if not extract_absolute_path_candidates(combined_text):
            if followup_request:
                return (
                    "我理解你已经确认术语表，下一步是启动翻译。请提供 input 目录和一个独立的 output 目录；翻译输出不能默认写回输入目录。",
                    None,
                )
            return None
        return (
            "我理解你想启动翻译。请提供 input 目录和一个独立的 output 目录；翻译输出不能默认写回输入目录。",
            None,
        )
    output_adjustment_note = ""
    if not output_dir and same_dir_requested:
        output_dir = safe_translation_output_dir(input_dir)
        output_adjustment_note = (
            f"你提到同目录输出；但翻译不能写回 input，我已在草案中改为安全同级输出目录：{output_dir}。"
        )
    if not output_dir:
        return (
            "我理解你想启动翻译，但还缺 output 目录。翻译输出需要使用独立目录，避免覆盖源文件。",
            None,
        )
    if input_dir.rstrip("/") == output_dir.rstrip("/"):
        if same_dir_requested:
            output_dir = safe_translation_output_dir(input_dir)
            output_adjustment_note = (
                f"你提到同目录输出；但翻译不能写回 input，我已在草案中改为安全同级输出目录：{output_dir}。"
            )
        else:
            return (
                "我理解你想启动翻译，但 input 和 output 目录不能相同。请提供一个独立的 output 目录。",
                None,
            )
    if is_output_inside_input(input_dir, output_dir):
        return (
            "我理解你想启动翻译，但 output 目录不能放在 input 目录里面，否则译后文件会被下一次扫描当成源文。请提供一个独立的 output 目录。",
            None,
        )
    source_language = extract_source_language(user_message) or _settings_default_value(
        current_state,
        "translation",
        "source_language",
    )
    target_language = extract_target_language(user_message) or _settings_default_value(
        current_state,
        "translation",
        "target_language",
    )
    payload: dict[str, object] = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "source_language": source_language,
        "target_language": target_language,
    }
    glossary_review_task_id = (
        extract_glossary_review_task_id(user_message)
        or extract_glossary_review_task_id(context_text)
        or (
            latest_completed_glossary_review_task_id(current_state)
            if (asks_for_reviewed_glossary(combined_text) or followup_request)
            else None
        )
    )
    glossary_task_id = None
    if glossary_review_task_id:
        payload["glossary_review_task_id"] = glossary_review_task_id
    else:
        glossary_task_id = extract_glossary_task_id(
            user_message
        ) or extract_glossary_task_id(context_text)
        if glossary_task_id:
            payload["glossary_task_id"] = glossary_task_id
    completeness = assess_start_draft(
        draft_kind="start_translation_task",
        state=state,
        payload=payload,
    )
    requested_stage_model = requested_existing_model_for_task(
        user_message,
        profile_store=profile_store,
    )
    if not completeness.complete:
        dumb_start = build_start_with_default_stage_config(
            draft_kind="start_translation_task",
            payload=payload,
            state=state,
            completeness_missing=completeness.missing,
            cache_root=cache_root,
            title="补齐翻译配置并启动任务",
            start_title="启动翻译",
            start_summary="使用补齐后的翻译模型和 Prompt 翻译指定目录。",
            reply=(
                f"{output_adjustment_note}"
                "我可以按当前工作区的傻瓜模式处理：先用当前工作模型补齐翻译模型，"
                "用内置默认翻译 Prompt 补齐缺失项，然后再启动翻译任务。"
                "下面是一个组合草案，只有点击「应用」后才会写入工作区配置并启动任务。"
            ),
            requested_stage_model=requested_stage_model,
        )
        if dumb_start is not None:
            return dumb_start
        missing = "、".join(completeness.missing)
        return (
            f"我理解你想启动翻译，但当前翻译任务配置还不完整：{missing}。请先补齐对应模型、Prompt 或语言设置。",
            None,
        )
    glossary_note = (
        f"并引用术语审查任务 {glossary_review_task_id} 的确认术语表。"
        if glossary_review_task_id
        else f"并引用术语提取任务 {glossary_task_id} 的术语表。"
        if glossary_task_id
        else "不引用历史术语任务，使用当前翻译设置中的术语表。"
    )
    draft = build_start_task_draft(
        draft_kind="start_translation_task",
        title="启动翻译",
        summary=f"使用当前翻译模型和 Prompt 翻译指定目录，{glossary_note}",
        payload=payload,
    )
    return (
        (
            "我已准备好翻译任务草案。"
            f"{output_adjustment_note}"
            f"{glossary_note}请确认后再启动；本次目录和语言只作为任务覆盖项，不会写入手动设置。"
        ),
        draft,
    )


def _conversation_context_text(
    conversation_context: list[Mapping[str, object]],
    *,
    role: str | None = None,
) -> str:
    parts: list[str] = []
    for item in conversation_context:
        if role is not None and item.get("role") != role:
            continue
        content = str(item.get("content") or "").strip()
        if content:
            parts.append(content)
    return "\n".join(parts)


def _latest_user_novel_background(
    conversation_context: list[Mapping[str, object]],
    *,
    current_message: str,
) -> str:
    current = current_message.strip()
    for item in reversed(conversation_context):
        if item.get("role") != "user":
            continue
        content = str(item.get("content") or "").strip()
        if not content or content == current:
            continue
        background = extract_novel_background(content)
        if background:
            return background
    return ""


def _settings_default_value(
    current_state: Mapping[str, object],
    section: str,
    key: str,
) -> str:
    defaults = current_state.get("settings_defaults")
    if not isinstance(defaults, Mapping):
        return ""
    section_defaults = defaults.get(section)
    if not isinstance(section_defaults, Mapping):
        return ""
    value = section_defaults.get(key)
    return value.strip() if isinstance(value, str) else ""
