"""Prompt-intent direct responses for Agent Lab bridge handlers."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path

from transoria.agent.schemas import AgentActionDraft, AgentWorkspaceState
from transoria.bridge.errors import BridgeError
from transoria.bridge.handlers.agent_draft_salvage import (
    build_prompt_body_from_request,
)
from transoria.bridge.handlers.agent_inventory import prompt_store_for
from transoria.bridge.handlers.agent_prompt_actions import coerce_prompt_kind
from transoria.bridge.handlers.agent_recipe_actions import (
    MAX_RECIPE_NAME_LENGTH,
    PROMPT_KIND_BY_SLOT,
)
from transoria.prompts import (
    DEFAULT_GLOSSARY_PRESET_ID,
    DEFAULT_GLOSSARY_REVIEW_PRESET_ID,
    DEFAULT_TRANSLATION_PRESET_ID,
    PromptKind,
    PromptPreset,
)

ExcludedRequestPredicate = Callable[[str], bool]

DEFAULT_PROMPT_ID_BY_SLOT = {
    "translation": DEFAULT_TRANSLATION_PRESET_ID,
    "term_extract": DEFAULT_GLOSSARY_PRESET_ID,
    "term_review": DEFAULT_GLOSSARY_REVIEW_PRESET_ID,
}


def direct_prompt_preset_response(
    *,
    user_message: str,
    excluded_request: ExcludedRequestPredicate | None = None,
) -> tuple[str, AgentActionDraft | None] | None:
    if not looks_like_direct_prompt_preset_request(
        user_message,
        excluded_request=excluded_request,
    ):
        return None
    name = extract_prompt_create_name(user_message)
    if not name:
        return None
    if not has_prompt_creation_requirements(user_message):
        return None
    try:
        prompt_kind = coerce_prompt_kind(None, fallback_text=user_message)
    except BridgeError:
        return (
            "我理解你想创建 Prompt 预设，但还不清楚它属于哪个阶段。请说明是翻译、术语提取还是术语审查 Prompt。",
            None,
        )
    system_prompt = extract_prompt_body_from_request(user_message)
    if not system_prompt:
        system_prompt = build_prompt_body_from_request(
            prompt_kind=prompt_kind,
            user_request=user_message,
        )
    title = f"创建{name}"
    payload = {
        "kind": prompt_kind.value,
        "name": name,
        "description": f"由 Agent Lab 根据聊天请求创建的{name}。",
        "system_prompt": system_prompt,
        "enabled": True,
    }
    draft = AgentActionDraft.create(
        kind="create_prompt_preset",
        title=title,
        summary=f"创建一套 {prompt_kind_label(prompt_kind)} Prompt 预设：{name}。",
        payload=payload,
    )
    return (
        (
            f"我已准备好创建 Prompt 预设「{name}」的草案。"
            "请检查内容，确认后才会写入 Prompt 配置。"
        ),
        draft,
    )


def direct_prompt_quality_response(
    *,
    user_message: str,
    state: AgentWorkspaceState,
    cache_root: Path,
) -> tuple[str, AgentActionDraft | None] | None:
    if not looks_like_prompt_quality_request(user_message):
        return None
    prompt_kind = quality_prompt_kind(user_message)
    issues = quality_issue_markers(user_message)
    if not issues:
        return (
            (
                "我可以帮你优化 Prompt，但还缺少可操作的问题描述。"
                "请贴一小段原文/译文，或说明具体问题类型，例如人名、术语、漏翻、源文残留、"
                "文风太直译或翻译腔。"
            ),
            None,
        )

    source = selected_prompt_for_kind(
        prompt_kind,
        state=state,
        cache_root=cache_root,
    )
    addendum = quality_prompt_addendum(
        issues=issues,
        user_message=user_message,
    )
    label = prompt_kind_label(prompt_kind)
    if source is not None and not source.is_system:
        system_prompt = f"{source.system_prompt.rstrip()}\n\n{addendum}"
        draft = AgentActionDraft.create(
            kind="update_prompt_preset",
            title=f"优化{source.name}",
            summary=f"在现有 {label} Prompt「{source.name}」中追加本次质量修正规则。",
            payload={
                "id": source.id,
                "patch": {
                    "system_prompt": system_prompt,
                    "description": source.description
                    or f"由 Agent Lab 根据质量反馈优化的{label} Prompt。",
                },
            },
        )
        return (
            (
                f"我会在当前 {label} Prompt「{source.name}」里追加质量修正规则，"
                "解决你指出的问题。请先检查草案，确认后才会写入配置。"
            ),
            draft,
        )

    base_prompt = (
        source.system_prompt
        if source is not None
        else build_prompt_body_from_request(
            prompt_kind=prompt_kind,
            user_request=user_message,
        )
    )
    name = quality_prompt_name(user_message, prompt_kind, source)
    draft = AgentActionDraft.create(
        kind="create_prompt_preset",
        title=f"创建{name}",
        summary=f"基于当前 {label} Prompt 创建可编辑副本，并加入本次质量修正规则。",
        payload={
            "kind": prompt_kind.value,
            "name": name,
            "description": f"由 Agent Lab 根据质量反馈创建的{label} Prompt。",
            "system_prompt": f"{base_prompt.rstrip()}\n\n{addendum}",
            "enabled": True,
        },
    )
    return (
        (
            f"当前选中的 {label} Prompt 是系统预设或尚未选择可编辑预设，"
            f"我会创建一套新的「{name}」供你确认保存。"
        ),
        draft,
    )


def has_prompt_creation_requirements(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "要求",
            "规则",
            "内容",
            "正文",
            "强调",
            "用于",
            "适合",
            "风格",
            "规范",
            "保留",
            "禁止",
        )
    )


def looks_like_direct_prompt_preset_request(
    text: str,
    *,
    excluded_request: ExcludedRequestPredicate | None = None,
) -> bool:
    if excluded_request is not None and excluded_request(text):
        return False
    normalized = text.lower()
    if "prompt" not in normalized and "提示词" not in text:
        return False
    return any(marker in text for marker in ("创建", "新建", "新增", "保存", "配置", "做一套", "准备一套"))


def extract_prompt_create_name(text: str) -> str:
    patterns = (
        r"(?:名字|名称|命名为|名为|叫做|叫)\s*[「『“\"'](.+?)[」』”\"']",
        r"(?:名字|名称|命名为|名为|叫做|叫)\s*[:：]?\s*([^\n，。,.]+)",
        r"(?:创建|新建|新增|配置|保存|准备)(?:一套|一个|新的)?\s*[「『“\"'](.+?)[」』”\"']\s*(?:的)?(?:Prompt|prompt|提示词)",
        r"(?:创建|新建|新增|配置|保存|准备)(?:一套|一个|新的)?\s*(.+?)(?:Prompt|prompt|提示词)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        value = match.group(1).strip(" \t\r\n「」『』“”\"'：:")
        value = re.sub(r"^(?:翻译|术语提取|术语审核|术语审查)\s*", "", value).strip()
        if value:
            return value[:MAX_RECIPE_NAME_LENGTH]
    return ""


def extract_prompt_body_from_request(text: str) -> str:
    marker_pattern = (
        r"(?:内容|正文|规则|要求|prompt|Prompt|提示词)\s*(?:如下|是|为)?\s*[:：]\s*(.+)"
    )
    match = re.search(marker_pattern, text, flags=re.DOTALL)
    if match:
        value = match.group(1).strip()
        if value:
            return value
    return ""


def prompt_kind_label(kind: PromptKind) -> str:
    return {
        PromptKind.TRANSLATION: "翻译",
        PromptKind.GLOSSARY: "术语提取",
        PromptKind.GLOSSARY_REVIEW: "术语审查",
    }[kind]


def looks_like_prompt_quality_request(text: str) -> bool:
    normalized = text.lower()
    mentions_prompt = "prompt" in normalized or "提示词" in text
    mentions_translation_quality = "翻译" in text and any(
        marker in text
        for marker in (
            "效果不好",
            "质量不好",
            "不好",
            "问题",
            "翻译腔",
            "不自然",
            "源文残留",
            "原文残留",
            "漏翻",
            "错译",
            "直译",
            "文风",
            "人名",
            "术语",
            "一致",
            "低置信",
            "质量",
            "很怪",
            "怪",
        )
    )
    if not mentions_prompt and not mentions_translation_quality:
        return False
    if not any(
        marker in text
        for marker in (
            "改",
            "修改",
            "优化",
            "调整",
            "重写",
            "设计",
            "重新设计",
            "看看",
            "看一下",
            "帮我看",
            "诊断",
        )
    ):
        return False
    return any(
        marker in text
        for marker in (
            "效果不好",
            "质量不好",
            "不好",
            "问题",
            "翻译腔",
            "不自然",
            "源文残留",
            "原文残留",
            "漏翻",
            "错译",
            "直译",
            "文风",
            "人名",
            "术语",
            "一致",
            "低置信",
            "质量",
            "很怪",
            "怪",
        )
    )


def quality_prompt_kind(text: str) -> PromptKind:
    try:
        return coerce_prompt_kind(None, fallback_text=text)
    except BridgeError:
        return PromptKind.TRANSLATION


def quality_issue_markers(text: str) -> list[str]:
    candidates = (
        ("源文残留", ("源文残留", "原文残留", "韩文残留", "日文残留", "英文残留")),
        ("人名不一致", ("人名不一致", "名字不一致", "称呼不一致", "人名")),
        ("术语不一致", ("术语不一致", "术语漂移", "术语")),
        ("文风不自然", ("文风不自然", "不自然", "翻译腔", "太直译", "直译", "文风")),
        ("漏翻", ("漏翻", "缺句", "少翻")),
        ("错译", ("错译", "误译", "理解错")),
        ("低置信度", ("低置信", "不确定")),
        ("长度或分段异常", ("长度", "分段", "行数")),
    )
    issues: list[str] = []
    for label, markers in candidates:
        if any(marker in text for marker in markers):
            issues.append(label)
    if issues == ["术语不一致"] and "术语" in text and "翻译" not in text:
        return []
    return issues


def selected_prompt_for_kind(
    kind: PromptKind,
    *,
    state: AgentWorkspaceState,
    cache_root: Path,
) -> PromptPreset | None:
    slot = prompt_slot_for_kind(kind)
    prompt_id = state.stage_prompt_ids.get(slot) or DEFAULT_PROMPT_ID_BY_SLOT.get(slot)
    if not prompt_id:
        return None
    for preset in prompt_store_for(cache_root, kind).load():
        if preset.id == prompt_id:
            return preset
    return None


def prompt_slot_for_kind(kind: PromptKind) -> str:
    for slot, prompt_kind in PROMPT_KIND_BY_SLOT.items():
        if prompt_kind == kind:
            return slot
    raise BridgeError.invalid_argument(
        f"unsupported prompt kind: {kind.value}",
        field="kind",
    )


def quality_prompt_addendum(*, issues: Sequence[str], user_message: str) -> str:
    issue_text = "、".join(dict.fromkeys(issues))
    return "\n".join(
        (
            "本次质量优化要求：",
            f"- 重点修正：{issue_text}。",
            "- 处理前先识别上下文、人名、称呼、术语和叙事视角，避免只逐句直译。",
            "- 输出时优先保持原文信息完整和分段稳定，不新增原文不存在的情节、心理解释或评价。",
            "- 对人名、称呼、组织、作品内专有名词保持一致；不确定时保留可校对的稳定译名。",
            "- 主动规避源语言残留、漏翻、重复漂移和明显翻译腔。",
            f"- 用户反馈原文：{user_message.strip()}",
        )
    )


def quality_prompt_name(
    text: str,
    kind: PromptKind,
    source: PromptPreset | None,
) -> str:
    explicit = extract_prompt_create_name(text)
    if explicit:
        return explicit
    base = source.name if source is not None else f"{prompt_kind_label(kind)} Prompt"
    name = f"{base} - 质量优化"
    return name[:MAX_RECIPE_NAME_LENGTH]
