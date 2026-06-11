"""Draft salvage helpers for Agent Lab configuration proposals."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

from transoria.agent.schemas import AgentActionDraft
from transoria.bridge.errors import BridgeError
from transoria.bridge.handlers.agent_inventory import prompt_store_for
from transoria.bridge.handlers.agent_model_intents import (
    normalize_lookup_label,
    resolve_model_profile_by_label,
)
from transoria.bridge.handlers.agent_prompt_actions import (
    coerce_prompt_kind,
    prompt_body_from_payload,
)
from transoria.bridge.handlers.agent_recipe_actions import (
    MAX_RECIPE_NAME_LENGTH,
    recipe_body_from_payload,
)
from transoria.model_profiles import ModelProfileStore
from transoria.prompts import PromptKind, PromptPreset

MAX_TITLE_LENGTH = 120


def salvage_invalid_agent_draft(
    *,
    user_message: str,
    reply: str,
    draft: AgentActionDraft,
    current_state: Mapping[str, object],
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> AgentActionDraft | None:
    text = "\n".join((user_message, reply, draft.title, draft.summary))
    if draft.kind == "create_prompt_preset":
        return salvage_create_prompt_draft(draft, text=text)
    if draft.kind == "create_recipe":
        return salvage_create_recipe_draft(
            draft,
            text=text,
            current_state=current_state,
        )
    if draft.kind == "compound_config_update":
        return salvage_compound_draft(
            draft,
            text=text,
            current_state=current_state,
            profile_store=profile_store,
            cache_root=cache_root,
        )
    return None


def salvage_create_prompt_draft(
    draft: AgentActionDraft,
    *,
    text: str,
) -> AgentActionDraft | None:
    try:
        prompt_kind = coerce_prompt_kind(None, fallback_text=text)
    except BridgeError:
        return None
    name = extract_quoted_name(text) or extract_named_value(text)
    if not name:
        return None
    system_prompt = prompt_body_from_payload(draft.payload)
    if not system_prompt:
        system_prompt = build_prompt_body_from_request(
            prompt_kind=prompt_kind,
            user_request=text,
        )
    payload = dict(draft.payload)
    payload.update(
        {
            "kind": prompt_kind.value,
            "name": name,
            "description": str(payload.get("description") or draft.summary).strip(),
            "system_prompt": system_prompt,
            "enabled": bool(payload.get("enabled", True)),
        }
    )
    return AgentActionDraft.create(
        kind=draft.kind,
        title=draft.title,
        summary=draft.summary,
        payload=payload,
    )


def salvage_create_recipe_draft(
    draft: AgentActionDraft,
    *,
    text: str,
    current_state: Mapping[str, object],
) -> AgentActionDraft | None:
    name = extract_quoted_name(text) or extract_named_value(text)
    if not name:
        return None
    stage_models = current_state.get("stage_model_ids")
    stage_prompts = current_state.get("stage_prompt_ids")
    payload = recipe_body_from_payload(draft.payload)
    payload.update(
        {
            "name": name,
            "description": str(payload.get("description") or draft.summary).strip(),
            "stage_model_ids": dict(stage_models)
            if isinstance(stage_models, Mapping)
            else {},
            "stage_prompt_ids": dict(stage_prompts)
            if isinstance(stage_prompts, Mapping)
            else {},
        }
    )
    return AgentActionDraft.create(
        kind=draft.kind,
        title=draft.title,
        summary=draft.summary,
        payload=payload,
    )


def salvage_compound_draft(
    draft: AgentActionDraft,
    *,
    text: str,
    current_state: Mapping[str, object],
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> AgentActionDraft | None:
    actions: list[dict[str, object]] = []
    concurrency = extract_model_concurrency_update(text, profile_store=profile_store)
    if concurrency is not None:
        profile_id, limit = concurrency
        actions.append(
            {
                "kind": "update_model_profile",
                "title": "更新模型并发数",
                "summary": f"将模型 {profile_id} 的并发数改为 {limit}。",
                "payload": {
                    "profile_id": profile_id,
                    "patch": {"concurrency_limit": limit},
                },
            }
        )
    prompt_rename = extract_prompt_rename(text, cache_root=cache_root)
    if prompt_rename is not None:
        preset, old_name, new_name = prompt_rename
        if preset.is_system:
            actions.append(
                {
                    "kind": "create_prompt_preset",
                    "title": "创建 Prompt 自定义副本",
                    "summary": (
                        f"内置 Prompt {old_name} 为只读；创建同内容的自定义副本 "
                        f"{new_name}。"
                    ),
                    "payload": {
                        "kind": preset.kind.value,
                        "name": new_name,
                        "description": f"从内置 Prompt {old_name} 复制创建。",
                        "system_prompt": preset.system_prompt,
                        "enabled": preset.enabled,
                    },
                }
            )
        else:
            actions.append(
                {
                    "kind": "update_prompt_preset",
                    "title": "重命名 Prompt",
                    "summary": f"将 Prompt {old_name} 重命名为 {new_name}。",
                    "payload": {
                        "id": preset.id,
                        "patch": {"name": new_name},
                    },
                }
            )
    recipe_name = extract_recipe_save_name(text)
    if recipe_name:
        stage_models = current_state.get("stage_model_ids")
        stage_prompts = current_state.get("stage_prompt_ids")
        actions.append(
            {
                "kind": "create_recipe",
                "title": "保存当前阶段配置",
                "summary": f"保存当前阶段配置为 {recipe_name}。",
                "payload": {
                    "name": recipe_name,
                    "description": "由 Agent Lab 根据当前工作区阶段配置创建。",
                    "stage_model_ids": dict(stage_models)
                    if isinstance(stage_models, Mapping)
                    else {},
                    "stage_prompt_ids": dict(stage_prompts)
                    if isinstance(stage_prompts, Mapping)
                    else {},
                },
            }
        )
    if not actions:
        return None
    return AgentActionDraft.create(
        kind=draft.kind,
        title=draft.title,
        summary=draft.summary,
        payload={"actions": actions},
    )


def build_prompt_body_from_request(
    *,
    prompt_kind: PromptKind,
    user_request: str,
) -> str:
    stage = {
        PromptKind.TRANSLATION: "翻译",
        PromptKind.GLOSSARY: "术语提取",
        PromptKind.GLOSSARY_REVIEW: "术语审核",
    }[prompt_kind]
    return "\n".join(
        (
            f"你是 Transoria {stage}流程中的专业模型。",
            "请严格遵守用户对这套 Prompt 的要求：",
            user_request.strip(),
            "保持输出清晰、稳定，并优先服务于小说翻译质量。",
        )
    )


def extract_quoted_name(text: str) -> str:
    for pattern in (r"[『「“](.+?)[』」”]", r"['\"](.+?)['\"]"):
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            if value:
                return value[:MAX_RECIPE_NAME_LENGTH]
    return ""


def extract_named_value(text: str) -> str:
    match = re.search(r"(?:名字|名称|叫|名为)\s*[:：]?\s*([^\s，。,.]+)", text)
    if not match:
        return ""
    return match.group(1).strip("『』「」“”\"' ")[:MAX_RECIPE_NAME_LENGTH]


def extract_model_concurrency_update(
    text: str,
    *,
    profile_store: ModelProfileStore,
) -> tuple[str, int] | None:
    patterns = (
        r"(?:模型|model)\s*[「『“\"']?(.+?)[」』”\"']?\s*的?并发(?:数)?(?:改成|修改为|设置为|设为|=)\s*(\d+)",
        r"[「『“\"']?([A-Za-z0-9_.\- ]+)[」』”\"']?\s*的?并发(?:数)?(?:改成|修改为|设置为|设为|=)\s*(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        label = match.group(1).strip(" \t\r\n「」『』“”\"'")
        profile = resolve_model_profile_by_label(label, profile_store)
        if profile is None:
            continue
        limit = int(match.group(2))
        if limit < 0:
            continue
        return profile.id, limit
    return None


def extract_prompt_rename(
    text: str,
    *,
    cache_root: Path,
) -> tuple[PromptPreset, str, str] | None:
    match = re.search(
        r"(?:(翻译|术语提取|术语审核|术语审查)\s*)?Prompt\s*[「『“\"](.+?)[」』”\"]\s*(?:重命名为|改名为|改成|修改为)\s*[「『“\"](.+?)[」』”\"]",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    kind_text, old_name, new_name = match.groups()
    kinds = (
        (coerce_prompt_kind(kind_text, fallback_text=kind_text),)
        if kind_text
        else tuple(PromptKind)
    )
    preset = resolve_prompt_by_name(old_name, kinds=kinds, cache_root=cache_root)
    if preset is None:
        return None
    return preset, old_name.strip(), new_name.strip()[:MAX_TITLE_LENGTH]


def resolve_prompt_by_name(
    name: str,
    *,
    kinds: tuple[PromptKind, ...],
    cache_root: Path,
) -> PromptPreset | None:
    normalized = normalize_lookup_label(name)
    matches: list[PromptPreset] = []
    for kind in kinds:
        matches.extend(
            preset
            for preset in prompt_store_for(cache_root, kind).load()
            if normalize_lookup_label(preset.id) == normalized
            or normalize_lookup_label(preset.name) == normalized
        )
    if len(matches) == 1:
        return matches[0]
    custom_matches = [preset for preset in matches if not preset.is_system]
    if len(custom_matches) == 1:
        return custom_matches[0]
    return None


def extract_recipe_save_name(text: str) -> str:
    matches = re.findall(
        r"(?:保存成|保存为|存成|存为)(?:名为|名字叫|叫)?[「『“\"](.+?)[」』”\"](?:的)?(?:预设|配方)",
        text,
    )
    if not matches:
        return ""
    return matches[-1].strip()[:MAX_RECIPE_NAME_LENGTH]
