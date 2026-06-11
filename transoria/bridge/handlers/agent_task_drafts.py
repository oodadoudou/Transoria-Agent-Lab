"""Task-start draft builders for Agent Lab bridge handlers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from transoria.agent.schemas import AgentActionDraft, AgentWorkspaceState
from transoria.llm.config import ModelConfig
from transoria.prompts import (
    DEFAULT_GLOSSARY_PRESET_ID,
    DEFAULT_GLOSSARY_REVIEW_PRESET_ID,
    DEFAULT_TRANSLATION_PRESET_ID,
    PromptKind,
    PromptPresetStore,
)

PROMPT_KIND_BY_STAGE = {
    "translation": PromptKind.TRANSLATION,
    "term_extract": PromptKind.GLOSSARY,
    "term_review": PromptKind.GLOSSARY_REVIEW,
}

DEFAULT_PROMPT_ID_BY_STAGE = {
    "translation": DEFAULT_TRANSLATION_PRESET_ID,
    "term_extract": DEFAULT_GLOSSARY_PRESET_ID,
    "term_review": DEFAULT_GLOSSARY_REVIEW_PRESET_ID,
}

STAGE_BY_START_DRAFT_KIND = {
    "start_glossary_task": "term_extract",
    "start_glossary_review_task": "term_review",
    "start_translation_task": "translation",
}


def build_start_task_draft(
    *,
    draft_kind: str,
    title: str,
    summary: str,
    payload: Mapping[str, object],
) -> AgentActionDraft:
    return AgentActionDraft.create(
        kind=draft_kind,
        title=title,
        summary=summary,
        payload=dict(payload),
    )


def build_start_with_default_stage_config(
    *,
    draft_kind: str,
    payload: Mapping[str, object],
    state: AgentWorkspaceState,
    completeness_missing: Sequence[str],
    cache_root: Path,
    title: str,
    start_title: str,
    start_summary: str,
    reply: str,
    requested_stage_model: ModelConfig | None = None,
) -> tuple[str, AgentActionDraft] | None:
    stage = STAGE_BY_START_DRAFT_KIND.get(draft_kind)
    if stage is None:
        return None

    missing = set(completeness_missing)
    model_key = f"stage_model_ids.{stage}"
    prompt_key = f"stage_prompt_ids.{stage}"
    if not missing or not missing.issubset({model_key, prompt_key}):
        return None

    workspace_patch: dict[str, object] = {}
    if model_key in missing:
        stage_model_id = (
            requested_stage_model.id
            if requested_stage_model is not None
            else state.workflow_model_id
        )
        if not stage_model_id:
            return None
        workspace_patch["stage_model_ids"] = {stage: stage_model_id}
    if prompt_key in missing:
        prompt_id = DEFAULT_PROMPT_ID_BY_STAGE.get(stage)
        if not prompt_id:
            return None
        prompt_kind = PROMPT_KIND_BY_STAGE[stage]
        presets = _prompt_store_for(cache_root, prompt_kind).load()
        if not any(preset.id == prompt_id for preset in presets):
            return None
        workspace_patch["stage_prompt_ids"] = {stage: prompt_id}
    if not workspace_patch:
        return None

    actions: list[dict[str, object]] = [
        {
            "kind": "update_workspace",
            "title": "补齐当前流程阶段配置",
            "summary": (
                f"使用请求中匹配到的现有模型 {requested_stage_model.display_name} "
                "和系统默认 Prompt 补齐本次任务需要的阶段配置。"
                if requested_stage_model is not None and model_key in missing
                else "使用当前工作模型和系统默认 Prompt 补齐本次任务需要的阶段配置。"
            ),
            "payload": workspace_patch,
        },
        {
            "kind": draft_kind,
            "title": start_title,
            "summary": start_summary,
            "payload": dict(payload),
        },
    ]
    draft = AgentActionDraft.create(
        kind="compound_config_update",
        title=title,
        summary="先补齐缺失的阶段模型/Prompt，再启动对应任务。所有动作都会在你点击「应用」后按顺序执行。",
        payload={"actions": actions},
    )
    return reply, draft


def _prompt_store_for(cache_root: Path, kind: PromptKind) -> PromptPresetStore:
    return PromptPresetStore(
        path=cache_root / f"prompts.{kind.value}.json",
        kind=kind,
    )


__all__ = [
    "DEFAULT_PROMPT_ID_BY_STAGE",
    "PROMPT_KIND_BY_STAGE",
    "STAGE_BY_START_DRAFT_KIND",
    "build_start_task_draft",
    "build_start_with_default_stage_config",
]
