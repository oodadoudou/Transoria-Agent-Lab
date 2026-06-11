from __future__ import annotations

from pathlib import Path

from transoria.agent.schemas import AgentWorkspaceState
from transoria.bridge.handlers.agent_task_drafts import (
    build_start_task_draft,
    build_start_with_default_stage_config,
)
from transoria.llm.config import ModelConfig, ProviderFormat
from transoria.prompts import DEFAULT_GLOSSARY_PRESET_ID


def test_build_start_task_draft_copies_payload() -> None:
    payload = {"input_dir": "/tmp/input"}

    draft = build_start_task_draft(
        draft_kind="start_glossary_task",
        title="启动术语提取",
        summary="使用当前配置启动术语提取。",
        payload=payload,
    )

    assert draft.kind == "start_glossary_task"
    assert draft.payload == payload
    assert draft.payload is not payload


def test_default_stage_config_builds_compound_start_draft(tmp_path: Path) -> None:
    state = AgentWorkspaceState.empty().with_config(
        workflow_model_id="profile-workflow"
    )

    result = build_start_with_default_stage_config(
        draft_kind="start_glossary_task",
        payload={
            "input_dir": "/tmp/input",
            "output_dir": "/tmp/output",
            "source_language": "kr",
            "target_language": "zh",
            "novel_background": "现代 BL",
        },
        state=state,
        completeness_missing=(
            "stage_model_ids.term_extract",
            "stage_prompt_ids.term_extract",
        ),
        cache_root=tmp_path,
        title="补齐术语提取配置并启动任务",
        start_title="启动术语提取",
        start_summary="启动术语提取。",
        reply="请确认组合草案。",
    )

    assert result is not None
    reply, draft = result
    assert reply == "请确认组合草案。"
    assert draft.kind == "compound_config_update"
    actions = draft.payload["actions"]
    assert isinstance(actions, list)
    assert [action["kind"] for action in actions] == [
        "update_workspace",
        "start_glossary_task",
    ]
    assert actions[0]["payload"] == {
        "stage_model_ids": {"term_extract": "profile-workflow"},
        "stage_prompt_ids": {"term_extract": DEFAULT_GLOSSARY_PRESET_ID},
    }
    assert actions[1]["payload"]["input_dir"] == "/tmp/input"


def test_requested_stage_model_overrides_workflow_model(tmp_path: Path) -> None:
    state = AgentWorkspaceState.empty().with_config(
        workflow_model_id="profile-workflow"
    )
    requested = ModelConfig(
        id="profile-requested",
        display_name="Requested Model",
        provider_format=ProviderFormat.OPENAI,
        base_url="https://example.com/v1",
        model_id="requested-model",
    )

    result = build_start_with_default_stage_config(
        draft_kind="start_glossary_task",
        payload={
            "input_dir": "/tmp/input",
            "output_dir": "/tmp/output",
            "source_language": "kr",
            "target_language": "zh",
            "novel_background": "现代 BL",
        },
        state=state,
        completeness_missing=("stage_model_ids.term_extract",),
        cache_root=tmp_path,
        title="补齐术语提取配置并启动任务",
        start_title="启动术语提取",
        start_summary="启动术语提取。",
        reply="请确认组合草案。",
        requested_stage_model=requested,
    )

    assert result is not None
    _, draft = result
    actions = draft.payload["actions"]
    assert isinstance(actions, list)
    assert actions[0]["payload"] == {
        "stage_model_ids": {"term_extract": "profile-requested"},
    }
    assert "Requested Model" in str(actions[0]["summary"])


def test_default_stage_config_ignores_missing_payload_fields(tmp_path: Path) -> None:
    state = AgentWorkspaceState.empty().with_config(
        workflow_model_id="profile-workflow"
    )

    result = build_start_with_default_stage_config(
        draft_kind="start_glossary_task",
        payload={},
        state=state,
        completeness_missing=("stage_model_ids.term_extract", "input_dir"),
        cache_root=tmp_path,
        title="补齐术语提取配置并启动任务",
        start_title="启动术语提取",
        start_summary="启动术语提取。",
        reply="请确认组合草案。",
    )

    assert result is None
