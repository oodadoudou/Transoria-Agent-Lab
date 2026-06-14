"""Live end-to-end checks for the Agent Lab loop against a real local LLM.

These talk to a real OpenAI-compatible endpoint, so they are opt-in:

    AGENT_E2E=1 uv run pytest -q tests/integration/test_agent_local_model.py

Without ``AGENT_E2E`` set the module is skipped, so the normal suite stays
fast and offline. Even when enabled, each test skips if the endpoint cannot be
reached, so a stopped local server never turns the run red.

Config via env (defaults target the local test endpoint):
    AGENT_E2E_BASE_URL   default http://127.0.0.1:7861/antigravity/v1
    AGENT_E2E_API_KEY    default pwd
    AGENT_E2E_MODEL      default gemini-3-flash-agent
"""

from __future__ import annotations

import functools
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pytest

from transoria.bridge import build_default_router
from transoria.bridge.task_registry import RunningTask
from transoria.bridge.task_service import TaskService
from transoria.domain import TaskKind, TaskStatus
from transoria.llm.config import ModelConfig, ProviderFormat
from transoria.model_profiles import ModelProfileStore
from transoria.runtime.task_record import TaskRecord

pytestmark = pytest.mark.skipif(
    not os.environ.get("AGENT_E2E"),
    reason="set AGENT_E2E=1 to run local-model end-to-end tests",
)

BASE_URL = os.environ.get("AGENT_E2E_BASE_URL", "http://127.0.0.1:7861/antigravity/v1")
API_KEY = os.environ.get("AGENT_E2E_API_KEY", "pwd")
MODEL = os.environ.get("AGENT_E2E_MODEL", "gemini-3-flash-agent")


def _probe(path: str, *, method: str = "GET", body: bytes | None = None) -> int | None:
    """Return the HTTP status for a probe call, or None if unreachable."""
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except (urllib.error.URLError, OSError):
        return None


@functools.cache
def _endpoint_reachable() -> bool:
    """Liveness check, computed once per session (``@cache``).

    Prefers a lightweight ``GET /models`` call so the opt-in suite adds almost
    no load on the local model; falls back to a tiny chat completion only when
    the list endpoint is unavailable. Caching means the whole run makes at most
    one liveness call instead of one per test, so a momentarily slow server no
    longer skips a subset of tests mid-run.
    """
    if _probe("/models") == 200:
        return True
    chat = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }
    ).encode("utf-8")
    return _probe("/chat/completions", method="POST", body=chat) == 200


def _build_local_router(tmp_path: Path):
    if not _endpoint_reachable():
        pytest.skip(f"local model endpoint unreachable: {BASE_URL}")
    ModelProfileStore.from_cache_root(tmp_path).create(
        ModelConfig(
            id="local-workflow",
            display_name="Local workflow model",
            provider_format=ProviderFormat.OPENAI,
            base_url=BASE_URL,
            model_id=MODEL,
            api_keys=(API_KEY,),
        )
    )
    built = build_default_router(cache_root=tmp_path)
    built.call(
        "agent.update_workspace", {"patch": {"workflow_model_id": "local-workflow"}}
    )
    return built


@pytest.fixture
def router(tmp_path: Path):
    return _build_local_router(tmp_path)


def test_real_chat_returns_assistant_reply(router) -> None:
    response = router.call(
        "agent.send_message", {"message": "你好，请用一句话说明你能帮我做什么。"}
    )
    last = response["workspace"]["messages"][-1]
    assert last["role"] == "assistant"
    assert last["content"].strip()


def test_real_agent_drafts_and_applies_prompt_preset(router) -> None:
    response = router.call(
        "agent.send_message",
        {
            "message": (
                "帮我创建一个翻译用的 prompt 预设，名字叫『现代中文叙事』，"
                "面向现代中文小说叙事，尽量避免源语言残留，人名保持一致。"
            )
        },
    )
    draft = response["workspace"]["pending_draft"]
    assert draft is not None, "expected the agent to draft a prompt preset"
    assert draft["kind"] == "create_prompt_preset"

    applied = router.call("agent.apply_draft", {"draft_id": draft["id"]})
    assert applied["workspace"]["pending_draft"] is None
    preset = applied["result"]["preset"]
    assert preset["kind"] == "translation"
    assert preset["system_prompt"].strip()
    # The newly created preset is now selectable for the translation slot.
    assert any(
        p["id"] == preset["id"]
        for p in applied["inventory"]["prompts"]["translation"]
    )


def test_real_agent_drafts_and_applies_recipe(router) -> None:
    response = router.call(
        "agent.send_message",
        {"message": "把当前模型和阶段选择存成一个叫『本地测试』的配方。"},
    )
    draft = response["workspace"]["pending_draft"]
    assert draft is not None, "expected the agent to draft a recipe"
    assert draft["kind"] == "create_recipe"

    applied = router.call("agent.apply_draft", {"draft_id": draft["id"]})
    assert applied["workspace"]["recipes"]
    assert applied["result"]["recipe"]["name"]


def test_real_agent_handles_naive_unknown_model_request(router) -> None:
    response = router.call(
        "agent.send_message",
        {
            "message": (
                "我想加一个更厉害的模型来翻译小说，但我不知道模型 ID、"
                "接口地址和 key 是什么，你帮我配一下。"
            )
        },
    )

    workspace = response["workspace"]
    assert workspace["pending_draft"] is None
    reply = workspace["messages"][-1]["content"]
    assert "model" in reply.lower() or "模型" in reply
    assert "key" in reply.lower() or "密钥" in reply


def test_real_agent_drafts_and_applies_dumb_glossary_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_start_agent_task(**kwargs: object) -> dict[str, object]:
        service = kwargs["task_service"]
        assert isinstance(service, TaskService)
        started_at = datetime.now(timezone.utc).isoformat()
        service.cache.save_task(
            TaskRecord(
                id="local-glossary-e2e",
                kind=TaskKind.GLOSSARY,
                status=TaskStatus.RUNNING,
                created_at=started_at,
                updated_at=started_at,
            )
        )
        service.registry.add(
            RunningTask(
                task_id="local-glossary-e2e",
                kind="glossary",
                cache=service.cache,
                created_at=started_at,
            )
        )
        return {
            "task_id": "local-glossary-e2e",
            "started_at": started_at,
        }

    monkeypatch.setattr(
        "transoria.bridge.handlers.agent.start_agent_task",
        fake_start_agent_task,
    )
    router = _build_local_router(tmp_path)
    source_dir = tmp_path / "novel"
    source_dir.mkdir()
    (source_dir / "sample.txt").write_text(
        "이승원은 윤정현을 바라보았다.\n윤정현은 대답하지 않았다.",
        encoding="utf-8",
    )

    response = router.call(
        "agent.send_message",
        {
            "message": (
                f"帮我傻瓜式处理术语：{source_dir}\n"
                "输出和输入放在同一个文件夹。\n"
                "BL 作品指南\n背景/类型：现代\n"
                "作品关键词：严肃、爱恨交织、禁忌关系。\n"
                "人物：李承元和尹正贤。"
            )
        },
    )

    draft = response["workspace"]["pending_draft"]
    assert draft is not None, "expected a glossary workflow draft"
    assert draft["kind"] == "compound_config_update"
    actions = draft["payload"]["actions"]
    assert actions[0]["kind"] == "update_workspace"
    assert actions[1]["kind"] == "start_glossary_task"
    payload = actions[1]["payload"]
    assert payload["input_dir"] == str(source_dir)
    assert payload["output_dir"] == str(source_dir)
    assert payload["source_language"] == "kr"
    assert payload["target_language"] == "zh"
    assert "严肃" in payload["novel_background"]

    applied = router.call("agent.apply_draft", {"draft_id": draft["id"]})
    assert applied["workspace"]["pending_draft"] is None
    assert applied["workspace"]["active_task"]["task_id"] == "local-glossary-e2e"
    assert applied["workspace"]["active_task"]["kind"] == "glossary"
    final_message = applied["workspace"]["messages"][-1]["content"]
    assert "任务 ID：local-glossary-e2e" in final_message
    assert "术语提取 dashboard" in final_message

    active = router.call("agent.get_active_task", {})
    assert active["active_task"]["task_id"] == "local-glossary-e2e"
    assert active["task"]["kind"] == "glossary"
    assert active["task"]["status"] == "running"
    recent = router.call(
        "agent.list_recent_task_summaries",
        {"kind": "glossary", "limit": 1},
    )
    assert recent["tasks"][0]["id"] == "local-glossary-e2e"
