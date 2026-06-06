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

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from transoria.bridge import build_default_router
from transoria.llm.config import ModelConfig, ProviderFormat
from transoria.model_profiles import ModelProfileStore

pytestmark = pytest.mark.skipif(
    not os.environ.get("AGENT_E2E"),
    reason="set AGENT_E2E=1 to run local-model end-to-end tests",
)

BASE_URL = os.environ.get("AGENT_E2E_BASE_URL", "http://127.0.0.1:7861/antigravity/v1")
API_KEY = os.environ.get("AGENT_E2E_API_KEY", "pwd")
MODEL = os.environ.get("AGENT_E2E_MODEL", "gemini-3-flash-agent")


def _endpoint_reachable() -> bool:
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


@pytest.fixture
def router(tmp_path: Path):
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
