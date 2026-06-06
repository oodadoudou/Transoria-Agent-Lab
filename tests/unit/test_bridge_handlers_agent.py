from __future__ import annotations

import json
from pathlib import Path

import pytest

from transoria.agent.configuration_agent import AGENT_SYSTEM_PROMPT
from transoria.agent.project_store import AgentProjectStore
from transoria.agent.schemas import AgentActiveTask
from transoria.bridge import BridgeError, build_default_router
from transoria.bridge.handlers.settings import default_store
from transoria.bridge.task_service import TaskService
from transoria.domain import TaskKind, TaskStatus
from transoria.llm.client import ChatRequest, ChatResponse, LlmRequestError
from transoria.llm.config import ModelConfig, ProviderFormat, ThinkingLevel
from transoria.llm.usage import TokenUsage
from transoria.model_profiles import ModelProfileStore
from transoria.prompts import (
    DEFAULT_GLOSSARY_REVIEW_PRESET_ID,
    DEFAULT_TRANSLATION_PRESET_ID,
    PromptKind,
    PromptPreset,
    PromptPresetStore,
)
from transoria.runtime.task_record import TaskRecord
from transoria.runtime.cache import TaskCache


class RaisingAgentClient:
    """Fake client whose chat() raises, to exercise error handling."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.requests: list[ChatRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        raise self.error


class FakeAgentClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.requests: list[ChatRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        return ChatResponse(
            content=self.content,
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )


class SequenceAgentClient:
    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.requests: list[ChatRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        content = self.contents.pop(0)
        return ChatResponse(
            content=content,
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )


class FallbackAgentClient:
    def __init__(self, error: Exception, content: str) -> None:
        self.error = error
        self.content = content
        self.requests: list[ChatRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            raise self.error
        return ChatResponse(
            content=self.content,
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )


def _seed_profile(cache_root: Path) -> ModelConfig:
    store = ModelProfileStore.from_cache_root(cache_root)
    return store.create(
        ModelConfig(
            id="profile-workflow",
            display_name="Workflow",
            provider_format=ProviderFormat.OPENAI,
            base_url="https://example.com/v1",
            model_id="workflow-model",
            api_keys=("test-key",),
            concurrency_limit=3,
            rpm_limit=120,
            tpm_limit=60000,
            retry_attempts=4,
        )
    )


def _seed_weak_profile(cache_root: Path) -> ModelConfig:
    store = ModelProfileStore.from_cache_root(cache_root)
    return store.create(
        ModelConfig(
            id="profile-flash",
            display_name="Flash Budget",
            provider_format=ProviderFormat.OPENAI,
            base_url="https://example.com/v1",
            model_id="gemini-3-flash",
            api_keys=("weak-key",),
            concurrency_limit=8,
        )
    )


def _seed_deepseek_profile(cache_root: Path) -> ModelConfig:
    store = ModelProfileStore.from_cache_root(cache_root)
    return store.create(
        ModelConfig(
            id="deepseek-f",
            display_name="DeepSeek-f",
            provider_format=ProviderFormat.OPENAI,
            base_url="http://127.0.0.1:7861/antigravity/v1",
            model_id="deepseek-v4-flash",
            api_keys=("deepseek-key",),
            concurrency_limit=2,
            rpm_limit=90,
            tpm_limit=12345,
            retry_attempts=3,
            thinking_level=ThinkingLevel.LOW,
        )
    )


def _seed_thinking_profile(cache_root: Path) -> ModelConfig:
    store = ModelProfileStore.from_cache_root(cache_root)
    return store.create(
        ModelConfig(
            id="profile-thinking",
            display_name="Thinking Workflow",
            provider_format=ProviderFormat.OPENAI,
            base_url="https://example.com/v1",
            model_id="thinking-model",
            api_keys=("thinking-key",),
            thinking_level=ThinkingLevel.MEDIUM,
        )
    )


def _seed_custom_prompt(
    cache_root: Path,
    *,
    kind: PromptKind,
    preset_id: str,
    name: str,
) -> PromptPreset:
    store = PromptPresetStore(
        path=cache_root / f"prompts.{kind.value}.json",
        kind=kind,
    )
    preset = PromptPreset(
        id=preset_id,
        name=name,
        kind=kind,
        system_prompt="Custom prompt body.",
        description="Custom test preset.",
        enabled=True,
        is_system=False,
    )
    store.save([*store.load(), preset])
    return preset


def _write_task(
    cache_root: Path,
    *,
    task_id: str,
    kind: TaskKind,
    status: TaskStatus,
) -> None:
    TaskCache(root=cache_root / "tasks").save_task(
        TaskRecord(
            id=task_id,
            kind=kind,
            status=status,
            created_at="2026-01-01T00:00:00.000+00:00",
            updated_at="2026-01-01T00:00:01.000+00:00",
        )
    )


def test_read_workspace_returns_default_history_and_inventory(tmp_path: Path) -> None:
    _seed_profile(tmp_path)
    router = build_default_router(cache_root=tmp_path)

    response = router.call("agent.read_workspace", {})

    workspace = response["workspace"]
    inventory = response["inventory"]
    assert isinstance(workspace, dict)
    assert isinstance(inventory, dict)
    assert workspace["messages"]
    assert workspace["memories"] == []
    assert workspace["workflow_thinking_level"] == "off"
    [profile] = inventory["profiles"]  # type: ignore[index]
    assert profile["id"] == "profile-workflow"
    assert profile["supports_thinking"] is False
    assert profile["concurrency_limit"] == 3
    assert profile["rpm_limit"] == 120
    assert profile["tpm_limit"] == 60000
    assert profile["retry_attempts"] == 4


def test_agent_workspace_thinking_follows_selected_workflow_profile(
    tmp_path: Path,
) -> None:
    _seed_profile(tmp_path)
    _seed_thinking_profile(tmp_path)
    router = build_default_router(cache_root=tmp_path)

    response = router.call(
        "agent.update_workspace",
        {"patch": {"workflow_model_id": "profile-thinking"}},
    )

    workspace = response["workspace"]
    assert workspace["workflow_model_id"] == "profile-thinking"
    assert workspace["workflow_thinking_level"] == "medium"
    thinking_profile = next(
        profile
        for profile in response["inventory"]["profiles"]  # type: ignore[index]
        if profile["id"] == "profile-thinking"
    )
    assert thinking_profile["supports_thinking"] is True


def test_agent_workspace_rejects_thinking_for_non_thinking_profile(
    tmp_path: Path,
) -> None:
    _seed_profile(tmp_path)
    router = build_default_router(cache_root=tmp_path)
    router.call(
        "agent.update_workspace",
        {"patch": {"workflow_model_id": "profile-workflow"}},
    )

    with pytest.raises(BridgeError):
        router.call(
            "agent.update_workspace",
            {"patch": {"workflow_thinking_level": "high"}},
        )


def test_agent_chat_uses_workspace_thinking_override(tmp_path: Path) -> None:
    _seed_thinking_profile(tmp_path)
    fake = FakeAgentClient('{"reply":"ok","draft":null}')
    router = build_default_router(cache_root=tmp_path, llm_client_factory=lambda: fake)
    router.call(
        "agent.update_workspace",
        {"patch": {"workflow_model_id": "profile-thinking"}},
    )
    router.call(
        "agent.update_workspace",
        {"patch": {"workflow_thinking_level": "high"}},
    )

    router.call("agent.send_message", {"message": "hello"})

    assert fake.requests
    assert fake.requests[0].model.id == "profile-thinking"
    assert fake.requests[0].model.thinking_level is ThinkingLevel.HIGH


def test_agent_read_only_queries_do_not_mutate_workspace(tmp_path: Path) -> None:
    _seed_profile(tmp_path)
    router = build_default_router(cache_root=tmp_path)
    router.call(
        "agent.create_recipe",
        {
            "name": "Baseline",
            "stage_model_ids": {"translation": "profile-workflow"},
            "stage_prompt_ids": {"translation": DEFAULT_TRANSLATION_PRESET_ID},
        },
    )
    before = router.call("agent.read_workspace", {})["workspace"]

    profiles = router.call("agent.list_model_profiles", {})
    prompts = router.call("agent.list_prompt_presets", {"kind": "translation"})
    recipes = router.call("agent.list_recipes", {})
    active = router.call("agent.get_active_task", {})
    recent = router.call("agent.list_recent_task_summaries", {"limit": 2})

    assert profiles["profiles"][0]["id"] == "profile-workflow"  # type: ignore[index]
    assert prompts["kind"] == "translation"
    assert recipes["recipes"][0]["name"] == "Baseline"  # type: ignore[index]
    assert active == {"active_task": None, "task": None}
    assert recent["tasks_by_kind"]["translation"] == []  # type: ignore[index]
    assert router.call("agent.read_workspace", {})["workspace"] == before


def test_agent_recent_task_summaries_include_artifact_availability(
    tmp_path: Path,
) -> None:
    _write_task(
        tmp_path,
        task_id="translation-done",
        kind=TaskKind.TRANSLATION,
        status=TaskStatus.COMPLETED,
    )
    result_path = tmp_path / "tasks" / "translation-done" / "result.json"
    result_path.write_text(
        json.dumps({"output_files": ["book.txt"], "statistics": {}}),
        encoding="utf-8",
    )
    router = build_default_router(cache_root=tmp_path)

    response = router.call(
        "agent.list_recent_task_summaries",
        {"kind": "translation", "limit": 1},
    )

    [task] = response["tasks"]  # type: ignore[index]
    assert task["id"] == "translation-done"
    assert task["artifact_available"] is True
    assert task["artifact_keys"] == ["output_files", "statistics"]


def test_agent_artifact_availability_returns_false_for_missing_task(
    tmp_path: Path,
) -> None:
    router = build_default_router(cache_root=tmp_path)

    response = router.call(
        "agent.get_artifact_availability",
        {"kind": "translation", "task_id": "translation-missing"},
    )

    assert response["available"] is False
    assert response["error"]["code"] == "bridge.not_found"  # type: ignore[index]


def test_agent_get_active_task_reconciles_terminal_cache_record(
    tmp_path: Path,
) -> None:
    _write_task(
        tmp_path,
        task_id="translation-done",
        kind=TaskKind.TRANSLATION,
        status=TaskStatus.COMPLETED,
    )
    store = AgentProjectStore.from_cache_root(tmp_path)
    state = store.load().with_active_task(
        AgentActiveTask.create(
            task_id="translation-done",
            kind="translation",
            conversation_id=store.load().active_conversation_id or "",
            started_at="2026-01-01T00:00:00.000+00:00",
        )
    )
    store.save(state)
    router = build_default_router(cache_root=tmp_path)

    response = router.call("agent.get_active_task", {})

    assert response == {"active_task": None, "task": None}
    assert router.call("agent.read_workspace", {})["workspace"]["active_task"] is None


def test_chat_without_workflow_model_is_saved_without_llm_call(tmp_path: Path) -> None:
    fake = FakeAgentClient('{"reply":"should not be called","draft":null}')
    router = build_default_router(cache_root=tmp_path, llm_client_factory=lambda: fake)

    response = router.call("agent.send_message", {"message": "hello"})

    workspace = response["workspace"]
    assert isinstance(workspace, dict)
    assert len(workspace["messages"]) == 3
    assert workspace["messages"][-2]["role"] == "user"  # type: ignore[index]
    assert workspace["messages"][-1]["role"] == "assistant"  # type: ignore[index]
    assert fake.requests == []

    reloaded = router.call("agent.read_workspace", {})["workspace"]
    assert isinstance(reloaded, dict)
    assert reloaded["messages"] == workspace["messages"]


def test_agent_draft_can_create_prompt_after_confirmation(tmp_path: Path) -> None:
    _seed_profile(tmp_path)
    fake = FakeAgentClient(
        """
        {
          "reply": "I prepared a prompt draft.",
          "draft": {
            "kind": "create_prompt_preset",
            "title": "Create literary prompt",
            "summary": "Adds a translation prompt preset.",
            "payload": {
              "kind": "translation",
              "name": "Literary",
              "description": "Literary translation",
              "system_prompt": "Translate with literary Chinese.",
              "enabled": true
            }
          }
        }
        """
    )
    router = build_default_router(cache_root=tmp_path, llm_client_factory=lambda: fake)
    router.call(
        "agent.update_workspace",
        {"patch": {"workflow_model_id": "profile-workflow"}},
    )

    response = router.call("agent.send_message", {"message": "make a prompt"})
    workspace = response["workspace"]
    assert isinstance(workspace, dict)
    draft = workspace["pending_draft"]
    assert draft["kind"] == "create_prompt_preset"  # type: ignore[index]

    applied = router.call("agent.apply_draft", {"draft_id": draft["id"]})

    applied_workspace = applied["workspace"]
    assert isinstance(applied_workspace, dict)
    assert applied_workspace["pending_draft"] is None
    assert applied["result"]["preset"]["name"] == "Literary"  # type: ignore[index]


def test_agent_memory_update_requires_confirmation(tmp_path: Path) -> None:
    _seed_profile(tmp_path)
    fake = FakeAgentClient(
        """
        {
          "reply": "I prepared memory updates.",
          "draft": {
            "kind": "update_memory",
            "title": "Update memory",
            "summary": "Remember the user's translation preference.",
            "payload": {
              "memories": [
                "Prefer literary but not archaic Chinese for narration."
              ]
            }
          }
        }
        """
    )
    router = build_default_router(cache_root=tmp_path, llm_client_factory=lambda: fake)
    router.call(
        "agent.update_workspace",
        {"patch": {"workflow_model_id": "profile-workflow"}},
    )

    response = router.call("agent.send_message", {"message": "remember my style"})
    workspace = response["workspace"]
    assert isinstance(workspace, dict)
    assert workspace["memories"] == []
    draft = workspace["pending_draft"]

    applied = router.call("agent.apply_draft", {"draft_id": draft["id"]})

    applied_workspace = applied["workspace"]
    assert isinstance(applied_workspace, dict)
    assert applied_workspace["memories"] == [
        "Prefer literary but not archaic Chinese for narration."
    ]


def _router_with_workflow(tmp_path: Path, content: str):
    _seed_profile(tmp_path)
    fake = FakeAgentClient(content)
    router = build_default_router(cache_root=tmp_path, llm_client_factory=lambda: fake)
    router.call(
        "agent.update_workspace",
        {"patch": {"workflow_model_id": "profile-workflow"}},
    )
    return router, fake


# --- Robustness: malformed / invalid model output ---------------------------


def test_malformed_model_json_keeps_chat_recoverable(tmp_path: Path) -> None:
    router, _ = _router_with_workflow(tmp_path, "this is not json at all")

    response = router.call("agent.send_message", {"message": "do something"})
    workspace = response["workspace"]

    assert isinstance(workspace, dict)
    assert workspace["pending_draft"] is None
    assert workspace["messages"][-1]["role"] == "assistant"
    assert workspace["messages"][-1]["content"]
    assert "this is not json" not in workspace["messages"][-1]["content"]


def test_malformed_model_json_can_be_repaired_into_pending_draft(
    tmp_path: Path,
) -> None:
    _seed_profile(tmp_path)
    fake = SequenceAgentClient(
        [
            '我准备了草案：{"reply": "broken", "draft": ',
            """
            {
              "reply": "我已把模型输出整理成可确认的草案。",
              "draft": {
                "kind": "create_prompt_preset",
                "title": "创建文学翻译 Prompt",
                "summary": "创建一个翻译提示词预设。",
                "payload": {
                  "kind": "translation",
                  "name": "文学翻译",
                  "description": "文学翻译提示词",
                  "system_prompt": "请忠实翻译。",
                  "enabled": true
                }
              }
            }
            """,
        ]
    )
    router = build_default_router(cache_root=tmp_path, llm_client_factory=lambda: fake)
    router.call(
        "agent.update_workspace",
        {"patch": {"workflow_model_id": "profile-workflow"}},
    )

    response = router.call("agent.send_message", {"message": "创建一个翻译 prompt"})
    workspace = response["workspace"]

    assert workspace["messages"][-1]["content"] == "我已把模型输出整理成可确认的草案。"
    assert workspace["pending_draft"]["kind"] == "create_prompt_preset"  # type: ignore[index]
    assert len(fake.requests) == 2
    assert fake.requests[0].log_label == "agent configuration chat"
    assert fake.requests[1].log_label == "agent draft repair"


def test_invalid_draft_payload_can_be_repaired_into_pending_draft(
    tmp_path: Path,
) -> None:
    _seed_profile(tmp_path)
    fake = SequenceAgentClient(
        [
            """
            {
              "reply": "我已起草。",
              "draft": {
                "kind": "create_prompt_preset",
                "title": "创建现代中文叙事",
                "summary": "创建翻译提示词预设。",
                "payload": {}
              }
            }
            """,
            """
            {
              "reply": "我已把草案补全为可确认的配置。",
              "draft": {
                "kind": "create_prompt_preset",
                "title": "创建现代中文叙事",
                "summary": "创建翻译提示词预设。",
                "payload": {
                  "kind": "translation",
                  "name": "现代中文叙事",
                  "description": "现代中文小说叙事提示词",
                  "system_prompt": "请用自然现代中文翻译，避免源语言残留。",
                  "enabled": true
                }
              }
            }
            """,
        ]
    )
    router = build_default_router(cache_root=tmp_path, llm_client_factory=lambda: fake)
    router.call(
        "agent.update_workspace",
        {"patch": {"workflow_model_id": "profile-workflow"}},
    )

    response = router.call("agent.send_message", {"message": "创建一个翻译 prompt"})
    workspace = response["workspace"]

    assert workspace["messages"][-1]["content"] == "我已把草案补全为可确认的配置。"
    assert workspace["pending_draft"]["kind"] == "create_prompt_preset"  # type: ignore[index]
    assert workspace["pending_draft"]["payload"]["name"] == "现代中文叙事"  # type: ignore[index]
    assert len(fake.requests) == 2
    assert fake.requests[1].log_label == "agent draft validation repair"


def test_empty_prompt_payload_can_be_salvaged_after_repair_stays_empty(
    tmp_path: Path,
) -> None:
    _seed_profile(tmp_path)
    fake = SequenceAgentClient(
        [
            """
            {
              "reply": "我已起草。",
              "draft": {
                "kind": "create_prompt_preset",
                "title": "创建翻译 Prompt：现代中文叙事",
                "summary": "创建现代中文叙事翻译提示词。",
                "payload": {}
              }
            }
            """,
            """
            {
              "reply": "我已起草现代中文叙事。",
              "draft": {
                "kind": "create_prompt_preset",
                "title": "创建翻译 Prompt：现代中文叙事",
                "summary": "创建现代中文叙事翻译提示词。",
                "payload": {}
              }
            }
            """,
        ]
    )
    router = build_default_router(cache_root=tmp_path, llm_client_factory=lambda: fake)
    router.call(
        "agent.update_workspace",
        {"patch": {"workflow_model_id": "profile-workflow"}},
    )

    response = router.call(
        "agent.send_message",
        {"message": "帮我创建一个翻译用的 prompt 预设，名字叫『现代中文叙事』。"},
    )
    draft = response["workspace"]["pending_draft"]

    assert draft["kind"] == "create_prompt_preset"
    assert draft["payload"]["kind"] == "translation"
    assert draft["payload"]["name"] == "现代中文叙事"
    assert draft["payload"]["system_prompt"]
    assert "补齐为可确认的配置草案" in response["workspace"]["messages"][-1]["content"]


def test_empty_recipe_payload_can_be_salvaged_after_repair_stays_empty(
    tmp_path: Path,
) -> None:
    _seed_profile(tmp_path)
    fake = SequenceAgentClient(
        [
            """
            {
              "reply": "我已起草。",
              "draft": {
                "kind": "create_recipe",
                "title": "创建配方『本地测试』",
                "summary": "保存当前配置。",
                "payload": {}
              }
            }
            """,
            """
            {
              "reply": "我已起草本地测试配方。",
              "draft": {
                "kind": "create_recipe",
                "title": "创建配方『本地测试』",
                "summary": "保存当前配置。",
                "payload": {}
              }
            }
            """,
        ]
    )
    router = build_default_router(cache_root=tmp_path, llm_client_factory=lambda: fake)
    router.call(
        "agent.update_workspace",
        {
            "patch": {
                "workflow_model_id": "profile-workflow",
                "stage_model_ids": {"translation": "profile-workflow"},
            }
        },
    )

    response = router.call(
        "agent.send_message",
        {"message": "把当前模型和阶段选择存成一个叫『本地测试』的配方。"},
    )
    draft = response["workspace"]["pending_draft"]

    assert draft["kind"] == "create_recipe"
    assert draft["payload"]["name"] == "本地测试"
    assert draft["payload"]["stage_model_ids"]["translation"] == "profile-workflow"
    assert "补齐为可确认的配置草案" in response["workspace"]["messages"][-1]["content"]


def test_invalid_draft_kind_is_not_stored(tmp_path: Path) -> None:
    router, _ = _router_with_workflow(
        tmp_path,
        """
        {
          "reply": "Here is a draft.",
          "draft": {
            "kind": "run_translation",
            "title": "Run it",
            "summary": "Should not be allowed.",
            "payload": {}
          }
        }
        """,
    )

    response = router.call("agent.send_message", {"message": "translate the book"})
    workspace = response["workspace"]

    assert workspace["pending_draft"] is None


def test_invalid_prompt_preset_draft_is_not_stored(tmp_path: Path) -> None:
    router, _ = _router_with_workflow(
        tmp_path,
        """
        {
          "reply": "Draft prepared.",
          "draft": {
            "kind": "create_prompt_preset",
            "title": "Bad preset",
            "summary": "Missing system prompt.",
            "payload": {"kind": "translation", "name": "X"}
          }
        }
        """,
    )

    response = router.call("agent.send_message", {"message": "make a preset"})

    assert response["workspace"]["pending_draft"] is None


def test_invalid_memory_draft_is_not_stored(tmp_path: Path) -> None:
    router, _ = _router_with_workflow(
        tmp_path,
        """
        {
          "reply": "Draft prepared.",
          "draft": {
            "kind": "update_memory",
            "title": "Bad memory",
            "summary": "Not a list.",
            "payload": {"memories": "just a string"}
          }
        }
        """,
    )

    response = router.call("agent.send_message", {"message": "remember this"})

    assert response["workspace"]["pending_draft"] is None


# --- Conversation lifecycle -------------------------------------------------


def test_create_and_switch_conversation_isolates_messages(tmp_path: Path) -> None:
    router, _ = _router_with_workflow(tmp_path, '{"reply":"ok","draft":null}')
    first_id = router.call("agent.read_workspace", {})["workspace"][
        "active_conversation_id"
    ]

    created = router.call("agent.create_conversation", {})["workspace"]
    second_id = created["active_conversation_id"]
    assert second_id != first_id
    assert len(created["conversations"]) == 2

    router.call("agent.send_message", {"message": "only in second"})

    switched = router.call(
        "agent.switch_conversation", {"conversation_id": first_id}
    )["workspace"]
    assert switched["active_conversation_id"] == first_id
    contents = [m["content"] for m in switched["messages"]]
    assert "only in second" not in contents


def test_rename_conversation_keeps_active_pointer(tmp_path: Path) -> None:
    router, _ = _router_with_workflow(tmp_path, '{"reply":"ok","draft":null}')
    workspace = router.call("agent.read_workspace", {})["workspace"]
    active_id = workspace["active_conversation_id"]
    other = router.call("agent.create_conversation", {})["workspace"]
    other_id = other["active_conversation_id"]

    renamed = router.call(
        "agent.rename_conversation",
        {"conversation_id": active_id, "title": "Renamed thread"},
    )["workspace"]

    assert renamed["active_conversation_id"] == other_id
    titles = {c["id"]: c["title"] for c in renamed["conversations"]}
    assert titles[active_id] == "Renamed thread"


def test_delete_active_conversation_falls_back(tmp_path: Path) -> None:
    router, _ = _router_with_workflow(tmp_path, '{"reply":"ok","draft":null}')
    first_id = router.call("agent.read_workspace", {})["workspace"][
        "active_conversation_id"
    ]
    second_id = router.call("agent.create_conversation", {})["workspace"][
        "active_conversation_id"
    ]

    remaining = router.call(
        "agent.delete_conversation", {"conversation_id": second_id}
    )["workspace"]

    ids = [c["id"] for c in remaining["conversations"]]
    assert second_id not in ids
    assert remaining["active_conversation_id"] == first_id


def test_delete_last_conversation_recreates_one(tmp_path: Path) -> None:
    router, _ = _router_with_workflow(tmp_path, '{"reply":"ok","draft":null}')
    only_id = router.call("agent.read_workspace", {})["workspace"][
        "active_conversation_id"
    ]

    workspace = router.call(
        "agent.delete_conversation", {"conversation_id": only_id}
    )["workspace"]

    assert len(workspace["conversations"]) == 1
    assert workspace["conversations"][0]["id"] != only_id
    assert workspace["active_conversation_id"] is not None


def test_conversation_methods_reject_unknown_id(tmp_path: Path) -> None:
    router, _ = _router_with_workflow(tmp_path, '{"reply":"ok","draft":null}')

    for method in (
        "agent.switch_conversation",
        "agent.delete_conversation",
    ):
        with pytest.raises(BridgeError):
            router.call(method, {"conversation_id": "missing"})
    with pytest.raises(BridgeError):
        router.call(
            "agent.rename_conversation",
            {"conversation_id": "missing", "title": "x"},
        )


# --- Direct (user-initiated) memory management ------------------------------


def test_update_memory_direct_replaces_and_dedupes(tmp_path: Path) -> None:
    router = build_default_router(cache_root=tmp_path)

    workspace = router.call(
        "agent.update_memory",
        {"memories": ["  keep names consistent  ", "keep names consistent", "no residue"]},
    )["workspace"]

    assert workspace["memories"] == ["keep names consistent", "no residue"]


def test_delete_memory_direct_removes_entry(tmp_path: Path) -> None:
    router = build_default_router(cache_root=tmp_path)
    router.call("agent.update_memory", {"memories": ["a", "b", "c"]})

    workspace = router.call("agent.delete_memory", {"memory": "b"})["workspace"]

    assert workspace["memories"] == ["a", "c"]


def test_update_memory_rejects_non_list(tmp_path: Path) -> None:
    router = build_default_router(cache_root=tmp_path)

    with pytest.raises(BridgeError):
        router.call("agent.update_memory", {"memories": "not a list"})


def test_legacy_flat_workspace_migrates_to_conversation(tmp_path: Path) -> None:
    import json

    legacy = {
        "workflow_model_id": None,
        "messages": [
            {"id": "msg-1", "role": "assistant", "content": "old thread"},
        ],
        "memories": ["old preference"],
    }
    target = tmp_path / "agent_lab" / "workspace.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(legacy), encoding="utf-8")

    router = build_default_router(cache_root=tmp_path)
    workspace = router.call("agent.read_workspace", {})["workspace"]

    assert len(workspace["conversations"]) == 1
    assert workspace["active_conversation_id"] is not None
    assert workspace["memories"] == ["old preference"]
    assert workspace["messages"][0]["content"] == "old thread"


# --- Recipes ----------------------------------------------------------------


def test_create_apply_recipe_copies_into_stage_selections(tmp_path: Path) -> None:
    _seed_profile(tmp_path)
    router = build_default_router(cache_root=tmp_path)

    created = router.call(
        "agent.create_recipe",
        {
            "name": "Cheap bulk",
            "description": "cheap translation model",
            "stage_model_ids": {"translation": "profile-workflow"},
        },
    )["workspace"]
    [recipe] = created["recipes"]
    assert recipe["name"] == "Cheap bulk"
    assert recipe["stage_model_ids"] == {
        "translation": "profile-workflow",
        "term_extract": None,
        "term_review": None,
    }

    applied = router.call("agent.apply_recipe", {"recipe_id": recipe["id"]})[
        "workspace"
    ]
    assert applied["stage_model_ids"]["translation"] == "profile-workflow"


def test_update_and_delete_recipe(tmp_path: Path) -> None:
    _seed_profile(tmp_path)
    router = build_default_router(cache_root=tmp_path)
    recipe_id = router.call("agent.create_recipe", {"name": "Draft"})["workspace"][
        "recipes"
    ][0]["id"]

    renamed = router.call(
        "agent.update_recipe", {"recipe_id": recipe_id, "name": "Final"}
    )["workspace"]
    assert renamed["recipes"][0]["name"] == "Final"

    emptied = router.call("agent.delete_recipe", {"recipe_id": recipe_id})[
        "workspace"
    ]
    assert emptied["recipes"] == []


def test_create_recipe_rejects_unknown_model_id(tmp_path: Path) -> None:
    router = build_default_router(cache_root=tmp_path)

    with pytest.raises(BridgeError):
        router.call(
            "agent.create_recipe",
            {"name": "Bad", "stage_model_ids": {"translation": "ghost"}},
        )


def test_create_recipe_requires_name(tmp_path: Path) -> None:
    router = build_default_router(cache_root=tmp_path)

    with pytest.raises(BridgeError):
        router.call("agent.create_recipe", {"name": "   "})


def test_recipe_methods_reject_unknown_id(tmp_path: Path) -> None:
    router = build_default_router(cache_root=tmp_path)

    for method in ("agent.delete_recipe", "agent.apply_recipe"):
        with pytest.raises(BridgeError):
            router.call(method, {"recipe_id": "missing"})
    with pytest.raises(BridgeError):
        router.call("agent.update_recipe", {"recipe_id": "missing", "name": "x"})


def test_agent_can_draft_recipe_after_confirmation(tmp_path: Path) -> None:
    router, _ = _router_with_workflow(
        tmp_path,
        """
        {
          "reply": "Here is a recipe.",
          "draft": {
            "kind": "create_recipe",
            "title": "Save bulk recipe",
            "summary": "Cheap translation stage.",
            "payload": {
              "name": "Bulk",
              "description": "cheap bulk translation",
              "stage_model_ids": {"translation": "profile-workflow"}
            }
          }
        }
        """,
    )

    response = router.call("agent.send_message", {"message": "save a recipe"})
    draft = response["workspace"]["pending_draft"]
    assert draft["kind"] == "create_recipe"

    applied = router.call("agent.apply_draft", {"draft_id": draft["id"]})
    workspace = applied["workspace"]
    assert workspace["pending_draft"] is None
    assert workspace["recipes"][0]["name"] == "Bulk"
    assert applied["result"]["recipe"]["name"] == "Bulk"


def test_invalid_recipe_draft_is_not_stored(tmp_path: Path) -> None:
    router, _ = _router_with_workflow(
        tmp_path,
        """
        {
          "reply": "Draft prepared.",
          "draft": {
            "kind": "create_recipe",
            "title": "Bad recipe",
            "summary": "Missing name.",
            "payload": {"description": "no name"}
          }
        }
        """,
    )

    response = router.call("agent.send_message", {"message": "save a recipe"})

    assert response["workspace"]["pending_draft"] is None


# --- Agent → API call mechanics ---------------------------------------------


def test_agent_start_translation_draft_applies_with_task_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_start_agent_task(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        service = kwargs["task_service"]
        assert isinstance(service, TaskService)
        service.cache.save_task(
            TaskRecord(
                id="translation-agent-1",
                kind=TaskKind.TRANSLATION,
                status=TaskStatus.RUNNING,
            )
        )
        return {
            "task_id": "translation-agent-1",
            "started_at": "2026-01-01T00:00:00+00:00",
        }

    monkeypatch.setattr(
        "transoria.bridge.handlers.agent.start_agent_task",
        fake_start_agent_task,
    )
    input_dir = tmp_path / "novel"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "book.txt").write_text("source text", encoding="utf-8")
    router, _ = _router_with_workflow(
        tmp_path,
        f"""
        {{
          "reply": "Ready to start translation.",
          "draft": {{
            "kind": "start_translation_task",
            "title": "Start translation",
            "summary": "Use per-task chat overrides.",
            "payload": {{
              "input_dir": "{input_dir}",
              "output_dir": "{output_dir}",
              "source_language": "kr",
              "target_language": "zh"
            }}
          }}
        }}
        """,
    )
    router.call(
        "agent.update_workspace",
        {
            "patch": {
                "stage_model_ids": {"translation": "profile-workflow"},
                "stage_prompt_ids": {
                    "translation": DEFAULT_TRANSLATION_PRESET_ID,
                },
            }
        },
    )

    response = router.call("agent.send_message", {"message": "start translation"})
    draft = response["workspace"]["pending_draft"]
    assert draft["kind"] == "start_translation_task"

    applied = router.call("agent.apply_draft", {"draft_id": draft["id"]})
    workspace = applied["workspace"]

    assert workspace["active_task"] == {
        "task_id": "translation-agent-1",
        "kind": "translation",
        "conversation_id": workspace["active_conversation_id"],
        "started_at": "2026-01-01T00:00:00+00:00",
    }
    assert applied["result"]["task"]["kind"] == "translation"
    final_message = workspace["messages"][-1]["content"]
    assert "已启动翻译任务" in final_message
    assert "任务 ID：translation-agent-1" in final_message
    assert "当前进度阶段：正在运行" in final_message
    assert "翻译 dashboard" in final_message
    assert captured["draft_kind"] == "start_translation_task"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["input_dir"] == str(input_dir)
    settings = default_store(tmp_path).load_all()
    assert settings.translation.input_folder == ""
    assert settings.translation.output_folder == ""


def test_agent_directly_drafts_glossary_extraction_from_chat_inputs(
    tmp_path: Path,
) -> None:
    _seed_profile(tmp_path)
    _seed_custom_prompt(
        tmp_path,
        kind=PromptKind.GLOSSARY,
        preset_id="glossary-custom",
        name="Glossary Custom",
    )
    fake = RaisingAgentClient(AssertionError("LLM should not be called"))
    router = build_default_router(cache_root=tmp_path, llm_client_factory=lambda: fake)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    router.call(
        "agent.update_workspace",
        {
            "patch": {
                "workflow_model_id": "profile-workflow",
                "stage_model_ids": {"term_extract": "profile-workflow"},
                "stage_prompt_ids": {"term_extract": "glossary-custom"},
            }
        },
    )

    response = router.call(
        "agent.send_message",
        {
            "message": (
                f"请提取术语。input: {source_dir}。"
                "小说背景：韩式现代奇幻，人物关系复杂。"
            )
        },
    )

    assert fake.requests == []
    draft = response["workspace"]["pending_draft"]
    assert draft["kind"] == "start_glossary_task"
    assert draft["payload"]["input_dir"].rstrip("/") == str(source_dir)
    assert draft["payload"]["output_dir"].rstrip("/") == str(source_dir)
    assert draft["payload"]["source_language"] == "kr"
    assert draft["payload"]["target_language"] == "zh"
    assert draft["payload"]["novel_background"] == "韩式现代奇幻，人物关系复杂"
    assert "默认输出到 input 目录" in response["workspace"]["messages"][-1]["content"]


def test_agent_new_glossary_request_discards_stale_model_copy_draft(
    tmp_path: Path,
) -> None:
    _seed_profile(tmp_path)
    _seed_deepseek_profile(tmp_path)
    fake = RaisingAgentClient(AssertionError("LLM should not be called"))
    router = build_default_router(cache_root=tmp_path, llm_client_factory=lambda: fake)
    router.call(
        "agent.update_workspace",
        {"patch": {"workflow_model_id": "profile-workflow"}},
    )

    stale = router.call(
        "agent.send_message",
        {
            "message": (
                "按照 DeepSeek flash 的配置复制一个新模型配置，"
                "模型名字叫做 DeepSeek-P。"
            )
        },
    )["workspace"]["pending_draft"]
    assert stale["kind"] == "create_model_profile"

    response = router.call("agent.send_message", {"message": "提取术语"})

    workspace = response["workspace"]
    assert workspace["pending_draft"] is None
    assert workspace["draft_history"][-1]["id"] == stale["id"]
    assert workspace["draft_history"][-1]["status"] == "discarded"
    assert "请提供 input 目录" in workspace["messages"][-1]["content"]


def test_agent_glossary_continuation_replaces_model_copy_context(
    tmp_path: Path,
) -> None:
    _seed_profile(tmp_path)
    _seed_deepseek_profile(tmp_path)
    _seed_custom_prompt(
        tmp_path,
        kind=PromptKind.GLOSSARY,
        preset_id="glossary-custom",
        name="Glossary Custom",
    )
    fake = RaisingAgentClient(AssertionError("LLM should not be called"))
    router = build_default_router(cache_root=tmp_path, llm_client_factory=lambda: fake)
    source_dir = tmp_path / "source copy"
    source_dir.mkdir()
    router.call(
        "agent.update_workspace",
        {
            "patch": {
                "workflow_model_id": "profile-workflow",
                "stage_model_ids": {"term_extract": "profile-workflow"},
                "stage_prompt_ids": {"term_extract": "glossary-custom"},
            }
        },
    )
    router.call(
        "agent.send_message",
        {
            "message": (
                "按照 DeepSeek flash 的配置复制一个新模型配置，"
                "模型名字叫做 DeepSeek-P。"
            )
        },
    )
    router.call("agent.send_message", {"message": "提取术语"})

    response = router.call(
        "agent.send_message",
        {
            "message": (
                f"{source_dir}/\n"
                "输出和输入放在同一个文件夹里。BL 作品指南\n\n"
                "背景/类型：现代\n\n"
                "作品关键词：严肃、爱恨交织、禁忌关系\n\n"
                "人物介绍\n攻：李承元"
            )
        },
    )

    assert fake.requests == []
    draft = response["workspace"]["pending_draft"]
    assert draft["kind"] == "start_glossary_task"
    assert draft["payload"]["input_dir"].rstrip("/") == str(source_dir)
    assert draft["payload"]["output_dir"].rstrip("/") == str(source_dir)
    assert draft["payload"]["novel_background"] == "现代"
    assert draft["payload"]["source_language"] == "kr"
    assert "启动术语提取" in draft["title"]


def test_agent_drafts_glossary_from_directory_and_background_without_llm(
    tmp_path: Path,
) -> None:
    _seed_profile(tmp_path)
    _seed_custom_prompt(
        tmp_path,
        kind=PromptKind.GLOSSARY,
        preset_id="glossary-custom",
        name="Glossary Custom",
    )
    fake = RaisingAgentClient(AssertionError("LLM should not be called"))
    router = build_default_router(cache_root=tmp_path, llm_client_factory=lambda: fake)
    source_dir = tmp_path / "keyword-source"
    source_dir.mkdir()
    router.call(
        "agent.update_workspace",
        {
            "patch": {
                "workflow_model_id": "profile-workflow",
                "stage_model_ids": {"term_extract": "profile-workflow"},
                "stage_prompt_ids": {"term_extract": "glossary-custom"},
            }
        },
    )

    response = router.call(
        "agent.send_message",
        {
            "message": (
                f"{source_dir}\n"
                "输出和输入放在同一个文件夹里。\n\n"
                "背景/类型：现代 BL，严肃，爱恨交织。"
            )
        },
    )

    assert fake.requests == []
    draft = response["workspace"]["pending_draft"]
    assert draft["kind"] == "start_glossary_task"
    assert draft["payload"]["input_dir"] == str(source_dir)
    assert draft["payload"]["output_dir"] == str(source_dir)
    assert draft["payload"]["novel_background"] == "现代 BL，严肃，爱恨交织"


def test_agent_glossary_extraction_requires_background(
    tmp_path: Path,
) -> None:
    _seed_profile(tmp_path)
    _seed_custom_prompt(
        tmp_path,
        kind=PromptKind.GLOSSARY,
        preset_id="glossary-custom",
        name="Glossary Custom",
    )
    fake = RaisingAgentClient(AssertionError("LLM should not be called"))
    router = build_default_router(cache_root=tmp_path, llm_client_factory=lambda: fake)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    router.call(
        "agent.update_workspace",
        {
            "patch": {
                "workflow_model_id": "profile-workflow",
                "stage_model_ids": {"term_extract": "profile-workflow"},
                "stage_prompt_ids": {"term_extract": "glossary-custom"},
            }
        },
    )

    response = router.call(
        "agent.send_message",
        {"message": f"提取术语，输入目录：{source_dir}"},
    )

    assert fake.requests == []
    assert response["workspace"]["pending_draft"] is None
    assert "请再提供小说背景" in response["workspace"]["messages"][-1]["content"]


def test_agent_compound_draft_can_fill_recipe_and_start_translation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_start_agent_task(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        service = kwargs["task_service"]
        assert isinstance(service, TaskService)
        service.cache.save_task(
            TaskRecord(
                id="translation-agent-compound",
                kind=TaskKind.TRANSLATION,
                status=TaskStatus.RUNNING,
            )
        )
        return {
            "task_id": "translation-agent-compound",
            "started_at": "2026-01-01T00:00:00+00:00",
        }

    monkeypatch.setattr(
        "transoria.bridge.handlers.agent.start_agent_task",
        fake_start_agent_task,
    )
    input_dir = tmp_path / "novel"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "book.txt").write_text("source text", encoding="utf-8")
    router, _ = _router_with_workflow(
        tmp_path,
        f"""
        {{
          "reply": "I prepared one proposal to configure and start translation.",
          "draft": {{
            "kind": "compound_config_update",
            "title": "Configure and start translation",
            "summary": "Fills the translation recipe slots and starts translation.",
            "payload": {{
              "actions": [
                {{
                  "kind": "update_workspace",
                  "title": "Fill translation slots",
                  "summary": "Selects the model and prompt for this run.",
                  "payload": {{
                    "stage_model_ids": {{"translation": "profile-workflow"}},
                    "stage_prompt_ids": {{
                      "translation": "{DEFAULT_TRANSLATION_PRESET_ID}"
                    }}
                  }}
                }},
                {{
                  "kind": "start_translation_task",
                  "title": "Start translation",
                  "summary": "Uses chat-provided per-task paths.",
                  "payload": {{
                    "input_dir": "{input_dir}",
                    "output_dir": "{output_dir}",
                    "source_language": "kr",
                    "target_language": "zh"
                  }}
                }}
              ]
            }}
          }}
        }}
        """,
    )

    draft = router.call(
        "agent.send_message",
        {"message": "把翻译模型和 prompt 配好，然后用这些目录开始翻译"},
    )["workspace"]["pending_draft"]

    assert draft["kind"] == "compound_config_update"

    applied = router.call("agent.apply_draft", {"draft_id": draft["id"]})
    workspace = applied["workspace"]

    assert workspace["active_task"] == {
        "task_id": "translation-agent-compound",
        "kind": "translation",
        "conversation_id": workspace["active_conversation_id"],
        "started_at": "2026-01-01T00:00:00+00:00",
    }
    assert captured["draft_kind"] == "start_translation_task"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["input_dir"] == str(input_dir)
    assert [item["kind"] for item in applied["result"]["results"]] == [
        "update_workspace",
        "start_translation_task",
    ]
    final_message = workspace["messages"][-1]["content"]
    assert "已启动翻译任务" in final_message
    assert "任务 ID：translation-agent-compound" in final_message
    assert "翻译 dashboard" in final_message
    settings = default_store(tmp_path).load_all()
    assert settings.translation.input_folder == ""
    assert settings.translation.output_folder == ""


def test_agent_start_translation_draft_is_dropped_when_recipe_incomplete(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "novel"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "book.txt").write_text("source text", encoding="utf-8")
    router, _ = _router_with_workflow(
        tmp_path,
        f"""
        {{
          "reply": "Ready to start translation.",
          "draft": {{
            "kind": "start_translation_task",
            "title": "Start translation",
            "summary": "Missing translation stage recipe.",
            "payload": {{
              "input_dir": "{input_dir}",
              "output_dir": "{output_dir}",
              "source_language": "kr",
              "target_language": "zh"
            }}
          }}
        }}
        """,
    )

    response = router.call("agent.send_message", {"message": "start translation"})

    assert response["workspace"]["pending_draft"] is None
    assert "已忽略无法应用的草案" in response["workspace"]["messages"][-1]["content"]


def test_agent_start_task_draft_is_blocked_by_active_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_start_agent_task(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        service = kwargs["task_service"]
        assert isinstance(service, TaskService)
        service.cache.save_task(
            TaskRecord(
                id=f"translation-agent-{calls}",
                kind=TaskKind.TRANSLATION,
                status=TaskStatus.RUNNING,
            )
        )
        return {
            "task_id": f"translation-agent-{calls}",
            "started_at": "2026-01-01T00:00:00+00:00",
        }

    monkeypatch.setattr(
        "transoria.bridge.handlers.agent.start_agent_task",
        fake_start_agent_task,
    )
    input_dir = tmp_path / "novel"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "book.txt").write_text("source text", encoding="utf-8")
    content = f"""
    {{
      "reply": "Ready to start translation.",
      "draft": {{
        "kind": "start_translation_task",
        "title": "Start translation",
        "summary": "Use per-task chat overrides.",
        "payload": {{
          "input_dir": "{input_dir}",
          "output_dir": "{output_dir}",
          "source_language": "kr",
          "target_language": "zh"
        }}
      }}
    }}
    """
    router, _ = _router_with_workflow(tmp_path, content)
    router.call(
        "agent.update_workspace",
        {
            "patch": {
                "stage_model_ids": {"translation": "profile-workflow"},
                "stage_prompt_ids": {
                    "translation": DEFAULT_TRANSLATION_PRESET_ID,
                },
            }
        },
    )
    draft = router.call("agent.send_message", {"message": "start translation"})[
        "workspace"
    ]["pending_draft"]
    router.call("agent.apply_draft", {"draft_id": draft["id"]})

    response = router.call("agent.send_message", {"message": "start another"})

    assert response["workspace"]["pending_draft"] is None
    assert calls == 1
    assert "检测到当前正在执行 translation 任务" in response["workspace"]["messages"][-1][
        "content"
    ]


def test_agent_active_task_lock_clears_after_terminal_cache_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_start_agent_task(**kwargs: object) -> dict[str, object]:
        service = kwargs["task_service"]
        assert isinstance(service, TaskService)
        service.cache.save_task(
            TaskRecord(
                id="translation-agent-done",
                kind=TaskKind.TRANSLATION,
                status=TaskStatus.COMPLETED,
            )
        )
        return {
            "task_id": "translation-agent-done",
            "started_at": "2026-01-01T00:00:00+00:00",
        }

    monkeypatch.setattr(
        "transoria.bridge.handlers.agent.start_agent_task",
        fake_start_agent_task,
    )
    input_dir = tmp_path / "novel"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "book.txt").write_text("source text", encoding="utf-8")
    router, _ = _router_with_workflow(
        tmp_path,
        f"""
        {{
          "reply": "Ready to start translation.",
          "draft": {{
            "kind": "start_translation_task",
            "title": "Start translation",
            "summary": "Use per-task chat overrides.",
            "payload": {{
              "input_dir": "{input_dir}",
              "output_dir": "{output_dir}",
              "source_language": "kr",
              "target_language": "zh"
            }}
          }}
        }}
        """,
    )
    router.call(
        "agent.update_workspace",
        {
            "patch": {
                "stage_model_ids": {"translation": "profile-workflow"},
                "stage_prompt_ids": {
                    "translation": DEFAULT_TRANSLATION_PRESET_ID,
                },
            }
        },
    )
    draft = router.call("agent.send_message", {"message": "start translation"})[
        "workspace"
    ]["pending_draft"]

    applied = router.call("agent.apply_draft", {"draft_id": draft["id"]})
    assert applied["workspace"]["active_task"]["task_id"] == "translation-agent-done"

    reloaded = router.call("agent.read_workspace", {})["workspace"]
    assert reloaded["active_task"] is None


def test_glossary_review_start_draft_with_unknown_task_id_is_dropped(
    tmp_path: Path,
) -> None:
    router, _ = _router_with_workflow(
        tmp_path,
        """
        {
          "reply": "Ready to review glossary.",
          "draft": {
            "kind": "start_glossary_review_task",
            "title": "Start glossary review",
            "summary": "Review a glossary task.",
            "payload": {
              "glossary_task_id": "missing-glossary-task"
            }
          }
        }
        """,
    )
    router.call(
        "agent.update_workspace",
        {
            "patch": {
                "stage_model_ids": {"term_review": "profile-workflow"},
                "stage_prompt_ids": {
                    "term_review": DEFAULT_GLOSSARY_REVIEW_PRESET_ID,
                },
            }
        },
    )

    response = router.call("agent.send_message", {"message": "review glossary"})

    assert response["workspace"]["pending_draft"] is None
    assert "已忽略无法应用的草案" in response["workspace"]["messages"][-1]["content"]


def test_send_message_calls_api_with_system_prompt_and_inventory(
    tmp_path: Path,
) -> None:
    router, fake = _router_with_workflow(tmp_path, '{"reply":"ok","draft":null}')

    router.call("agent.send_message", {"message": "configure things"})
    router.call("agent.send_message", {"message": "use the previous context"})

    assert len(fake.requests) == 2
    request = fake.requests[-1]
    assert request.model.id == "profile-workflow"
    assert request.system_prompt == AGENT_SYSTEM_PROMPT
    assert request.stream is False
    assert request.json_response_schema is not None
    assert request.json_response_schema_name == "agent_configuration_response"
    # The user prompt carries the inventory and the user's message.
    assert "profile-workflow" in request.user_prompt
    assert "base_url" in request.user_prompt
    assert "configure things" in request.user_prompt
    assert "use the previous context" in request.user_prompt


def test_send_message_falls_back_when_schema_request_is_rejected(
    tmp_path: Path,
) -> None:
    _seed_profile(tmp_path)
    fake = FallbackAgentClient(
        LlmRequestError("unsupported response_format", code="llm.http_error"),
        '{"reply":"ok","draft":null}',
    )
    router = build_default_router(cache_root=tmp_path, llm_client_factory=lambda: fake)
    router.call(
        "agent.update_workspace",
        {"patch": {"workflow_model_id": "profile-workflow"}},
    )

    response = router.call("agent.send_message", {"message": "configure things"})

    assert response["workspace"]["messages"][-1]["content"] == "ok"
    assert len(fake.requests) == 2
    assert fake.requests[0].json_response_schema is not None
    assert fake.requests[1].json_response_schema is None


def test_send_message_skips_api_when_workflow_profile_missing(tmp_path: Path) -> None:
    router, fake = _router_with_workflow(tmp_path, '{"reply":"ok","draft":null}')
    # Delete the selected profile out from under the workspace.
    ModelProfileStore.from_cache_root(tmp_path).delete("profile-workflow")

    response = router.call("agent.send_message", {"message": "hi"})

    assert fake.requests == []
    assert response["workspace"]["pending_draft"] is None
    assert response["workspace"]["messages"][-1]["role"] == "assistant"


def test_send_message_skips_api_when_no_api_key(tmp_path: Path) -> None:
    store = ModelProfileStore.from_cache_root(tmp_path)
    store.create(
        ModelConfig(
            id="profile-nokey",
            display_name="No key",
            provider_format=ProviderFormat.OPENAI,
            base_url="https://example.com/v1",
            model_id="m",
            api_keys=(),
        )
    )
    fake = FakeAgentClient('{"reply":"ok","draft":null}')
    router = build_default_router(cache_root=tmp_path, llm_client_factory=lambda: fake)
    router.call(
        "agent.update_workspace", {"patch": {"workflow_model_id": "profile-nokey"}}
    )

    response = router.call("agent.send_message", {"message": "hi"})

    assert fake.requests == []
    assert response["workspace"]["messages"][-1]["role"] == "assistant"


def test_send_message_handles_llm_request_error(tmp_path: Path) -> None:
    _seed_profile(tmp_path)
    raising = RaisingAgentClient(LlmRequestError("boom", code="llm.http_error"))
    router = build_default_router(
        cache_root=tmp_path, llm_client_factory=lambda: raising
    )
    router.call(
        "agent.update_workspace", {"patch": {"workflow_model_id": "profile-workflow"}}
    )

    response = router.call("agent.send_message", {"message": "hi"})

    workspace = response["workspace"]
    assert workspace["pending_draft"] is None
    assert "llm.http_error" in workspace["messages"][-1]["content"]


def test_send_message_handles_unexpected_error(tmp_path: Path) -> None:
    _seed_profile(tmp_path)
    raising = RaisingAgentClient(RuntimeError("kaboom"))
    router = build_default_router(
        cache_root=tmp_path, llm_client_factory=lambda: raising
    )
    router.call(
        "agent.update_workspace", {"patch": {"workflow_model_id": "profile-workflow"}}
    )

    response = router.call("agent.send_message", {"message": "hi"})

    assert response["workspace"]["pending_draft"] is None
    assert response["workspace"]["messages"][-1]["role"] == "assistant"


def test_send_message_validates_input(tmp_path: Path) -> None:
    router = build_default_router(cache_root=tmp_path)

    with pytest.raises(BridgeError):
        router.call("agent.send_message", {"message": "   "})
    with pytest.raises(BridgeError):
        router.call("agent.send_message", {"message": "x" * 8001})


# --- Agent configures a model / prompt (draft → confirm) --------------------


def test_agent_drafts_model_selection_and_apply(tmp_path: Path) -> None:
    router, _ = _router_with_workflow(
        tmp_path,
        """
        {
          "reply": "Selecting the translation model.",
          "draft": {
            "kind": "update_workspace",
            "title": "Pick translation model",
            "summary": "Use the workflow profile for translation too.",
            "payload": {"stage_model_ids": {"translation": "profile-workflow"}}
          }
        }
        """,
    )

    draft = router.call("agent.send_message", {"message": "set translation model"})[
        "workspace"
    ]["pending_draft"]
    assert draft["kind"] == "update_workspace"

    applied = router.call("agent.apply_draft", {"draft_id": draft["id"]})["workspace"]
    assert applied["stage_model_ids"]["translation"] == "profile-workflow"


def test_agent_compound_draft_applies_workspace_and_model_profile_updates(
    tmp_path: Path,
) -> None:
    router, _ = _router_with_workflow(
        tmp_path,
        """
        {
          "reply": "I prepared one proposal with both changes.",
          "draft": {
            "kind": "compound_config_update",
            "title": "Update translation setup",
            "summary": "Selects the translation model and updates model limits.",
            "payload": {
              "actions": [
                {
                  "kind": "update_workspace",
                  "title": "Select translation model",
                  "summary": "Use Workflow for translation.",
                  "payload": {
                    "stage_model_ids": {"translation": "profile-workflow"}
                  }
                },
                {
                  "kind": "update_model_profile",
                  "title": "Update model limits",
                  "summary": "Raises concurrency for the selected model.",
                  "payload": {
                    "profile_id": "profile-workflow",
                    "patch": {
                      "concurrency_limit": 7,
                      "rpm_limit": 90
                    }
                  }
                }
              ]
            }
          }
        }
        """,
    )

    draft = router.call(
        "agent.send_message",
        {"message": "翻译模型用 Workflow，并把并发改成 7，RPM 改成 90"},
    )["workspace"]["pending_draft"]

    assert draft["kind"] == "compound_config_update"
    before = ModelProfileStore.from_cache_root(tmp_path).get("profile-workflow")
    assert before is not None
    assert before.concurrency_limit == 3

    applied = router.call("agent.apply_draft", {"draft_id": draft["id"]})

    workspace = applied["workspace"]
    assert workspace["stage_model_ids"]["translation"] == "profile-workflow"
    stored = ModelProfileStore.from_cache_root(tmp_path).get("profile-workflow")
    assert stored is not None
    assert stored.concurrency_limit == 7
    assert stored.rpm_limit == 90
    assert [item["kind"] for item in applied["result"]["results"]] == [
        "update_workspace",
        "update_model_profile",
    ]


def test_agent_compound_draft_accepts_action_aliases(
    tmp_path: Path,
) -> None:
    router, _ = _router_with_workflow(
        tmp_path,
        """
        {
          "reply": "I prepared one proposal with a model update.",
          "draft": {
            "kind": "compound_config_update",
            "title": "Update model limits",
            "summary": "Raises concurrency for the selected model.",
            "payload": {
              "changes": [
                {
                  "kind": "update_model_profile",
                  "profile_id": "profile-workflow",
                  "patch": {
                    "concurrency_limit": 4
                  }
                }
              ]
            }
          }
        }
        """,
    )

    draft = router.call(
        "agent.send_message",
        {"message": "把模型 Workflow 的并发数改成 4"},
    )["workspace"]["pending_draft"]

    assert draft["kind"] == "compound_config_update"
    [action] = draft["payload"]["actions"]
    assert action["kind"] == "update_model_profile"
    assert action["payload"] == {
        "profile_id": "profile-workflow",
        "patch": {"concurrency_limit": 4},
    }

    applied = router.call("agent.apply_draft", {"draft_id": draft["id"]})

    stored = ModelProfileStore.from_cache_root(tmp_path).get("profile-workflow")
    assert stored is not None
    assert stored.concurrency_limit == 4
    assert applied["result"]["results"][0]["kind"] == "update_model_profile"


def test_agent_salvages_malformed_compound_draft_from_chinese_request(
    tmp_path: Path,
) -> None:
    _seed_profile(tmp_path)
    custom_prompt = _seed_custom_prompt(
        tmp_path,
        kind=PromptKind.TRANSLATION,
        preset_id="translation-custom-default",
        name="默认",
    )
    fake = SequenceAgentClient(
        [
            """
            {
              "reply": "我已为您准备了复合配置修改草案。",
              "draft": {
                "kind": "compound_config_update",
                "title": "复合配置修改",
                "summary": "修改模型并发数、Prompt 名称，并保存当前阶段配置。",
                "payload": {}
              }
            }
            """,
            """
            {
              "reply": "已重新整理草案。",
              "draft": {
                "kind": "compound_config_update",
                "title": "复合配置修改",
                "summary": "修改模型并发数、Prompt 名称，并保存当前阶段配置。",
                "payload": {}
              }
            }
            """,
        ]
    )
    router = build_default_router(cache_root=tmp_path, llm_client_factory=lambda: fake)
    router.call(
        "agent.update_workspace",
        {
            "patch": {
                "workflow_model_id": "profile-workflow",
                "stage_model_ids": {
                    "translation": "profile-workflow",
                    "term_extract": "profile-workflow",
                    "term_review": "profile-workflow",
                },
                "stage_prompt_ids": {"translation": custom_prompt.id},
            }
        },
    )

    message = (
        "请帮我准备一个配置修改草案：把模型 Workflow 的并发数改成 4，"
        "把翻译 Prompt「默认」重命名为「标准中文翻译预设」，"
        "并把当前阶段配置保存成名为「测试复合配置」的预设。"
    )
    response = router.call("agent.send_message", {"message": message})
    draft = response["workspace"]["pending_draft"]

    assert draft["kind"] == "compound_config_update"
    actions = draft["payload"]["actions"]
    assert [action["kind"] for action in actions] == [
        "update_model_profile",
        "update_prompt_preset",
        "create_recipe",
    ]

    applied = router.call("agent.apply_draft", {"draft_id": draft["id"]})

    stored = ModelProfileStore.from_cache_root(tmp_path).get("profile-workflow")
    assert stored is not None
    assert stored.concurrency_limit == 4
    prompts = PromptPresetStore(
        path=tmp_path / "prompts.translation.json",
        kind=PromptKind.TRANSLATION,
    ).load()
    renamed = next(preset for preset in prompts if preset.id == custom_prompt.id)
    assert renamed.name == "标准中文翻译预设"
    recipes = applied["workspace"]["recipes"]
    assert any(recipe["name"] == "测试复合配置" for recipe in recipes)
    assert [item["kind"] for item in applied["result"]["results"]] == [
        "update_model_profile",
        "update_prompt_preset",
        "create_recipe",
    ]


def test_agent_salvages_system_prompt_rename_as_custom_copy(
    tmp_path: Path,
) -> None:
    _seed_profile(tmp_path)
    fake = SequenceAgentClient(
        [
            """
            {
              "reply": "我已为您准备了复合配置修改草案。",
              "draft": {
                "kind": "compound_config_update",
                "title": "复合配置修改",
                "summary": "重命名 Prompt。",
                "payload": {}
              }
            }
            """,
            """
            {
              "reply": "已重新整理草案。",
              "draft": {
                "kind": "compound_config_update",
                "title": "复合配置修改",
                "summary": "重命名 Prompt。",
                "payload": {}
              }
            }
            """,
        ]
    )
    router = build_default_router(cache_root=tmp_path, llm_client_factory=lambda: fake)
    router.call(
        "agent.update_workspace",
        {"patch": {"workflow_model_id": "profile-workflow"}},
    )

    response = router.call(
        "agent.send_message",
        {"message": "把翻译 Prompt「默认」重命名为「标准中文翻译预设」"},
    )
    draft = response["workspace"]["pending_draft"]

    assert draft["kind"] == "compound_config_update"
    [action] = draft["payload"]["actions"]
    assert action["kind"] == "create_prompt_preset"
    assert action["payload"]["kind"] == "translation"
    assert action["payload"]["name"] == "标准中文翻译预设"
    assert "内置 Prompt 默认 为只读" in action["summary"]

    router.call("agent.apply_draft", {"draft_id": draft["id"]})

    prompts = PromptPresetStore(
        path=tmp_path / "prompts.translation.json",
        kind=PromptKind.TRANSLATION,
    ).load()
    custom = [
        preset
        for preset in prompts
        if preset.name == "标准中文翻译预设" and not preset.is_system
    ]
    assert len(custom) == 1


def test_agent_model_draft_with_unknown_profile_is_dropped(tmp_path: Path) -> None:
    router, _ = _router_with_workflow(
        tmp_path,
        """
        {
          "reply": "Selecting a model.",
          "draft": {
            "kind": "update_workspace",
            "title": "Pick model",
            "summary": "bad id",
            "payload": {"stage_model_ids": {"translation": "ghost"}}
          }
        }
        """,
    )

    response = router.call("agent.send_message", {"message": "set model"})
    assert response["workspace"]["pending_draft"] is None


def test_agent_prompt_preset_then_selectable_in_inventory(tmp_path: Path) -> None:
    router, _ = _router_with_workflow(
        tmp_path,
        """
        {
          "reply": "Drafting a prompt.",
          "draft": {
            "kind": "create_prompt_preset",
            "title": "Literary",
            "summary": "translation prompt",
            "payload": {
              "kind": "translation",
              "name": "Literary KR-ZH",
              "description": "literary",
              "system_prompt": "Translate faithfully.",
              "enabled": true
            }
          }
        }
        """,
    )
    draft = router.call("agent.send_message", {"message": "make a prompt"})[
        "workspace"
    ]["pending_draft"]
    applied = router.call("agent.apply_draft", {"draft_id": draft["id"]})
    preset_id = applied["result"]["preset"]["id"]

    inventory = applied["inventory"]
    translation_prompts = inventory["prompts"]["translation"]
    assert any(p["id"] == preset_id for p in translation_prompts)

    # The newly created prompt can then be selected for the stage slot.
    selected = router.call(
        "agent.update_workspace",
        {"patch": {"stage_prompt_ids": {"translation": preset_id}}},
    )["workspace"]
    assert selected["stage_prompt_ids"]["translation"] == preset_id


def test_agent_create_model_profile_draft_masks_api_key_preview(
    tmp_path: Path,
) -> None:
    secret = "sk-local-secret-123456"
    router, _ = _router_with_workflow(
        tmp_path,
        f"""
        {{
          "reply": "I prepared a masked model profile draft.",
          "draft": {{
            "kind": "create_model_profile",
            "title": "Create local profile",
            "summary": "Adds a local workflow model with the supplied key.",
            "payload": {{
              "profile": {{
                "id": "local-agent",
                "display_name": "Local Agent",
                "provider_format": "openai",
                "base_url": "http://127.0.0.1:7861/antigravity/v1",
                "model_id": "gemini-3-flash-agent",
                "api_keys": ["{secret}"],
                "concurrency_limit": 2
              }}
            }}
          }}
        }}
        """,
    )

    response = router.call("agent.send_message", {"message": "add this model"})
    draft = response["workspace"]["pending_draft"]
    assert "<masked>" in json.dumps(draft)
    assert secret not in json.dumps(draft)
    assert ModelProfileStore.from_cache_root(tmp_path).get("local-agent") is None

    applied = router.call("agent.apply_draft", {"draft_id": draft["id"]})

    stored = ModelProfileStore.from_cache_root(tmp_path).get("local-agent")
    assert stored is not None
    assert stored.api_keys == (secret,)
    assert applied["result"]["profile"]["api_key_status"] == "present"
    assert secret not in json.dumps(applied["workspace"]["draft_history"])
    assert secret not in json.dumps(applied["result"])


def test_agent_directly_drafts_model_profile_copy_from_existing_config(
    tmp_path: Path,
) -> None:
    _seed_profile(tmp_path)
    source = _seed_deepseek_profile(tmp_path)
    fake = RaisingAgentClient(AssertionError("LLM should not be called"))
    router = build_default_router(cache_root=tmp_path, llm_client_factory=lambda: fake)
    router.call(
        "agent.update_workspace",
        {"patch": {"workflow_model_id": "profile-workflow"}},
    )

    response = router.call(
        "agent.send_message",
        {
            "message": (
                "再给我按照 DeepSeek flash 的配置，配置一个 DeepSeek Pro。"
                "除了模型名称之外，其他的都是一样的，模型名字叫做 DeepSeek-P。"
            )
        },
    )

    assert fake.requests == []
    draft = response["workspace"]["pending_draft"]
    assert draft["kind"] == "create_model_profile"
    assert draft["payload"]["profile"]["copy_from_profile_id"] == source.id
    assert draft["payload"]["profile"]["display_name"] == "DeepSeek-P"
    assert draft["payload"]["profile"]["model_id"] == "deepseek-v4-flash"
    assert "deepseek-key" not in json.dumps(draft)

    router.call("agent.apply_draft", {"draft_id": draft["id"]})

    created = [
        profile
        for profile in ModelProfileStore.from_cache_root(tmp_path).load()
        if profile.display_name == "DeepSeek-P"
    ]
    assert len(created) == 1
    assert created[0].base_url == source.base_url
    assert created[0].model_id == source.model_id
    assert created[0].api_keys == source.api_keys
    assert created[0].concurrency_limit == source.concurrency_limit
    assert created[0].thinking_level == source.thinking_level


def test_agent_model_profile_copy_uses_recent_context_for_short_confirmation(
    tmp_path: Path,
) -> None:
    _seed_profile(tmp_path)
    _seed_deepseek_profile(tmp_path)
    fake = FakeAgentClient('{"reply":"我需要确认。","draft":null}')
    router = build_default_router(cache_root=tmp_path, llm_client_factory=lambda: fake)
    router.call(
        "agent.update_workspace",
        {"patch": {"workflow_model_id": "profile-workflow"}},
    )
    router.call(
        "agent.send_message",
        {
            "message": (
                "按照 DeepSeek flash 的配置复制一个新模型配置，"
                "模型名字叫做 DeepSeek-P。"
            )
        },
    )
    workspace = router.call("agent.read_workspace", {})["workspace"]
    router.call(
        "agent.discard_draft",
        {"draft_id": workspace["pending_draft"]["id"]},
    )

    response = router.call(
        "agent.send_message",
        {"message": "对的，创建一个新的。"},
    )

    draft = response["workspace"]["pending_draft"]
    assert draft["kind"] == "create_model_profile"
    assert draft["payload"]["profile"]["display_name"] == "DeepSeek-P"


def test_agent_model_profile_copy_asks_only_for_missing_provider_model_id(
    tmp_path: Path,
) -> None:
    _seed_profile(tmp_path)
    _seed_deepseek_profile(tmp_path)
    fake = RaisingAgentClient(AssertionError("LLM should not be called"))
    router = build_default_router(cache_root=tmp_path, llm_client_factory=lambda: fake)
    router.call(
        "agent.update_workspace",
        {"patch": {"workflow_model_id": "profile-workflow"}},
    )

    response = router.call(
        "agent.send_message",
        {
            "message": (
                "从 DeepSeek flash 的配置复制创建一个新的，不要用相同的模型 ID，"
                "而是用一样的 URL 和 provider format，"
                "然后将模型名称改为 DeepSeek 4 Pro。"
            )
        },
    )

    assert fake.requests == []
    assert response["workspace"]["pending_draft"] is None
    content = response["workspace"]["messages"][-1]["content"]
    assert "准确 model_id" in content
    assert "提供 base_url" not in content
    assert "确认 base_url" not in content


def test_agent_model_profile_placeholder_model_id_draft_is_dropped(
    tmp_path: Path,
) -> None:
    router, _ = _router_with_workflow(
        tmp_path,
        """
        {
          "reply": "我准备了模型配置草案。",
          "draft": {
            "kind": "create_model_profile",
            "title": "复制模型配置为 DeepSeek 4 Pro",
            "summary": "基于 DeepSeek 4 Pro 创建新模型配置。",
            "payload": {
              "profile": {
                "id": "deepseek-4-pro",
                "display_name": "DeepSeek 4 Pro",
                "provider_format": "openai",
                "base_url": "https://api.deepseek.com/v1",
                "model_id": "新的值（例如",
                "concurrency_limit": 4
              }
            }
          }
        }
        """,
    )

    response = router.call("agent.send_message", {"message": "创建 DeepSeek Pro 模型"})

    assert response["workspace"]["pending_draft"] is None
    assert "已忽略无法应用的草案" in response["workspace"]["messages"][-1]["content"]
    assert ModelProfileStore.from_cache_root(tmp_path).get("deepseek-4-pro") is None


def test_agent_update_model_profile_draft_can_rotate_keys_with_masked_preview(
    tmp_path: Path,
) -> None:
    _seed_profile(tmp_path)
    secret = "sk-updated-secret-abcdef"
    fake = FakeAgentClient(
        f"""
        {{
          "reply": "I prepared a key rotation draft.",
          "draft": {{
            "kind": "update_model_profile",
            "title": "Update workflow profile",
            "summary": "Updates concurrency and rotates the supplied key.",
            "payload": {{
              "profile_id": "profile-workflow",
              "patch": {{
                "api_keys": ["{secret}"],
                "concurrency_limit": 5
              }}
            }}
          }}
        }}
        """
    )
    router = build_default_router(cache_root=tmp_path, llm_client_factory=lambda: fake)
    router.call(
        "agent.update_workspace",
        {"patch": {"workflow_model_id": "profile-workflow"}},
    )

    draft = router.call("agent.send_message", {"message": "rotate key"})[
        "workspace"
    ]["pending_draft"]
    assert secret not in json.dumps(draft)

    router.call("agent.apply_draft", {"draft_id": draft["id"]})

    stored = ModelProfileStore.from_cache_root(tmp_path).get("profile-workflow")
    assert stored is not None
    assert stored.api_keys == (secret,)
    assert stored.concurrency_limit == 5


def test_agent_warns_when_weak_model_selected_for_translation(
    tmp_path: Path,
) -> None:
    _seed_weak_profile(tmp_path)
    router, _ = _router_with_workflow(
        tmp_path,
        """
        {
          "reply": "I prepared a model selection draft.",
          "draft": {
            "kind": "update_workspace",
            "title": "Use flash for translation",
            "summary": "Selects the budget model for translation.",
            "payload": {
              "stage_model_ids": {"translation": "profile-flash"}
            }
          }
        }
        """,
    )

    response = router.call("agent.send_message", {"message": "翻译阶段用 flash"})
    content = response["workspace"]["messages"][-1]["content"]

    assert response["workspace"]["pending_draft"]["kind"] == "update_workspace"
    assert "模型风险提示" in content
    assert "翻译阶段" in content
    assert "gemini-3-flash" in content


def test_agent_warns_when_creating_weak_model_profile(tmp_path: Path) -> None:
    router, _ = _router_with_workflow(
        tmp_path,
        """
        {
          "reply": "I prepared a model profile draft.",
          "draft": {
            "kind": "create_model_profile",
            "title": "Create flash model",
            "summary": "Adds a budget model.",
            "payload": {
              "profile": {
                "id": "new-flash",
                "display_name": "Flash Agent",
                "provider_format": "openai",
                "base_url": "https://example.com/v1",
                "model_id": "gemini-3-flash-agent"
              }
            }
          }
        }
        """,
    )

    response = router.call("agent.send_message", {"message": "加一个 flash 模型"})
    content = response["workspace"]["messages"][-1]["content"]

    assert response["workspace"]["pending_draft"]["kind"] == "create_model_profile"
    assert "模型风险提示" in content
    assert "新模型配置" in content
    assert "gemini-3-flash-agent" in content


def test_agent_update_prompt_preset_draft_applies_to_custom_prompt(
    tmp_path: Path,
) -> None:
    router, fake = _router_with_workflow(
        tmp_path,
        """
        {
          "reply": "Drafting a prompt.",
          "draft": {
            "kind": "create_prompt_preset",
            "title": "Create custom prompt",
            "summary": "translation prompt",
            "payload": {
              "kind": "translation",
              "name": "Custom",
              "description": "initial",
              "system_prompt": "Translate plainly.",
              "enabled": true
            }
          }
        }
        """,
    )
    draft = router.call("agent.send_message", {"message": "make prompt"})[
        "workspace"
    ]["pending_draft"]
    preset_id = router.call("agent.apply_draft", {"draft_id": draft["id"]})[
        "result"
    ]["preset"]["id"]
    fake.content = f"""
        {{
          "reply": "Drafting an update.",
          "draft": {{
            "kind": "update_prompt_preset",
            "title": "Update custom prompt",
            "summary": "Tightens the prompt.",
            "payload": {{
              "id": "{preset_id}",
              "patch": {{
                "description": "revised",
                "system_prompt": "Translate with faithful literary Chinese."
              }}
            }}
          }}
        }}
        """

    update_draft = router.call("agent.send_message", {"message": "revise prompt"})[
        "workspace"
    ]["pending_draft"]
    applied = router.call("agent.apply_draft", {"draft_id": update_draft["id"]})

    assert applied["result"]["preset"]["description"] == "revised"
    assert (
        applied["result"]["preset"]["system_prompt"]
        == "Translate with faithful literary Chinese."
    )


def test_agent_recipe_registry_drafts_update_apply_and_delete(
    tmp_path: Path,
) -> None:
    router, fake = _router_with_workflow(tmp_path, '{"reply":"ok","draft":null}')
    recipe_id = router.call("agent.create_recipe", {"name": "Baseline"})["workspace"][
        "recipes"
    ][0]["id"]
    fake.content = f"""
        {{
          "reply": "Drafting a recipe update.",
          "draft": {{
            "kind": "update_recipe",
            "title": "Update recipe",
            "summary": "Sets the translation model.",
            "payload": {{
              "recipe_id": "{recipe_id}",
              "stage_model_ids": {{"translation": "profile-workflow"}}
            }}
          }}
        }}
        """

    update_draft = router.call("agent.send_message", {"message": "update recipe"})[
        "workspace"
    ]["pending_draft"]
    updated = router.call("agent.apply_draft", {"draft_id": update_draft["id"]})[
        "workspace"
    ]
    recipe = next(item for item in updated["recipes"] if item["id"] == recipe_id)
    assert recipe["stage_model_ids"]["translation"] == "profile-workflow"

    fake.content = f"""
        {{
          "reply": "Drafting apply.",
          "draft": {{
            "kind": "apply_recipe",
            "title": "Apply recipe",
            "summary": "Copies the recipe slots.",
            "payload": {{"recipe_id": "{recipe_id}"}}
          }}
        }}
        """
    apply_draft = router.call("agent.send_message", {"message": "apply recipe"})[
        "workspace"
    ]["pending_draft"]
    applied = router.call("agent.apply_draft", {"draft_id": apply_draft["id"]})[
        "workspace"
    ]
    assert applied["stage_model_ids"]["translation"] == "profile-workflow"

    fake.content = f"""
        {{
          "reply": "Drafting delete.",
          "draft": {{
            "kind": "delete_recipe",
            "title": "Delete recipe",
            "summary": "Deletes the recipe.",
            "payload": {{"recipe_id": "{recipe_id}"}}
          }}
        }}
        """
    delete_draft = router.call("agent.send_message", {"message": "delete recipe"})[
        "workspace"
    ]["pending_draft"]
    deleted = router.call("agent.apply_draft", {"draft_id": delete_draft["id"]})[
        "workspace"
    ]
    assert all(item["id"] != recipe_id for item in deleted["recipes"])


def test_agent_memory_registry_drafts_add_and_delete(tmp_path: Path) -> None:
    router, fake = _router_with_workflow(
        tmp_path,
        """
        {
          "reply": "Drafting memory.",
          "draft": {
            "kind": "add_memory",
            "title": "Add memory",
            "summary": "Adds one preference.",
            "payload": {"memory": "Keep honorific nuance when possible."}
          }
        }
        """,
    )

    add_draft = router.call("agent.send_message", {"message": "remember this"})[
        "workspace"
    ]["pending_draft"]
    added = router.call("agent.apply_draft", {"draft_id": add_draft["id"]})[
        "workspace"
    ]
    assert added["memories"] == ["Keep honorific nuance when possible."]

    fake.content = """
        {
          "reply": "Drafting delete.",
          "draft": {
            "kind": "delete_memory",
            "title": "Delete memory",
            "summary": "Removes one preference.",
            "payload": {"memory": "Keep honorific nuance when possible."}
          }
        }
        """
    delete_draft = router.call("agent.send_message", {"message": "forget it"})[
        "workspace"
    ]["pending_draft"]
    deleted = router.call("agent.apply_draft", {"draft_id": delete_draft["id"]})[
        "workspace"
    ]
    assert deleted["memories"] == []


# --- Draft / patch validation edge cases ------------------------------------


def test_update_workspace_requires_patch(tmp_path: Path) -> None:
    router = build_default_router(cache_root=tmp_path)
    with pytest.raises(BridgeError):
        router.call("agent.update_workspace", {})


def test_update_workspace_rejects_unknown_slot(tmp_path: Path) -> None:
    router = build_default_router(cache_root=tmp_path)
    with pytest.raises(BridgeError):
        router.call(
            "agent.update_workspace",
            {"patch": {"stage_model_ids": {"bogus": None}}},
        )


def test_update_workspace_rejects_non_string_id(tmp_path: Path) -> None:
    router = build_default_router(cache_root=tmp_path)
    with pytest.raises(BridgeError):
        router.call(
            "agent.update_workspace",
            {"patch": {"stage_model_ids": {"translation": 5}}},
        )


def test_update_workspace_rejects_unknown_prompt_id(tmp_path: Path) -> None:
    router = build_default_router(cache_root=tmp_path)
    with pytest.raises(BridgeError):
        router.call(
            "agent.update_workspace",
            {"patch": {"stage_prompt_ids": {"translation": "ghost"}}},
        )


def test_apply_draft_without_pending_draft_raises(tmp_path: Path) -> None:
    router = build_default_router(cache_root=tmp_path)
    with pytest.raises(BridgeError):
        router.call("agent.apply_draft", {"draft_id": "nope"})


def test_discard_draft_without_pending_draft_raises(tmp_path: Path) -> None:
    router = build_default_router(cache_root=tmp_path)
    with pytest.raises(BridgeError):
        router.call("agent.discard_draft", {"draft_id": "nope"})


def test_pending_draft_persists_across_reload(tmp_path: Path) -> None:
    router, _ = _router_with_workflow(
        tmp_path,
        """
        {
          "reply": "Drafting.",
          "draft": {
            "kind": "update_memory",
            "title": "Remember",
            "summary": "save",
            "payload": {"memories": ["keep names consistent"]}
          }
        }
        """,
    )
    draft = router.call("agent.send_message", {"message": "remember"})["workspace"][
        "pending_draft"
    ]

    reloaded = router.call("agent.read_workspace", {})["workspace"]
    assert reloaded["pending_draft"]["id"] == draft["id"]


def test_revise_draft_replaces_pending_without_applying(tmp_path: Path) -> None:
    _seed_profile(tmp_path)
    fake = SequenceAgentClient(
        [
            """
            {
              "reply": "Drafting.",
              "draft": {
                "kind": "create_prompt_preset",
                "title": "Create prompt",
                "summary": "Create first prompt.",
                "payload": {
                  "kind": "translation",
                  "name": "First",
                  "description": "first",
                  "system_prompt": "First prompt body.",
                  "enabled": true
                }
              }
            }
            """,
            """
            {
              "reply": "I revised the draft.",
              "draft": {
                "kind": "create_prompt_preset",
                "title": "Create renamed prompt",
                "summary": "Create renamed prompt.",
                "payload": {
                  "kind": "translation",
                  "name": "Renamed",
                  "description": "renamed",
                  "system_prompt": "Renamed prompt body.",
                  "enabled": true
                }
              }
            }
            """,
        ]
    )
    router = build_default_router(cache_root=tmp_path, llm_client_factory=lambda: fake)
    router.call(
        "agent.update_workspace",
        {"patch": {"workflow_model_id": "profile-workflow"}},
    )
    original = router.call("agent.send_message", {"message": "make prompt"})[
        "workspace"
    ]["pending_draft"]

    revised = router.call(
        "agent.revise_draft",
        {"draft_id": original["id"], "adjustment": "把名称改成 Renamed"},
    )["workspace"]

    assert revised["pending_draft"]["id"] != original["id"]
    assert revised["pending_draft"]["payload"]["name"] == "Renamed"
    assert revised["draft_history"][-1]["id"] == original["id"]
    assert revised["draft_history"][-1]["status"] == "discarded"
    saved_names = {
        preset.name
        for preset in PromptPresetStore(
            path=tmp_path / "prompts.translation.json",
            kind=PromptKind.TRANSLATION,
        ).load()
    }
    assert "First" not in saved_names
    assert "Renamed" not in saved_names
    assert "Current pending draft" in fake.requests[-1].user_prompt


def test_revise_draft_keeps_original_when_model_returns_no_draft(
    tmp_path: Path,
) -> None:
    _seed_profile(tmp_path)
    fake = SequenceAgentClient(
        [
            """
            {
              "reply": "Drafting.",
              "draft": {
                "kind": "create_prompt_preset",
                "title": "Create prompt",
                "summary": "Create first prompt.",
                "payload": {
                  "kind": "translation",
                  "name": "First",
                  "description": "first",
                  "system_prompt": "First prompt body.",
                  "enabled": true
                }
              }
            }
            """,
            """
            {
              "reply": "请说明要怎么调整这个草案。",
              "draft": null
            }
            """,
        ]
    )
    router = build_default_router(cache_root=tmp_path, llm_client_factory=lambda: fake)
    router.call(
        "agent.update_workspace",
        {"patch": {"workflow_model_id": "profile-workflow"}},
    )
    original = router.call("agent.send_message", {"message": "make prompt"})[
        "workspace"
    ]["pending_draft"]

    revised = router.call(
        "agent.revise_draft",
        {"draft_id": original["id"], "adjustment": "随便调整一下"},
    )["workspace"]

    assert revised["pending_draft"]["id"] == original["id"]
    assert revised["pending_draft"]["payload"]["name"] == "First"
    assert revised["draft_history"] == []


def test_apply_recipe_fails_after_referenced_profile_deleted(tmp_path: Path) -> None:
    _seed_profile(tmp_path)
    router = build_default_router(cache_root=tmp_path)
    recipe_id = router.call(
        "agent.create_recipe",
        {"name": "R", "stage_model_ids": {"translation": "profile-workflow"}},
    )["workspace"]["recipes"][0]["id"]
    ModelProfileStore.from_cache_root(tmp_path).delete("profile-workflow")

    with pytest.raises(BridgeError):
        router.call("agent.apply_recipe", {"recipe_id": recipe_id})


# --- Inventory shape --------------------------------------------------------


def test_inventory_reports_api_key_state_and_prompt_groups(tmp_path: Path) -> None:
    store = ModelProfileStore.from_cache_root(tmp_path)
    store.create(
        ModelConfig(
            id="with-key",
            display_name="With key",
            provider_format=ProviderFormat.OPENAI,
            base_url="https://example.com/v1",
            model_id="m",
            api_keys=("k",),
        )
    )
    store.create(
        ModelConfig(
            id="without-key",
            display_name="No key",
            provider_format=ProviderFormat.OPENAI,
            base_url="https://example.com/v1",
            model_id="m",
            api_keys=(),
        )
    )
    router = build_default_router(cache_root=tmp_path)

    inventory = router.call("agent.read_workspace", {})["inventory"]
    by_id = {p["id"]: p for p in inventory["profiles"]}
    assert by_id["with-key"]["api_key_configured"] is True
    assert by_id["without-key"]["api_key_configured"] is False
    assert set(inventory["prompts"]) == {"translation", "glossary", "glossary_review"}


def test_agent_inventory_excludes_placeholder_model_profiles(tmp_path: Path) -> None:
    store = ModelProfileStore.from_cache_root(tmp_path)
    store.create(
        ModelConfig(
            id="valid-profile",
            display_name="Valid",
            provider_format=ProviderFormat.OPENAI,
            base_url="https://example.com/v1",
            model_id="valid-model",
            api_keys=("k",),
        )
    )
    store.create(
        ModelConfig(
            id="bad-profile",
            display_name="DeepSeek 4 Pro",
            provider_format=ProviderFormat.OPENAI,
            base_url="https://example.com/v1",
            model_id="新的值（例如",
            api_keys=("k",),
        )
    )
    router = build_default_router(cache_root=tmp_path)

    inventory = router.call("agent.read_workspace", {})["inventory"]

    assert [profile["id"] for profile in inventory["profiles"]] == ["valid-profile"]
    assert inventory["excluded_profile_count"] == 1


# --- Conversation create/rename validation ----------------------------------


def test_create_conversation_with_title(tmp_path: Path) -> None:
    router = build_default_router(cache_root=tmp_path)
    workspace = router.call("agent.create_conversation", {"title": "Planning"})[
        "workspace"
    ]
    titles = {c["title"] for c in workspace["conversations"]}
    assert "Planning" in titles


def test_create_conversation_rejects_long_title(tmp_path: Path) -> None:
    router = build_default_router(cache_root=tmp_path)
    with pytest.raises(BridgeError):
        router.call("agent.create_conversation", {"title": "x" * 121})


def test_rename_conversation_rejects_empty_and_long_titles(tmp_path: Path) -> None:
    router = build_default_router(cache_root=tmp_path)
    conv_id = router.call("agent.read_workspace", {})["workspace"][
        "active_conversation_id"
    ]
    with pytest.raises(BridgeError):
        router.call(
            "agent.rename_conversation", {"conversation_id": conv_id, "title": "   "}
        )
    with pytest.raises(BridgeError):
        router.call(
            "agent.rename_conversation",
            {"conversation_id": conv_id, "title": "x" * 121},
        )


def test_discard_real_pending_draft(tmp_path: Path) -> None:
    router, _ = _router_with_workflow(
        tmp_path,
        """
        {
          "reply": "Drafting.",
          "draft": {
            "kind": "update_memory",
            "title": "Remember",
            "summary": "save",
            "payload": {"memories": ["keep names"]}
          }
        }
        """,
    )
    draft = router.call("agent.send_message", {"message": "remember"})["workspace"][
        "pending_draft"
    ]

    discarded = router.call("agent.discard_draft", {"draft_id": draft["id"]})[
        "workspace"
    ]
    assert discarded["pending_draft"] is None
    assert discarded["draft_history"][-1]["status"] == "discarded"
    # A second discard of the same id now fails.
    with pytest.raises(BridgeError):
        router.call("agent.discard_draft", {"draft_id": draft["id"]})


# --- Memory payload validation ----------------------------------------------


def test_update_memory_skips_blanks(tmp_path: Path) -> None:
    router = build_default_router(cache_root=tmp_path)
    workspace = router.call("agent.update_memory", {"memories": ["", "  ", "real"]})[
        "workspace"
    ]
    assert workspace["memories"] == ["real"]


def test_update_memory_rejects_non_string_item(tmp_path: Path) -> None:
    router = build_default_router(cache_root=tmp_path)
    with pytest.raises(BridgeError):
        router.call("agent.update_memory", {"memories": [123]})


def test_update_memory_rejects_overlong_entry(tmp_path: Path) -> None:
    router = build_default_router(cache_root=tmp_path)
    with pytest.raises(BridgeError):
        router.call("agent.update_memory", {"memories": ["x" * 601]})


def test_update_memory_rejects_too_many_entries(tmp_path: Path) -> None:
    router = build_default_router(cache_root=tmp_path)
    with pytest.raises(BridgeError):
        router.call(
            "agent.update_memory", {"memories": [str(i) for i in range(31)]}
        )


# --- Recipe payload validation ----------------------------------------------


def test_create_recipe_rejects_long_name_and_description(tmp_path: Path) -> None:
    router = build_default_router(cache_root=tmp_path)
    with pytest.raises(BridgeError):
        router.call("agent.create_recipe", {"name": "x" * 121})
    with pytest.raises(BridgeError):
        router.call(
            "agent.create_recipe", {"name": "ok", "description": "y" * 401}
        )


# --- Patch coercion edge cases ----------------------------------------------


def test_update_workspace_rejects_non_object_slot_maps(tmp_path: Path) -> None:
    router = build_default_router(cache_root=tmp_path)
    with pytest.raises(BridgeError):
        router.call(
            "agent.update_workspace", {"patch": {"stage_model_ids": "nope"}}
        )
    with pytest.raises(BridgeError):
        router.call(
            "agent.update_workspace", {"patch": {"stage_prompt_ids": "nope"}}
        )


def test_update_workspace_rejects_unknown_prompt_slot(tmp_path: Path) -> None:
    router = build_default_router(cache_root=tmp_path)
    with pytest.raises(BridgeError):
        router.call(
            "agent.update_workspace",
            {"patch": {"stage_prompt_ids": {"bogus": None}}},
        )


# --- Prompt-preset draft validation -----------------------------------------


def test_prompt_preset_draft_bad_kind_is_dropped(tmp_path: Path) -> None:
    router, _ = _router_with_workflow(
        tmp_path,
        """
        {
          "reply": "Drafting.",
          "draft": {
            "kind": "create_prompt_preset",
            "title": "Bad",
            "summary": "bad kind",
            "payload": {"kind": "nonsense", "name": "X", "system_prompt": "y"}
          }
        }
        """,
    )
    response = router.call("agent.send_message", {"message": "make a prompt"})
    assert response["workspace"]["pending_draft"] is None


def test_create_recipe_enforces_cap(tmp_path: Path) -> None:
    router = build_default_router(cache_root=tmp_path)
    for index in range(50):
        router.call("agent.create_recipe", {"name": f"R{index}"})
    with pytest.raises(BridgeError):
        router.call("agent.create_recipe", {"name": "overflow"})


def test_prompt_preset_name_without_alphanumerics_still_creates(tmp_path: Path) -> None:
    router, _ = _router_with_workflow(
        tmp_path,
        """
        {
          "reply": "Drafting.",
          "draft": {
            "kind": "create_prompt_preset",
            "title": "Symbols",
            "summary": "symbol-only name",
            "payload": {
              "kind": "translation",
              "name": "###",
              "system_prompt": "Translate.",
              "enabled": true
            }
          }
        }
        """,
    )
    draft = router.call("agent.send_message", {"message": "make a prompt"})[
        "workspace"
    ]["pending_draft"]
    applied = router.call("agent.apply_draft", {"draft_id": draft["id"]})
    assert applied["result"]["preset"]["name"] == "###"
    assert applied["result"]["preset"]["id"].startswith("agent-translation-")


def test_prompt_preset_draft_accepts_stage_and_prompt_aliases(tmp_path: Path) -> None:
    router, _ = _router_with_workflow(
        tmp_path,
        """
        {
          "reply": "Drafting.",
          "draft": {
            "kind": "create_prompt_preset",
            "title": "Alias prompt",
            "summary": "uses model-style aliases",
            "payload": {
              "kind": "term_review",
              "name": "审查别名",
              "prompt": "Review terms carefully.",
              "enabled": true
            }
          }
        }
        """,
    )
    draft = router.call("agent.send_message", {"message": "make a prompt"})[
        "workspace"
    ]["pending_draft"]

    assert draft["kind"] == "create_prompt_preset"
    applied = router.call("agent.apply_draft", {"draft_id": draft["id"]})

    assert applied["result"]["preset"]["kind"] == "glossary_review"
    assert applied["result"]["preset"]["system_prompt"] == "Review terms carefully."


def test_prompt_preset_draft_accepts_chinese_prompt_kind_alias(
    tmp_path: Path,
) -> None:
    router, _ = _router_with_workflow(
        tmp_path,
        """
        {
          "reply": "Drafting.",
          "draft": {
            "kind": "create_prompt_preset",
            "title": "Chinese alias prompt",
            "summary": "uses a Chinese kind alias",
            "payload": {
              "kind": "翻译用 Prompt",
              "name": "现代中文叙事",
              "prompt": "Translate into modern Chinese narration.",
              "enabled": true
            }
          }
        }
        """,
    )
    draft = router.call("agent.send_message", {"message": "make a prompt"})[
        "workspace"
    ]["pending_draft"]

    assert draft["kind"] == "create_prompt_preset"
    applied = router.call("agent.apply_draft", {"draft_id": draft["id"]})

    assert applied["result"]["preset"]["kind"] == "translation"
    assert applied["result"]["preset"]["name"] == "现代中文叙事"


def test_create_recipe_draft_accepts_wrapped_recipe_payload(
    tmp_path: Path,
) -> None:
    router, _ = _router_with_workflow(
        tmp_path,
        """
        {
          "reply": "Drafting.",
          "draft": {
            "kind": "create_recipe",
            "title": "Create recipe",
            "summary": "uses a wrapped recipe payload",
            "payload": {
              "recipe": {
                "recipe_name": "本地测试",
                "description": "Use current stage selections."
              }
            }
          }
        }
        """,
    )
    draft = router.call("agent.send_message", {"message": "make a recipe"})[
        "workspace"
    ]["pending_draft"]

    assert draft["kind"] == "create_recipe"
    applied = router.call("agent.apply_draft", {"draft_id": draft["id"]})

    assert applied["result"]["recipe"]["name"] == "本地测试"


def test_apply_draft_with_corrupted_cache_kind_raises(tmp_path: Path) -> None:
    """A hand-corrupted cache with an unsupported pending-draft kind is rejected."""
    import json

    target = tmp_path / "agent_lab" / "workspace.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "conversations": [
                    {
                        "id": "conv-1",
                        "title": "",
                        "messages": [
                            {"id": "m1", "role": "assistant", "content": "hi"}
                        ],
                        "pending_draft": {
                            "id": "draft-bad",
                            "kind": "run_translation",
                            "title": "bad",
                            "summary": "",
                            "payload": {},
                            "status": "pending",
                        },
                        "draft_history": [],
                    }
                ],
                "active_conversation_id": "conv-1",
            }
        ),
        encoding="utf-8",
    )
    router = build_default_router(cache_root=tmp_path)

    with pytest.raises(BridgeError):
        router.call("agent.apply_draft", {"draft_id": "draft-bad"})


def test_prompt_preset_draft_missing_name_is_dropped(tmp_path: Path) -> None:
    router, _ = _router_with_workflow(
        tmp_path,
        """
        {
          "reply": "Drafting.",
          "draft": {
            "kind": "create_prompt_preset",
            "title": "Bad",
            "summary": "missing name",
            "payload": {"kind": "translation", "system_prompt": "y"}
          }
        }
        """,
    )
    response = router.call("agent.send_message", {"message": "make a prompt"})
    assert response["workspace"]["pending_draft"] is None
