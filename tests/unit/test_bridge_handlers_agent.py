from __future__ import annotations

from pathlib import Path

import pytest

from transoria.agent.configuration_agent import AGENT_SYSTEM_PROMPT
from transoria.bridge import BridgeError, build_default_router
from transoria.bridge.handlers.settings import default_store
from transoria.bridge.task_service import TaskService
from transoria.domain import TaskKind, TaskStatus
from transoria.llm.client import ChatRequest, ChatResponse, LlmRequestError
from transoria.llm.config import ModelConfig, ProviderFormat
from transoria.llm.usage import TokenUsage
from transoria.model_profiles import ModelProfileStore
from transoria.prompts import DEFAULT_TRANSLATION_PRESET_ID
from transoria.runtime.task_record import TaskRecord


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
    [profile] = inventory["profiles"]  # type: ignore[index]
    assert profile["id"] == "profile-workflow"
    assert profile["concurrency_limit"] == 3
    assert profile["rpm_limit"] == 120
    assert profile["tpm_limit"] == 60000
    assert profile["retry_attempts"] == 4


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
    assert captured["draft_kind"] == "start_translation_task"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["input_dir"] == str(input_dir)
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


def test_send_message_calls_api_with_system_prompt_and_inventory(
    tmp_path: Path,
) -> None:
    router, fake = _router_with_workflow(tmp_path, '{"reply":"ok","draft":null}')

    router.call("agent.send_message", {"message": "configure things"})

    assert len(fake.requests) == 1
    request = fake.requests[0]
    assert request.model.id == "profile-workflow"
    assert request.system_prompt == AGENT_SYSTEM_PROMPT
    assert request.stream is False
    # The user prompt carries the inventory and the user's message.
    assert "profile-workflow" in request.user_prompt
    assert "configure things" in request.user_prompt


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
