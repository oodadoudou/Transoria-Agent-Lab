from __future__ import annotations

from transoria.agent.schemas import (
    AgentActiveTask,
    AgentActionDraft,
    AgentConversation,
    AgentMessage,
    AgentRecipe,
    AgentWorkspaceState,
)


def test_message_from_dict_normalizes_unknown_role() -> None:
    message = AgentMessage.from_dict({"role": "robot", "content": "hi"})
    assert message.role == "assistant"


def test_draft_from_dict_normalizes_unknown_status() -> None:
    draft = AgentActionDraft.from_dict(
        {"kind": "update_memory", "status": "weird", "payload": {}}
    )
    assert draft.status == "pending"


def test_conversation_archive_pending_without_draft_is_noop() -> None:
    conversation = AgentConversation.seeded()
    assert conversation.pending_draft is None
    assert conversation.archive_pending("applied") is conversation


def test_conversation_append_message_trims_history() -> None:
    conversation = AgentConversation.create()
    for index in range(100):
        conversation = conversation.append_message(
            AgentMessage.create("user", f"m{index}")
        )
    assert len(conversation.messages) == 80
    assert conversation.messages[-1].content == "m99"


def test_recipe_with_updates_partial_keeps_other_fields() -> None:
    recipe = AgentRecipe.create(
        name="A",
        description="desc",
        stage_model_ids={"translation": "p1"},
    )
    updated = recipe.with_updates(name="B")
    assert updated.name == "B"
    assert updated.description == "desc"
    assert updated.stage_model_ids["translation"] == "p1"


def test_recipe_create_normalizes_all_slots() -> None:
    recipe = AgentRecipe.create(name="A", stage_model_ids={"translation": "p1"})
    assert set(recipe.stage_model_ids) == {"translation", "term_extract", "term_review"}
    assert recipe.stage_model_ids["term_extract"] is None


def test_workspace_active_falls_back_to_first_when_pointer_stale() -> None:
    state = AgentWorkspaceState.empty()
    stale = state.set_active("does-not-exist")
    assert stale.active() is state.conversations[0]


def test_workspace_round_trips_recipes_and_memories() -> None:
    base = AgentWorkspaceState.empty()
    conversation = base.active()
    assert conversation is not None
    active_task = AgentActiveTask.create(
        task_id="translation-1",
        kind="translation",
        conversation_id=conversation.id,
        started_at="2026-01-01T00:00:00+00:00",
    )
    state = (
        base
        .with_memories(("keep names",))
        .add_recipe(
            AgentRecipe.create(name="R", stage_prompt_ids={"translation": "prompt-1"})
        )
        .with_active_task(active_task)
    )
    restored = AgentWorkspaceState.from_dict(state.to_dict())
    assert restored.memories == ("keep names",)
    assert restored.recipes[0].name == "R"
    assert restored.recipes[0].stage_prompt_ids["translation"] == "prompt-1"
    assert restored.active_task is not None
    assert restored.active_task.task_id == "translation-1"


def test_empty_workspace_has_single_seeded_conversation() -> None:
    state = AgentWorkspaceState.empty()
    assert len(state.conversations) == 1
    assert state.active_conversation_id == state.conversations[0].id
    assert state.conversations[0].messages[0].role == "assistant"


def test_bare_workspace_has_no_active_conversation() -> None:
    assert AgentWorkspaceState().active() is None


def test_conversation_from_dict_with_non_list_messages() -> None:
    conversation = AgentConversation.from_dict({"id": "c1", "messages": "oops"})
    assert conversation.messages == ()


def test_from_dict_repairs_stale_active_pointer_in_stored_conversations() -> None:
    state = AgentWorkspaceState.from_dict(
        {
            "conversations": [
                {"id": "c1", "title": "first", "messages": []},
                {"id": "c2", "title": "second", "messages": []},
            ],
            "active_conversation_id": "missing",
        }
    )
    assert state.active_conversation_id == "c1"


def test_from_dict_without_conversations_or_messages_is_empty() -> None:
    state = AgentWorkspaceState.from_dict({"workflow_model_id": None})
    assert state.conversations == ()
    assert state.active_conversation_id is None


def test_invalid_active_task_is_ignored_on_load() -> None:
    state = AgentWorkspaceState.from_dict(
        {
            "active_task": {
                "task_id": "t",
                "kind": "not-a-task-kind",
                "conversation_id": "c",
            }
        }
    )
    assert state.active_task is None
