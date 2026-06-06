from __future__ import annotations

from transoria.agent.schemas import (
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
    state = AgentWorkspaceState.empty().with_memories(("keep names",)).add_recipe(
        AgentRecipe.create(name="R", stage_prompt_ids={"translation": "prompt-1"})
    )
    restored = AgentWorkspaceState.from_dict(state.to_dict())
    assert restored.memories == ("keep names",)
    assert restored.recipes[0].name == "R"
    assert restored.recipes[0].stage_prompt_ids["translation"] == "prompt-1"


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


# -------- Project schemas (Phase A scaffolding) --------

from transoria.agent.schemas import (  # noqa: E402
    AgentProject,
    AgentProjectSummary,
    ProjectCheckpoint,
    ProjectDocument,
    ProjectPlan,
    ProjectScan,
    ProjectTaskLinks,
)


def test_project_create_starts_in_draft_with_no_scan_or_plan() -> None:
    project = AgentProject.create(name="N", input_dir="/abs")
    assert project.status == "draft"
    assert project.scan is None
    assert project.plan is None
    assert project.task_links == ProjectTaskLinks()
    assert project.checkpoints == ()


def test_project_round_trips_through_dict() -> None:
    project = AgentProject.create(
        name="N",
        input_dir="/abs",
        source_language="kr",
        target_language="zh",
    )
    restored = AgentProject.from_dict(project.to_dict())
    assert restored == project


def test_project_with_scan_advances_status() -> None:
    project = AgentProject.create(name="N", input_dir="/abs")
    scan = ProjectScan(
        scanned_at="2026-06-06T00:00:00+00:00",
        input_dir="/abs",
        documents=(ProjectDocument(relative_path="a.epub", format="epub", size_bytes=10),),
        document_count=1,
        total_bytes=10,
        epub_count=1,
        txt_count=0,
        truncated=False,
    )
    updated = project.with_scan(scan)
    assert updated.status == "scanned"
    assert updated.scan == scan


def test_project_with_plan_approved_appends_checkpoint() -> None:
    project = AgentProject.create(name="N", input_dir="/abs")
    plan = ProjectPlan(
        stages=("glossary",),
        recipe_snapshot=None,
        auto_chain=True,
        notes="go",
        proposed_at="2026-06-06T00:00:00+00:00",
    )
    checkpoint = ProjectCheckpoint.create(stage="project_plan", status="approved", notes="ok")
    updated = project.with_plan_approved(
        plan=plan,
        task_links=ProjectTaskLinks(glossary_task_id="glossary-x"),
        checkpoint=checkpoint,
    )
    assert updated.status == "plan_approved"
    assert updated.plan == plan
    assert updated.task_links.glossary_task_id == "glossary-x"
    assert updated.checkpoints[-1] is checkpoint


def test_project_summary_round_trip() -> None:
    summary = AgentProjectSummary(
        id="proj-x",
        name="N",
        input_dir="/abs",
        status="draft",
        updated_at="2026-06-06T00:00:00+00:00",
    )
    assert AgentProjectSummary.from_dict(summary.to_dict()) == summary


def test_workspace_upsert_project_summary_inserts_and_updates() -> None:
    state = AgentWorkspaceState.empty()
    s1 = AgentProjectSummary(
        id="proj-1", name="A", input_dir="/abs", status="draft", updated_at="t1"
    )
    s2 = AgentProjectSummary(
        id="proj-1", name="B", input_dir="/abs", status="scanned", updated_at="t2"
    )
    after_first = state.upsert_project_summary(s1, make_active=True)
    assert after_first.active_project_id == "proj-1"
    assert after_first.projects == (s1,)
    after_second = after_first.upsert_project_summary(s2)
    assert len(after_second.projects) == 1
    assert after_second.projects[0].name == "B"
    assert after_second.active_project_id == "proj-1"


def test_workspace_from_dict_drops_unknown_active_project_id() -> None:
    state = AgentWorkspaceState.from_dict(
        {
            "projects": [
                {
                    "id": "proj-1",
                    "name": "A",
                    "input_dir": "/abs",
                    "status": "draft",
                    "updated_at": "t1",
                }
            ],
            "active_project_id": "missing",
        }
    )
    assert state.active_project_id is None


def test_project_from_dict_normalizes_bad_status() -> None:
    project = AgentProject.from_dict(
        {"id": "proj-1", "name": "N", "input_dir": "/abs", "status": "weird"}
    )
    assert project.status == "draft"


def test_project_plan_from_dict_filters_non_string_stages() -> None:
    plan = ProjectPlan.from_dict({"stages": ["a", 7, "b", None, "c"]})
    assert plan.stages == ("a", "b", "c")
