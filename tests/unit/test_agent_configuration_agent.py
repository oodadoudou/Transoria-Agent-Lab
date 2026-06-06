from __future__ import annotations

from transoria.agent.configuration_agent import (
    AGENT_SYSTEM_PROMPT,
    build_user_prompt,
    parse_agent_response,
)


def test_parse_plain_json_object() -> None:
    reply, draft = parse_agent_response('{"reply": "hello", "draft": null}')
    assert reply == "hello"
    assert draft is None


def test_parse_fenced_json() -> None:
    content = """```json
    {"reply": "fenced", "draft": null}
    ```"""
    reply, draft = parse_agent_response(content)
    assert reply == "fenced"
    assert draft is None


def test_parse_json_embedded_in_prose() -> None:
    content = 'Sure! {"reply": "embedded", "draft": null} done.'
    reply, draft = parse_agent_response(content)
    assert reply == "embedded"
    assert draft is None


def test_parse_non_json_falls_back_to_text() -> None:
    reply, draft = parse_agent_response("just talking, no json here")
    assert reply == "just talking, no json here"
    assert draft is None


def test_parse_draft_without_payload_is_dropped() -> None:
    content = '{"reply": "x", "draft": {"kind": "update_memory", "title": "t"}}'
    reply, draft = parse_agent_response(content)
    assert reply == "x"
    assert draft is None


def test_parse_fenced_json_without_language_tag() -> None:
    content = "```\n{\"reply\": \"plain fence\", \"draft\": null}\n```"
    reply, draft = parse_agent_response(content)
    assert reply == "plain fence"
    assert draft is None


def test_parse_draft_not_an_object_is_ignored() -> None:
    reply, draft = parse_agent_response('{"reply": "x", "draft": "nope"}')
    assert reply == "x"
    assert draft is None


def test_parse_draft_payload_wrong_type_is_ignored() -> None:
    content = '{"reply": "x", "draft": {"kind": "update_memory", "payload": []}}'
    reply, draft = parse_agent_response(content)
    assert reply == "x"
    assert draft is None


def test_parse_empty_reply_falls_back_to_placeholder() -> None:
    reply, draft = parse_agent_response('{"reply": "", "draft": null}')
    assert reply
    assert draft is None


def test_parse_top_level_non_object_returns_text() -> None:
    reply, draft = parse_agent_response("[1, 2, 3]")
    assert reply == "[1, 2, 3]"
    assert draft is None


def test_parse_valid_draft() -> None:
    content = """
    {
      "reply": "drafting",
      "draft": {
        "kind": "update_memory",
        "title": "Remember",
        "summary": "save it",
        "payload": {"memories": ["a"]}
      }
    }
    """
    reply, draft = parse_agent_response(content)
    assert reply == "drafting"
    assert draft is not None
    assert draft.kind == "update_memory"
    assert draft.status == "pending"
    assert draft.payload == {"memories": ["a"]}


def test_system_prompt_documents_phase_b3_configuration_rules() -> None:
    assert "ask a concise follow-up question" in AGENT_SYSTEM_PROMPT
    assert "Do not silently invent" in AGENT_SYSTEM_PROMPT
    assert "Read-only status questions do not need drafts" in AGENT_SYSTEM_PROMPT
    assert "create_model_profile" in AGENT_SYSTEM_PROMPT
    assert "update_model_profile" in AGENT_SYSTEM_PROMPT
    assert "weak model" in AGENT_SYSTEM_PROMPT
    assert "Never save API" in AGENT_SYSTEM_PROMPT


def test_build_user_prompt_includes_response_checklist() -> None:
    prompt = build_user_prompt(
        user_message="配置一个翻译 preset",
        inventory={"profiles": [{"id": "profile-a"}]},
        current_state={"workflow_model_id": "profile-a"},
    )

    assert "Response checklist:" in prompt
    assert "Ask a follow-up instead of inventing missing config." in prompt
    assert "Warn about weak/low-cost models" in prompt
    assert "配置一个翻译 preset" in prompt
