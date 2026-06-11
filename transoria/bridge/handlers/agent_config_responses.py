"""Direct configuration response builders for Agent Lab chat."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from transoria.agent.schemas import AgentActionDraft
from transoria.bridge.handlers.agent_action_registry import compound_raw_actions
from transoria.bridge.handlers.agent_draft_salvage import salvage_compound_draft
from transoria.model_profiles import ModelProfileStore

RequestPredicate = Callable[[str], bool]


def direct_compound_config_response(
    *,
    user_message: str,
    current_state: Mapping[str, object],
    profile_store: ModelProfileStore,
    cache_root: Path,
    excluded_request: RequestPredicate | None = None,
) -> tuple[str, AgentActionDraft | None] | None:
    if not looks_like_direct_compound_config_request(
        user_message,
        excluded_request=excluded_request,
    ):
        return None
    draft = salvage_compound_draft(
        AgentActionDraft.create(
            kind="compound_config_update",
            title="复合配置修改",
            summary="根据你的请求准备多个配置修改，并在确认后一次性应用。",
            payload={"actions": []},
        ),
        text=user_message,
        current_state=current_state,
        profile_store=profile_store,
        cache_root=cache_root,
    )
    if draft is None:
        return None
    actions = compound_raw_actions(draft.payload)
    count = len(actions) if isinstance(actions, list) else 0
    return (
        (
            f"我已准备好包含 {count} 项修改的配置草案。"
            "请检查每项修改，确认后我才会写入配置。"
        ),
        draft,
    )


def looks_like_direct_compound_config_request(
    text: str,
    *,
    excluded_request: RequestPredicate | None = None,
) -> bool:
    if excluded_request is not None and excluded_request(text):
        return False
    normalized = text.lower()
    return any(
        marker in normalized
        for marker in (
            "并发",
            "concurrency",
            "重命名",
            "改名",
            "保存当前",
            "保存成",
            "保存为",
            "存成",
            "存为",
        )
    )
