"""Status response entry points for Agent Lab chat."""

from __future__ import annotations

from collections.abc import Mapping

from transoria.bridge.handlers.agent_task_intents import (
    extract_glossary_review_task_id,
    extract_glossary_task_id,
)
from transoria.bridge.handlers.agent_task_status import (
    direct_stage_status_response as build_direct_stage_status_response,
)
from transoria.bridge.task_service import TaskService


def direct_stage_status_response(
    *,
    user_message: str,
    current_state: Mapping[str, object],
    task_service: TaskService,
) -> tuple[str, None] | None:
    return build_direct_stage_status_response(
        user_message=user_message,
        current_state=current_state,
        task_service=task_service,
        extract_glossary_task_id=extract_glossary_task_id,
        extract_glossary_review_task_id=extract_glossary_review_task_id,
    )
