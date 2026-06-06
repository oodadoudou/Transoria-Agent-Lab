"""JSON schemas for the experimental Agent Lab workspace.

These dataclasses intentionally model only orchestration state. They do not
replace stable Translation, Glossary, or Proofreading task caches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Mapping
from uuid import uuid4

AgentRole = Literal["user", "assistant", "system"]
DraftStatus = Literal["pending", "applied", "discarded"]

MODEL_SLOTS: tuple[str, ...] = ("translation", "term_extract", "term_review")
PROMPT_SLOTS: tuple[str, ...] = ("translation", "term_extract", "term_review")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


@dataclass(frozen=True)
class AgentMessage:
    id: str
    role: AgentRole
    content: str
    created_at: str

    @classmethod
    def create(cls, role: AgentRole, content: str) -> "AgentMessage":
        return cls(
            id=new_id("msg"),
            role=role,
            content=content,
            created_at=now_iso(),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "AgentMessage":
        role = str(data.get("role", "assistant"))
        if role not in ("user", "assistant", "system"):
            role = "assistant"
        return cls(
            id=str(data.get("id") or new_id("msg")),
            role=role,  # type: ignore[arg-type]
            content=str(data.get("content", "")),
            created_at=str(data.get("created_at") or now_iso()),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class AgentActionDraft:
    id: str
    kind: str
    title: str
    summary: str
    payload: Mapping[str, object]
    status: DraftStatus
    created_at: str

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        title: str,
        summary: str,
        payload: Mapping[str, object],
    ) -> "AgentActionDraft":
        return cls(
            id=new_id("draft"),
            kind=kind,
            title=title,
            summary=summary,
            payload=dict(payload),
            status="pending",
            created_at=now_iso(),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "AgentActionDraft":
        payload = data.get("payload")
        status = str(data.get("status", "pending"))
        if status not in ("pending", "applied", "discarded"):
            status = "pending"
        return cls(
            id=str(data.get("id") or new_id("draft")),
            kind=str(data.get("kind", "")),
            title=str(data.get("title", "")),
            summary=str(data.get("summary", "")),
            payload=dict(payload) if isinstance(payload, Mapping) else {},
            status=status,  # type: ignore[arg-type]
            created_at=str(data.get("created_at") or now_iso()),
        )

    def with_status(self, status: DraftStatus) -> "AgentActionDraft":
        return AgentActionDraft(
            id=self.id,
            kind=self.kind,
            title=self.title,
            summary=self.summary,
            payload=dict(self.payload),
            status=status,
            created_at=self.created_at,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "summary": self.summary,
            "payload": dict(self.payload),
            "status": self.status,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class AgentWorkspaceState:
    workflow_model_id: str | None = None
    stage_model_ids: Mapping[str, str | None] = field(
        default_factory=lambda: {slot: None for slot in MODEL_SLOTS}
    )
    stage_prompt_ids: Mapping[str, str | None] = field(
        default_factory=lambda: {slot: None for slot in PROMPT_SLOTS}
    )
    messages: tuple[AgentMessage, ...] = ()
    memories: tuple[str, ...] = ()
    pending_draft: AgentActionDraft | None = None
    draft_history: tuple[AgentActionDraft, ...] = ()
    updated_at: str = field(default_factory=now_iso)

    @classmethod
    def empty(cls) -> "AgentWorkspaceState":
        return cls(
            messages=(
                AgentMessage.create(
                    "assistant",
                    "Agent Lab is ready. Choose a workflow model, then ask me to draft prompt or workflow configuration changes.",
                ),
            )
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "AgentWorkspaceState":
        stage_model_ids = _slot_mapping(data.get("stage_model_ids"), MODEL_SLOTS)
        stage_prompt_ids = _slot_mapping(data.get("stage_prompt_ids"), PROMPT_SLOTS)
        messages_raw = data.get("messages")
        drafts_raw = data.get("draft_history")
        pending_raw = data.get("pending_draft")
        memories_raw = data.get("memories")
        return cls(
            workflow_model_id=_optional_str(data.get("workflow_model_id")),
            stage_model_ids=stage_model_ids,
            stage_prompt_ids=stage_prompt_ids,
            messages=tuple(
                AgentMessage.from_dict(item)
                for item in messages_raw
                if isinstance(item, Mapping)
            )
            if isinstance(messages_raw, list)
            else (),
            memories=tuple(
                str(item).strip()
                for item in memories_raw
                if isinstance(item, str) and item.strip()
            )
            if isinstance(memories_raw, list)
            else (),
            pending_draft=AgentActionDraft.from_dict(pending_raw)
            if isinstance(pending_raw, Mapping)
            else None,
            draft_history=tuple(
                AgentActionDraft.from_dict(item)
                for item in drafts_raw
                if isinstance(item, Mapping)
            )
            if isinstance(drafts_raw, list)
            else (),
            updated_at=str(data.get("updated_at") or now_iso()),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "workflow_model_id": self.workflow_model_id,
            "stage_model_ids": dict(self.stage_model_ids),
            "stage_prompt_ids": dict(self.stage_prompt_ids),
            "messages": [message.to_dict() for message in self.messages],
            "memories": list(self.memories),
            "pending_draft": (
                self.pending_draft.to_dict() if self.pending_draft else None
            ),
            "draft_history": [draft.to_dict() for draft in self.draft_history],
            "updated_at": self.updated_at,
        }


def with_updates(
    state: AgentWorkspaceState,
    *,
    workflow_model_id: str | None | object = ...,
    stage_model_ids: Mapping[str, str | None] | None = None,
    stage_prompt_ids: Mapping[str, str | None] | None = None,
    messages: tuple[AgentMessage, ...] | None = None,
    memories: tuple[str, ...] | None = None,
    pending_draft: AgentActionDraft | None | object = ...,
    draft_history: tuple[AgentActionDraft, ...] | None = None,
) -> AgentWorkspaceState:
    return AgentWorkspaceState(
        workflow_model_id=(
            state.workflow_model_id
            if workflow_model_id is ...
            else workflow_model_id  # type: ignore[assignment]
        ),
        stage_model_ids=stage_model_ids or state.stage_model_ids,
        stage_prompt_ids=stage_prompt_ids or state.stage_prompt_ids,
        messages=messages if messages is not None else state.messages,
        memories=memories if memories is not None else state.memories,
        pending_draft=(
            state.pending_draft
            if pending_draft is ...
            else pending_draft  # type: ignore[assignment]
        ),
        draft_history=(
            draft_history if draft_history is not None else state.draft_history
        ),
        updated_at=now_iso(),
    )


def _optional_str(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _slot_mapping(value: object, slots: tuple[str, ...]) -> dict[str, str | None]:
    result: dict[str, str | None] = {slot: None for slot in slots}
    if not isinstance(value, Mapping):
        return result
    for slot in slots:
        result[slot] = _optional_str(value.get(slot))
    return result
