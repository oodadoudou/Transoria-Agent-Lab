"""JSON schemas for the experimental Agent Lab workspace.

These dataclasses intentionally model only orchestration state. They do not
replace stable Translation, Glossary, or Proofreading task caches.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Literal, Mapping
from uuid import uuid4

AgentRole = Literal["user", "assistant", "system"]
DraftStatus = Literal["pending", "applied", "discarded"]

MODEL_SLOTS: tuple[str, ...] = ("translation", "term_extract", "term_review")
PROMPT_SLOTS: tuple[str, ...] = ("translation", "term_extract", "term_review")

MAX_CONVERSATION_MESSAGES = 80

SEED_ASSISTANT_MESSAGE = (
    "Agent Lab is ready. Choose a workflow model, then ask me to draft prompt "
    "or workflow configuration changes."
)


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
class AgentConversation:
    id: str
    title: str
    messages: tuple[AgentMessage, ...]
    pending_draft: AgentActionDraft | None
    draft_history: tuple[AgentActionDraft, ...]
    created_at: str
    updated_at: str

    @classmethod
    def create(
        cls,
        *,
        title: str = "",
        messages: tuple[AgentMessage, ...] = (),
    ) -> "AgentConversation":
        timestamp = now_iso()
        return cls(
            id=new_id("conv"),
            title=title,
            messages=messages,
            pending_draft=None,
            draft_history=(),
            created_at=timestamp,
            updated_at=timestamp,
        )

    @classmethod
    def seeded(cls) -> "AgentConversation":
        return cls.create(
            messages=(AgentMessage.create("assistant", SEED_ASSISTANT_MESSAGE),),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "AgentConversation":
        pending_raw = data.get("pending_draft")
        created_at = str(data.get("created_at") or now_iso())
        return cls(
            id=str(data.get("id") or new_id("conv")),
            title=str(data.get("title") or ""),
            messages=_messages_from(data.get("messages")),
            pending_draft=AgentActionDraft.from_dict(pending_raw)
            if isinstance(pending_raw, Mapping)
            else None,
            draft_history=_drafts_from(data.get("draft_history")),
            created_at=created_at,
            updated_at=str(data.get("updated_at") or created_at),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "messages": [message.to_dict() for message in self.messages],
            "pending_draft": (
                self.pending_draft.to_dict() if self.pending_draft else None
            ),
            "draft_history": [draft.to_dict() for draft in self.draft_history],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def append_message(self, message: AgentMessage) -> "AgentConversation":
        messages = (*self.messages, message)[-MAX_CONVERSATION_MESSAGES:]
        return replace(self, messages=messages, updated_at=now_iso())

    def with_pending_draft(
        self, draft: AgentActionDraft | None
    ) -> "AgentConversation":
        return replace(self, pending_draft=draft, updated_at=now_iso())

    def archive_pending(self, status: DraftStatus) -> "AgentConversation":
        if self.pending_draft is None:
            return self
        archived = self.pending_draft.with_status(status)
        return replace(
            self,
            pending_draft=None,
            draft_history=(*self.draft_history, archived),
            updated_at=now_iso(),
        )

    def with_title(self, title: str) -> "AgentConversation":
        return replace(self, title=title, updated_at=now_iso())


@dataclass(frozen=True)
class AgentWorkspaceState:
    workflow_model_id: str | None = None
    stage_model_ids: Mapping[str, str | None] = field(
        default_factory=lambda: {slot: None for slot in MODEL_SLOTS}
    )
    stage_prompt_ids: Mapping[str, str | None] = field(
        default_factory=lambda: {slot: None for slot in PROMPT_SLOTS}
    )
    memories: tuple[str, ...] = ()
    conversations: tuple[AgentConversation, ...] = ()
    active_conversation_id: str | None = None
    updated_at: str = field(default_factory=now_iso)

    @classmethod
    def empty(cls) -> "AgentWorkspaceState":
        conversation = AgentConversation.seeded()
        return cls(
            conversations=(conversation,),
            active_conversation_id=conversation.id,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "AgentWorkspaceState":
        conversations, active_id = _conversations_from(data)
        return cls(
            workflow_model_id=_optional_str(data.get("workflow_model_id")),
            stage_model_ids=_slot_mapping(data.get("stage_model_ids"), MODEL_SLOTS),
            stage_prompt_ids=_slot_mapping(data.get("stage_prompt_ids"), PROMPT_SLOTS),
            memories=_memories_from(data.get("memories")),
            conversations=conversations,
            active_conversation_id=active_id,
            updated_at=str(data.get("updated_at") or now_iso()),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "workflow_model_id": self.workflow_model_id,
            "stage_model_ids": dict(self.stage_model_ids),
            "stage_prompt_ids": dict(self.stage_prompt_ids),
            "memories": list(self.memories),
            "conversations": [
                conversation.to_dict() for conversation in self.conversations
            ],
            "active_conversation_id": self.active_conversation_id,
            "updated_at": self.updated_at,
        }

    def active(self) -> AgentConversation | None:
        if not self.conversations:
            return None
        for conversation in self.conversations:
            if conversation.id == self.active_conversation_id:
                return conversation
        return self.conversations[0]

    def with_active(self, conversation: AgentConversation) -> "AgentWorkspaceState":
        conversations = tuple(
            conversation if existing.id == conversation.id else existing
            for existing in self.conversations
        )
        return replace(
            self,
            conversations=conversations,
            active_conversation_id=conversation.id,
            updated_at=now_iso(),
        )

    def replace_conversation(
        self, conversation: AgentConversation
    ) -> "AgentWorkspaceState":
        """Replace a conversation by id without changing the active pointer."""
        conversations = tuple(
            conversation if existing.id == conversation.id else existing
            for existing in self.conversations
        )
        return replace(self, conversations=conversations, updated_at=now_iso())

    def add_conversation(
        self, conversation: AgentConversation
    ) -> "AgentWorkspaceState":
        return replace(
            self,
            conversations=(*self.conversations, conversation),
            active_conversation_id=conversation.id,
            updated_at=now_iso(),
        )

    def remove_conversation(self, conversation_id: str) -> "AgentWorkspaceState":
        remaining = tuple(
            conversation
            for conversation in self.conversations
            if conversation.id != conversation_id
        )
        active_id = self.active_conversation_id
        if active_id == conversation_id:
            active_id = (
                max(remaining, key=lambda conv: conv.updated_at).id
                if remaining
                else None
            )
        return replace(
            self,
            conversations=remaining,
            active_conversation_id=active_id,
            updated_at=now_iso(),
        )

    def set_active(self, conversation_id: str) -> "AgentWorkspaceState":
        return replace(
            self,
            active_conversation_id=conversation_id,
            updated_at=now_iso(),
        )

    def with_config(
        self,
        *,
        workflow_model_id: str | None | object = ...,
        stage_model_ids: Mapping[str, str | None] | None = None,
        stage_prompt_ids: Mapping[str, str | None] | None = None,
    ) -> "AgentWorkspaceState":
        return replace(
            self,
            workflow_model_id=(
                self.workflow_model_id
                if workflow_model_id is ...
                else workflow_model_id  # type: ignore[assignment]
            ),
            stage_model_ids=stage_model_ids or self.stage_model_ids,
            stage_prompt_ids=stage_prompt_ids or self.stage_prompt_ids,
            updated_at=now_iso(),
        )

    def with_memories(self, memories: tuple[str, ...]) -> "AgentWorkspaceState":
        return replace(self, memories=memories, updated_at=now_iso())


def _messages_from(value: object) -> tuple[AgentMessage, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        AgentMessage.from_dict(item) for item in value if isinstance(item, Mapping)
    )


def _drafts_from(value: object) -> tuple[AgentActionDraft, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        AgentActionDraft.from_dict(item) for item in value if isinstance(item, Mapping)
    )


def _memories_from(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        item.strip() for item in value if isinstance(item, str) and item.strip()
    )


def _conversations_from(
    data: Mapping[str, object],
) -> tuple[tuple[AgentConversation, ...], str | None]:
    raw = data.get("conversations")
    if isinstance(raw, list):
        conversations = tuple(
            AgentConversation.from_dict(item)
            for item in raw
            if isinstance(item, Mapping)
        )
        active_id = _optional_str(data.get("active_conversation_id"))
    else:
        conversations, active_id = _migrate_flat_conversation(data)
    if conversations and (
        active_id is None
        or all(conversation.id != active_id for conversation in conversations)
    ):
        active_id = conversations[0].id
    return conversations, active_id


def _migrate_flat_conversation(
    data: Mapping[str, object],
) -> tuple[tuple[AgentConversation, ...], str | None]:
    """Wrap a legacy single-thread workspace into one conversation."""
    pending_raw = data.get("pending_draft")
    legacy = AgentConversation(
        id=new_id("conv"),
        title="",
        messages=_messages_from(data.get("messages")),
        pending_draft=AgentActionDraft.from_dict(pending_raw)
        if isinstance(pending_raw, Mapping)
        else None,
        draft_history=_drafts_from(data.get("draft_history")),
        created_at=now_iso(),
        updated_at=now_iso(),
    )
    if legacy.messages or legacy.pending_draft or legacy.draft_history:
        return (legacy,), legacy.id
    return (), None


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
