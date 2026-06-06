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

ProjectStatus = Literal["draft", "scanned", "plan_approved"]
ProjectTaskSlot = Literal["glossary", "glossary_review", "translation"]
PROJECT_TASK_SLOTS: tuple[str, ...] = ("glossary", "glossary_review", "translation")
DocumentFormatName = Literal["epub", "txt"]
ProjectCheckpointStage = Literal["project_plan"]
ProjectCheckpointStatus = Literal["approved"]

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
class AgentRecipe:
    """A named, savable bundle of stage model + prompt choices."""

    id: str
    name: str
    description: str
    stage_model_ids: Mapping[str, str | None]
    stage_prompt_ids: Mapping[str, str | None]
    created_at: str
    updated_at: str

    @classmethod
    def create(
        cls,
        *,
        name: str,
        description: str = "",
        stage_model_ids: Mapping[str, str | None] | None = None,
        stage_prompt_ids: Mapping[str, str | None] | None = None,
    ) -> "AgentRecipe":
        timestamp = now_iso()
        return cls(
            id=new_id("recipe"),
            name=name,
            description=description,
            stage_model_ids=_full_slot_mapping(stage_model_ids, MODEL_SLOTS),
            stage_prompt_ids=_full_slot_mapping(stage_prompt_ids, PROMPT_SLOTS),
            created_at=timestamp,
            updated_at=timestamp,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "AgentRecipe":
        created_at = str(data.get("created_at") or now_iso())
        return cls(
            id=str(data.get("id") or new_id("recipe")),
            name=str(data.get("name") or ""),
            description=str(data.get("description") or ""),
            stage_model_ids=_slot_mapping(data.get("stage_model_ids"), MODEL_SLOTS),
            stage_prompt_ids=_slot_mapping(data.get("stage_prompt_ids"), PROMPT_SLOTS),
            created_at=created_at,
            updated_at=str(data.get("updated_at") or created_at),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "stage_model_ids": dict(self.stage_model_ids),
            "stage_prompt_ids": dict(self.stage_prompt_ids),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def with_updates(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        stage_model_ids: Mapping[str, str | None] | None = None,
        stage_prompt_ids: Mapping[str, str | None] | None = None,
    ) -> "AgentRecipe":
        return replace(
            self,
            name=name if name is not None else self.name,
            description=description if description is not None else self.description,
            stage_model_ids=_full_slot_mapping(stage_model_ids, MODEL_SLOTS)
            if stage_model_ids is not None
            else self.stage_model_ids,
            stage_prompt_ids=_full_slot_mapping(stage_prompt_ids, PROMPT_SLOTS)
            if stage_prompt_ids is not None
            else self.stage_prompt_ids,
            updated_at=now_iso(),
        )


@dataclass(frozen=True)
class ProjectDocument:
    relative_path: str
    format: DocumentFormatName
    size_bytes: int

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ProjectDocument":
        fmt = str(data.get("format") or "")
        if fmt not in ("epub", "txt"):
            fmt = "txt"
        size_raw = data.get("size_bytes", 0)
        try:
            size = int(size_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            size = 0
        return cls(
            relative_path=str(data.get("relative_path") or ""),
            format=fmt,  # type: ignore[arg-type]
            size_bytes=max(0, size),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "format": self.format,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class ProjectScan:
    scanned_at: str
    input_dir: str
    documents: tuple[ProjectDocument, ...]
    document_count: int
    total_bytes: int
    epub_count: int
    txt_count: int
    truncated: bool

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ProjectScan":
        docs = data.get("documents")
        documents = (
            tuple(
                ProjectDocument.from_dict(item)
                for item in docs
                if isinstance(item, Mapping)
            )
            if isinstance(docs, list)
            else ()
        )
        return cls(
            scanned_at=str(data.get("scanned_at") or now_iso()),
            input_dir=str(data.get("input_dir") or ""),
            documents=documents,
            document_count=_safe_int(data.get("document_count"), len(documents)),
            total_bytes=_safe_int(data.get("total_bytes"), 0),
            epub_count=_safe_int(data.get("epub_count"), 0),
            txt_count=_safe_int(data.get("txt_count"), 0),
            truncated=bool(data.get("truncated", False)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "scanned_at": self.scanned_at,
            "input_dir": self.input_dir,
            "documents": [doc.to_dict() for doc in self.documents],
            "document_count": self.document_count,
            "total_bytes": self.total_bytes,
            "epub_count": self.epub_count,
            "txt_count": self.txt_count,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class ProjectTaskLinks:
    """Existing task_id references for an Agent Lab project.

    Each slot points at a task under <cache_root>/tasks/<task_id>/. The
    project does NOT own these tasks; the user may swap to any earlier
    task_id of the matching kind at plan-approval time.
    """

    glossary_task_id: str | None = None
    glossary_review_task_id: str | None = None
    translation_task_id: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ProjectTaskLinks":
        return cls(
            glossary_task_id=_optional_str(data.get("glossary_task_id")),
            glossary_review_task_id=_optional_str(data.get("glossary_review_task_id")),
            translation_task_id=_optional_str(data.get("translation_task_id")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "glossary_task_id": self.glossary_task_id,
            "glossary_review_task_id": self.glossary_review_task_id,
            "translation_task_id": self.translation_task_id,
        }


@dataclass(frozen=True)
class ProjectCheckpoint:
    id: str
    stage: ProjectCheckpointStage
    status: ProjectCheckpointStatus
    notes: str
    decided_at: str

    @classmethod
    def create(
        cls,
        *,
        stage: ProjectCheckpointStage,
        status: ProjectCheckpointStatus,
        notes: str = "",
    ) -> "ProjectCheckpoint":
        return cls(
            id=new_id("ckpt"),
            stage=stage,
            status=status,
            notes=notes,
            decided_at=now_iso(),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ProjectCheckpoint":
        stage = str(data.get("stage") or "project_plan")
        if stage not in ("project_plan",):
            stage = "project_plan"
        status = str(data.get("status") or "approved")
        if status not in ("approved",):
            status = "approved"
        return cls(
            id=str(data.get("id") or new_id("ckpt")),
            stage=stage,  # type: ignore[arg-type]
            status=status,  # type: ignore[arg-type]
            notes=str(data.get("notes") or ""),
            decided_at=str(data.get("decided_at") or now_iso()),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "stage": self.stage,
            "status": self.status,
            "notes": self.notes,
            "decided_at": self.decided_at,
        }


@dataclass(frozen=True)
class ProjectPlan:
    stages: tuple[str, ...]
    recipe_snapshot: AgentRecipe | None
    auto_chain: bool
    notes: str
    proposed_at: str

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ProjectPlan":
        stages_raw = data.get("stages")
        stages = (
            tuple(str(item) for item in stages_raw if isinstance(item, str))
            if isinstance(stages_raw, list)
            else ()
        )
        recipe_raw = data.get("recipe_snapshot")
        recipe = (
            AgentRecipe.from_dict(recipe_raw)
            if isinstance(recipe_raw, Mapping)
            else None
        )
        return cls(
            stages=stages,
            recipe_snapshot=recipe,
            auto_chain=bool(data.get("auto_chain", False)),
            notes=str(data.get("notes") or ""),
            proposed_at=str(data.get("proposed_at") or now_iso()),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stages": list(self.stages),
            "recipe_snapshot": (
                self.recipe_snapshot.to_dict() if self.recipe_snapshot else None
            ),
            "auto_chain": self.auto_chain,
            "notes": self.notes,
            "proposed_at": self.proposed_at,
        }


@dataclass(frozen=True)
class AgentProject:
    id: str
    name: str
    input_dir: str
    source_language: str | None
    target_language: str | None
    status: ProjectStatus
    scan: ProjectScan | None
    plan: ProjectPlan | None
    task_links: ProjectTaskLinks
    checkpoints: tuple[ProjectCheckpoint, ...]
    created_at: str
    updated_at: str

    @classmethod
    def create(
        cls,
        *,
        name: str,
        input_dir: str,
        source_language: str | None = None,
        target_language: str | None = None,
    ) -> "AgentProject":
        timestamp = now_iso()
        return cls(
            id=new_id("proj"),
            name=name,
            input_dir=input_dir,
            source_language=source_language,
            target_language=target_language,
            status="draft",
            scan=None,
            plan=None,
            task_links=ProjectTaskLinks(),
            checkpoints=(),
            created_at=timestamp,
            updated_at=timestamp,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "AgentProject":
        status = str(data.get("status") or "draft")
        if status not in ("draft", "scanned", "plan_approved"):
            status = "draft"
        scan_raw = data.get("scan")
        plan_raw = data.get("plan")
        links_raw = data.get("task_links")
        checkpoints_raw = data.get("checkpoints")
        created_at = str(data.get("created_at") or now_iso())
        return cls(
            id=str(data.get("id") or new_id("proj")),
            name=str(data.get("name") or ""),
            input_dir=str(data.get("input_dir") or ""),
            source_language=_optional_str(data.get("source_language")),
            target_language=_optional_str(data.get("target_language")),
            status=status,  # type: ignore[arg-type]
            scan=ProjectScan.from_dict(scan_raw) if isinstance(scan_raw, Mapping) else None,
            plan=ProjectPlan.from_dict(plan_raw) if isinstance(plan_raw, Mapping) else None,
            task_links=(
                ProjectTaskLinks.from_dict(links_raw)
                if isinstance(links_raw, Mapping)
                else ProjectTaskLinks()
            ),
            checkpoints=(
                tuple(
                    ProjectCheckpoint.from_dict(item)
                    for item in checkpoints_raw
                    if isinstance(item, Mapping)
                )
                if isinstance(checkpoints_raw, list)
                else ()
            ),
            created_at=created_at,
            updated_at=str(data.get("updated_at") or created_at),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "input_dir": self.input_dir,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "status": self.status,
            "scan": self.scan.to_dict() if self.scan else None,
            "plan": self.plan.to_dict() if self.plan else None,
            "task_links": self.task_links.to_dict(),
            "checkpoints": [ckpt.to_dict() for ckpt in self.checkpoints],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def summary(self) -> "AgentProjectSummary":
        return AgentProjectSummary(
            id=self.id,
            name=self.name,
            input_dir=self.input_dir,
            status=self.status,
            updated_at=self.updated_at,
        )

    def with_scan(self, scan: ProjectScan) -> "AgentProject":
        return replace(self, scan=scan, status="scanned", updated_at=now_iso())

    def with_plan_approved(
        self,
        *,
        plan: ProjectPlan,
        task_links: ProjectTaskLinks,
        checkpoint: ProjectCheckpoint,
    ) -> "AgentProject":
        return replace(
            self,
            plan=plan,
            task_links=task_links,
            checkpoints=(*self.checkpoints, checkpoint),
            status="plan_approved",
            updated_at=now_iso(),
        )


@dataclass(frozen=True)
class AgentProjectSummary:
    id: str
    name: str
    input_dir: str
    status: str
    updated_at: str

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "AgentProjectSummary":
        return cls(
            id=str(data.get("id") or ""),
            name=str(data.get("name") or ""),
            input_dir=str(data.get("input_dir") or ""),
            status=str(data.get("status") or "draft"),
            updated_at=str(data.get("updated_at") or now_iso()),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "input_dir": self.input_dir,
            "status": self.status,
            "updated_at": self.updated_at,
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
    memories: tuple[str, ...] = ()
    recipes: tuple[AgentRecipe, ...] = ()
    conversations: tuple[AgentConversation, ...] = ()
    active_conversation_id: str | None = None
    projects: tuple[AgentProjectSummary, ...] = ()
    active_project_id: str | None = None
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
        projects = _project_summaries_from(data.get("projects"))
        active_project_id = _optional_str(data.get("active_project_id"))
        if projects and (
            active_project_id is None
            or all(project.id != active_project_id for project in projects)
        ):
            active_project_id = None
        return cls(
            workflow_model_id=_optional_str(data.get("workflow_model_id")),
            stage_model_ids=_slot_mapping(data.get("stage_model_ids"), MODEL_SLOTS),
            stage_prompt_ids=_slot_mapping(data.get("stage_prompt_ids"), PROMPT_SLOTS),
            memories=_memories_from(data.get("memories")),
            recipes=_recipes_from(data.get("recipes")),
            conversations=conversations,
            active_conversation_id=active_id,
            projects=projects,
            active_project_id=active_project_id,
            updated_at=str(data.get("updated_at") or now_iso()),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "workflow_model_id": self.workflow_model_id,
            "stage_model_ids": dict(self.stage_model_ids),
            "stage_prompt_ids": dict(self.stage_prompt_ids),
            "memories": list(self.memories),
            "recipes": [recipe.to_dict() for recipe in self.recipes],
            "conversations": [
                conversation.to_dict() for conversation in self.conversations
            ],
            "active_conversation_id": self.active_conversation_id,
            "projects": [project.to_dict() for project in self.projects],
            "active_project_id": self.active_project_id,
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

    def get_recipe(self, recipe_id: str) -> AgentRecipe | None:
        for recipe in self.recipes:
            if recipe.id == recipe_id:
                return recipe
        return None

    def add_recipe(self, recipe: AgentRecipe) -> "AgentWorkspaceState":
        return replace(
            self, recipes=(*self.recipes, recipe), updated_at=now_iso()
        )

    def replace_recipe(self, recipe: AgentRecipe) -> "AgentWorkspaceState":
        recipes = tuple(
            recipe if existing.id == recipe.id else existing
            for existing in self.recipes
        )
        return replace(self, recipes=recipes, updated_at=now_iso())

    def remove_recipe(self, recipe_id: str) -> "AgentWorkspaceState":
        recipes = tuple(
            recipe for recipe in self.recipes if recipe.id != recipe_id
        )
        return replace(self, recipes=recipes, updated_at=now_iso())

    def upsert_project_summary(
        self, summary: AgentProjectSummary, *, make_active: bool = False
    ) -> "AgentWorkspaceState":
        if any(existing.id == summary.id for existing in self.projects):
            projects = tuple(
                summary if existing.id == summary.id else existing
                for existing in self.projects
            )
        else:
            projects = (*self.projects, summary)
        active_id = (
            summary.id if make_active else self.active_project_id
        )
        return replace(
            self,
            projects=projects,
            active_project_id=active_id,
            updated_at=now_iso(),
        )


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


def _recipes_from(value: object) -> tuple[AgentRecipe, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        AgentRecipe.from_dict(item) for item in value if isinstance(item, Mapping)
    )


def _full_slot_mapping(
    value: Mapping[str, str | None] | None, slots: tuple[str, ...]
) -> dict[str, str | None]:
    """Normalize a (possibly partial) slot map into one with every slot present."""
    result: dict[str, str | None] = {slot: None for slot in slots}
    if isinstance(value, Mapping):
        for slot in slots:
            result[slot] = _optional_str(value.get(slot))
    return result


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


def _safe_int(value: object, fallback: int) -> int:
    try:
        return max(0, int(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback


def _project_summaries_from(value: object) -> tuple["AgentProjectSummary", ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        AgentProjectSummary.from_dict(item)
        for item in value
        if isinstance(item, Mapping)
    )


def _slot_mapping(value: object, slots: tuple[str, ...]) -> dict[str, str | None]:
    result: dict[str, str | None] = {slot: None for slot in slots}
    if not isinstance(value, Mapping):
        return result
    for slot in slots:
        result[slot] = _optional_str(value.get(slot))
    return result
