import { useEffect, useRef, useState } from "react";
import type {
  AgentRecipe,
  AgentTaskKind,
  AgentWorkspace,
  ThinkingLevel,
} from "@/bridge/types";
import { Pill } from "@/components/Pill";
import { useMessages } from "@/locales";
import { useRuntimeStore, usePollRunSnapshot } from "@/store/useRuntimeStore";
import { useTaskStore } from "@/store/useTaskStore";
import { ChatComposer } from "./components/ChatComposer";
import { ChatMessageList } from "./components/ChatMessageList";
import {
  DraftConfirmationChecklist,
  DraftPreview,
} from "./components/DraftPreview";
import { HistoryPane } from "./components/HistoryPane";
import styles from "./ChatPage.module.css";
import { useAgentWorkspace } from "./useAgentWorkspace";

export function ChatPage() {
  const messages = useMessages();
  const t = messages.agentLab;
  const navigate = useTaskStore((state) => state.navigate);
  const translationHeader = useRuntimeStore((state) => state.translation.header);
  const glossaryHeader = useRuntimeStore((state) => state.glossary.header);
  const glossaryReviewHeader = useRuntimeStore(
    (state) => state.glossary_review.header,
  );
  const [input, setInput] = useState("");
  const [editingConvId, setEditingConvId] = useState<string | null>(null);
  const [editingConvTitle, setEditingConvTitle] = useState("");
  const [newMemory, setNewMemory] = useState("");
  const [editingMemoryIndex, setEditingMemoryIndex] = useState<number | null>(
    null,
  );
  const [editingMemoryText, setEditingMemoryText] = useState("");
  const [historyCollapsed, setHistoryCollapsed] = useState(false);
  const [modelOpen, setModelOpen] = useState(false);
  const [recipeOpen, setRecipeOpen] = useState(false);
  const [reasoningOpen, setReasoningOpen] = useState(false);
  const [processExpanded, setProcessExpanded] = useState(false);
  const [adjustingDraft, setAdjustingDraft] = useState(false);
  const [draftAdjustment, setDraftAdjustment] = useState("");
  const [draftConfirmed, setDraftConfirmed] = useState(false);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const messagesRef = useRef<HTMLDivElement | null>(null);
  const [expandedMessages, setExpandedMessages] = useState<ReadonlySet<string>>(
    () => new Set(),
  );
  usePollRunSnapshot("translation");
  usePollRunSnapshot("glossary");
  usePollRunSnapshot("glossary_review");
  const {
    workspace,
    inventory,
    busy,
    error,
    setError,
    sendMessage: sendWorkspaceMessage,
    applyDraft: applyWorkspaceDraft,
    discardDraft: discardWorkspaceDraft,
    reviseDraft,
    createConversation: createWorkspaceConversation,
    switchConversation: switchWorkspaceConversation,
    renameConversation,
    deleteConversation: deleteWorkspaceConversation,
    updateMemory,
    deleteMemory: deleteWorkspaceMemory,
    applyRecipe,
    updateWorkspace,
  } = useAgentWorkspace({
    loadFailedText: t.loadFailed,
    saveFailedText: t.saveFailed,
  });

  useEffect(() => {
    const container = messagesRef.current;
    if (!container) return;
    container.scrollTo({
      top: container.scrollHeight,
      behavior: "smooth",
    });
  }, [busy, workspace?.messages.length, workspace?.pending_draft?.id]);

  useEffect(() => {
    setDraftConfirmed(false);
  }, [workspace?.pending_draft?.id]);

  const sendText = async (rawText: string, restoreOnError = false) => {
    const text = rawText.trim();
    if (!text) return;
    if (restoreOnError) {
      setInput("");
    }
    const ok = await sendWorkspaceMessage(text);
    if (!ok && restoreOnError) {
      setInput(text);
    }
  };

  const sendMessage = async () => {
    await sendText(input, true);
  };

  const resendMessage = (content: string) => {
    void sendText(content);
  };

  const copyMessage = async (id: string, content: string) => {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(content);
      } else {
        fallbackCopy(content);
      }
      setCopiedMessageId(id);
      window.setTimeout(() => {
        setCopiedMessageId((current) => (current === id ? null : current));
      }, 1400);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const applyDraft = () => {
    if (!workspace?.pending_draft || !draftConfirmed) return;
    const draftId = workspace.pending_draft.id;
    setAdjustingDraft(false);
    setDraftAdjustment("");
    setDraftConfirmed(false);
    void applyWorkspaceDraft(draftId);
  };

  const discardDraft = () => {
    if (!workspace?.pending_draft) return;
    const draftId = workspace.pending_draft.id;
    setAdjustingDraft(false);
    setDraftAdjustment("");
    void discardWorkspaceDraft(draftId);
  };

  const startAdjustDraft = () => {
    if (!workspace?.pending_draft) return;
    setAdjustingDraft(true);
    setDraftAdjustment("");
  };

  const cancelDraftAdjustment = () => {
    setAdjustingDraft(false);
    setDraftAdjustment("");
  };

  const submitDraftAdjustment = async () => {
    const adjustment = draftAdjustment.trim();
    if (!workspace?.pending_draft || !adjustment) return;
    const draftId = workspace.pending_draft.id;
    setAdjustingDraft(false);
    setDraftAdjustment("");
    await reviseDraft(draftId, adjustment);
  };

  const createConversation = () => createWorkspaceConversation();

  const switchConversation = (id: string) => {
    if (id === workspace?.active_conversation_id) return;
    void switchWorkspaceConversation(id);
  };

  const deleteConversation = (id: string) =>
    void deleteWorkspaceConversation(id);

  const startRename = (id: string, title: string) => {
    setEditingConvId(id);
    setEditingConvTitle(title);
  };

  const commitRename = async () => {
    const id = editingConvId;
    const title = editingConvTitle.trim();
    setEditingConvId(null);
    if (!id || !title) return;
    await renameConversation(id, title);
  };

  const addMemory = async () => {
    const text = newMemory.trim();
    if (!text) return;
    setNewMemory("");
    await updateMemory([...(workspace?.memories ?? []), text]);
  };

  const deleteMemory = (memory: string) =>
    void deleteWorkspaceMemory(memory);

  const startEditMemory = (index: number, memory: string) => {
    setEditingMemoryIndex(index);
    setEditingMemoryText(memory);
  };

  const commitMemoryEdit = async () => {
    const index = editingMemoryIndex;
    const text = editingMemoryText.trim();
    setEditingMemoryIndex(null);
    if (index === null) return;
    const current = workspace?.memories ?? [];
    const next = text
      ? current.map((memory, i) => (i === index ? text : memory))
      : current.filter((_, i) => i !== index);
    await updateMemory(next);
  };

  const switchRecipe = (id: string) => {
    if (!id) return;
    setRecipeOpen(false);
    void applyRecipe(id);
  };

  const updateWorkflowModel = (value: string) => {
    setModelOpen(false);
    void updateWorkspace({ workflow_model_id: value || null });
  };

  const updateWorkflowThinking = (value: ThinkingLevel) => {
    setReasoningOpen(false);
    void updateWorkspace({ workflow_thinking_level: value });
  };

  const openActiveTaskDashboard = (kind: AgentTaskKind) => {
    if (kind === "translation") {
      navigate({ module: "translation", page: "run" });
      return;
    }
    if (kind === "glossary") {
      navigate({ module: "glossary", page: "run" });
      return;
    }
    navigate({ module: "glossary-review", page: "run" });
  };

  const toggleMessageExpanded = (id: string) => {
    setExpandedMessages((current) => {
      const next = new Set(current);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const activeRecipe = pickActiveRecipe(workspace);
  const activeTask = workspace?.active_task ?? null;
  const activeTaskHeader =
    activeTask?.kind === "translation"
      ? translationHeader
      : activeTask?.kind === "glossary"
        ? glossaryHeader
        : activeTask?.kind === "glossary_review"
          ? glossaryReviewHeader
          : null;
  const matchedActiveTaskHeader =
    activeTaskHeader?.id === activeTask?.task_id ? activeTaskHeader : null;
  const profileLookup = new Map(
    (inventory?.profiles ?? []).map((profile) => [profile.id, profile]),
  );
  const workflowProfile = workspace?.workflow_model_id
    ? profileLookup.get(workspace.workflow_model_id)
    : null;
  const workflowThinkingEnabled = Boolean(workflowProfile?.supports_thinking);
  const thinkingLevel = workspace?.workflow_thinking_level ?? "off";
  const workflowModelLabel = workflowProfile
    ? workflowProfile.display_name
    : t.noModel;
  const activeRecipeLabel = activeRecipe
    ? activeRecipe.name
    : t.activeRecipeNone;

  return (
    <div className={styles.page}>
      {error ? <div className={styles.error}>{error}</div> : null}

      <div
        className={`${styles.chatShell} ${
          historyCollapsed ? styles.historyCollapsedShell : ""
        }`.trim()}
      >
        <HistoryPane
          workspace={workspace}
          busy={busy}
          t={t}
          collapsed={historyCollapsed}
          editingConversationId={editingConvId}
          editingConversationTitle={editingConvTitle}
          newMemory={newMemory}
          editingMemoryIndex={editingMemoryIndex}
          editingMemoryText={editingMemoryText}
          onToggleCollapsed={() =>
            setHistoryCollapsed((collapsed) => !collapsed)
          }
          onCreateConversation={() => void createConversation()}
          onSwitchConversation={switchConversation}
          onStartRename={startRename}
          onCommitRename={commitRename}
          onCancelRename={() => setEditingConvId(null)}
          onEditingConversationTitleChange={setEditingConvTitle}
          onDeleteConversation={deleteConversation}
          onNewMemoryChange={setNewMemory}
          onAddMemory={addMemory}
          onStartEditMemory={startEditMemory}
          onEditingMemoryTextChange={setEditingMemoryText}
          onCommitMemoryEdit={commitMemoryEdit}
          onCancelMemoryEdit={() => setEditingMemoryIndex(null)}
          onDeleteMemory={deleteMemory}
        />

        <main className={styles.chatMain}>
          <div className={styles.topBar}>
            <div className={styles.titleBlock}>
              <h1>{t.title}</h1>
              <p>{t.sub}</p>
            </div>
            <button
              type="button"
              className={styles.manageButton}
              onClick={() => navigate({ module: "agent-lab", page: "recipes" })}
            >
              {t.activeRecipeManageHere}
            </button>
          </div>

          {activeTask ? (
            <div className={styles.activeTaskBar}>
              <div className={styles.activeTaskMain}>
                <span className={styles.activeTaskBadge}>
                  {t.activeTaskTitle}
                </span>
                <div>
                  <strong>{t.activeTaskKind[activeTask.kind]}</strong>
                  <p>
                    {t.activeTaskStatus}:{" "}
                    {formatTaskStatus(matchedActiveTaskHeader?.status, messages)}
                    <span aria-hidden="true"> · </span>
                    ID {activeTask.task_id}
                    <span aria-hidden="true"> · </span>
                    {t.activeTaskStartedAt}{" "}
                    {formatDateTime(activeTask.started_at)}
                  </p>
                </div>
              </div>
              <button
                type="button"
                className={styles.activeTaskButton}
                onClick={() => openActiveTaskDashboard(activeTask.kind)}
              >
                {t.activeTaskOpenDashboard}
              </button>
            </div>
          ) : null}

          <ChatMessageList
            messages={workspace?.messages ?? []}
            busy={busy}
            copiedMessageId={copiedMessageId}
            expandedMessages={expandedMessages}
            messagesRef={messagesRef}
            processExpanded={processExpanded}
            t={t}
            onCopyMessage={copyMessage}
            onQuickAction={setInput}
            onResendMessage={resendMessage}
            onToggleMessageExpanded={toggleMessageExpanded}
            onToggleProcessExpanded={() =>
              setProcessExpanded((expanded) => !expanded)
            }
          />

          {workspace?.pending_draft ? (
            <div className={styles.draft}>
              <div>
                <h3>{workspace.pending_draft.title}</h3>
                <p>{workspace.pending_draft.summary}</p>
              </div>
              <DraftConfirmationChecklist draft={workspace.pending_draft} />
              <div className={styles.payloadLabel}>{t.draftPayload}</div>
              <DraftPreview draft={workspace.pending_draft} />
              <label
                className={`${styles.draftConfirmBox} ${
                  draftConfirmed ? styles.draftConfirmBoxReady : ""
                }`}
              >
                <input
                  className={styles.draftConfirmInput}
                  type="checkbox"
                  checked={draftConfirmed}
                  disabled={busy}
                  onChange={(event) => setDraftConfirmed(event.target.checked)}
                />
                <span className={styles.draftConfirmVisual} aria-hidden="true" />
                <div className={styles.draftConfirmCopy}>
                  <strong>{t.draftConfirmTitle}</strong>
                  <p>
                    {draftConfirmed
                      ? t.draftConfirmChecked
                      : t.draftConfirmUnchecked}
                  </p>
                </div>
                <span className={styles.draftConfirmStatus}>
                  {draftConfirmed
                    ? t.draftConfirmReadyLabel
                    : t.draftConfirmLabel}
                </span>
              </label>
              <div className={styles.actions}>
                <Pill disabled={busy || !draftConfirmed} onClick={applyDraft}>
                  {t.applyDraft}
                </Pill>
                <Pill variant="ghost" disabled={busy} onClick={discardDraft}>
                  {t.discardDraft}
                </Pill>
                <Pill variant="ghost" disabled={busy} onClick={startAdjustDraft}>
                  {t.adjustDraft}
                </Pill>
              </div>
              {adjustingDraft ? (
                <div className={styles.adjustBox}>
                  <label htmlFor="agent-draft-adjustment">
                    {t.adjustDraftTitle}
                  </label>
                  <textarea
                    id="agent-draft-adjustment"
                    value={draftAdjustment}
                    autoFocus
                    placeholder={t.adjustDraftPlaceholder}
                    disabled={busy}
                    onChange={(event) =>
                      setDraftAdjustment(event.target.value)
                    }
                    onKeyDown={(event) => {
                      if (event.key !== "Enter") return;
                      if (event.shiftKey || event.nativeEvent.isComposing) {
                        return;
                      }
                      event.preventDefault();
                      void submitDraftAdjustment();
                    }}
                  />
                  <div className={styles.adjustActions}>
                    <Pill
                      disabled={busy || !draftAdjustment.trim()}
                      onClick={() => void submitDraftAdjustment()}
                    >
                      {t.submitAdjustment}
                    </Pill>
                    <Pill
                      variant="ghost"
                      disabled={busy}
                      onClick={cancelDraftAdjustment}
                    >
                      {t.cancelAdjustment}
                    </Pill>
                  </div>
                </div>
              ) : null}
            </div>
          ) : null}

          <ChatComposer
            input={input}
            busy={busy}
            workspace={workspace}
            inventory={inventory}
            activeRecipe={activeRecipe}
            workflowModelLabel={workflowModelLabel}
            activeRecipeLabel={activeRecipeLabel}
            workflowThinkingEnabled={workflowThinkingEnabled}
            thinkingLevel={thinkingLevel}
            t={t}
            thinkingLabels={messages.model.thinking}
            modelOpen={modelOpen}
            recipeOpen={recipeOpen}
            reasoningOpen={reasoningOpen}
            onInputChange={setInput}
            onSend={sendMessage}
            onToggleModelOpen={() => setModelOpen((open) => !open)}
            onToggleRecipeOpen={() => setRecipeOpen((open) => !open)}
            onToggleReasoningOpen={() => setReasoningOpen((open) => !open)}
            onUpdateWorkflowModel={updateWorkflowModel}
            onUpdateWorkflowThinking={updateWorkflowThinking}
            onSwitchRecipe={switchRecipe}
          />
        </main>
      </div>
    </div>
  );
}

function fallbackCopy(text: string): void {
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "true");
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  textarea.select();
  try {
    document.execCommand("copy");
  } finally {
    document.body.removeChild(textarea);
  }
}

function formatDateTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function formatTaskStatus(
  status: string | undefined,
  messages: ReturnType<typeof useMessages>,
): string {
  switch (status) {
    case "running":
      return messages.status.running;
    case "stopping":
    case "pausing":
      return messages.status.stopping;
    case "failed":
      return messages.status.failed;
    case "completed":
      return messages.status.completed;
    case "stopped":
    case "paused":
      return messages.status.stopped;
    case "pending":
    default:
      return messages.status.running;
  }
}

function pickActiveRecipe(workspace: AgentWorkspace | null): AgentRecipe | null {
  if (!workspace?.active_recipe_id) return null;
  return (
    workspace.recipes.find((recipe) => recipe.id === workspace.active_recipe_id) ??
    null
  );
}
