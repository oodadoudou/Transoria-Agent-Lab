import { useEffect, useState } from "react";
import type {
  AgentRecipe,
  AgentTaskKind,
  AgentWorkspace,
  ThinkingLevel,
} from "@/bridge/types";
import { useMessages } from "@/locales";
import { useRuntimeStore, usePollRunSnapshot } from "@/store/useRuntimeStore";
import { useTaskStore } from "@/store/useTaskStore";
import { ActiveTaskBar } from "./components/ActiveTaskBar";
import { ChatComposer } from "./components/ChatComposer";
import { ChatMessageList } from "./components/ChatMessageList";
import { DraftSection } from "./components/DraftSection";
import { HistoryPane } from "./components/HistoryPane";
import styles from "./ChatPage.module.css";
import { useAgentWorkspace } from "./useAgentWorkspace";
import { useChatMessageActions } from "./useChatMessageActions";
import { useDraftControls } from "./useDraftControls";
import { useHistoryPaneState } from "./useHistoryPaneState";

export function ChatPage() {
  const messages = useMessages();
  const t = messages.agentLab;
  const navigate = useTaskStore((state) => state.navigate);
  const translationHeader = useRuntimeStore((state) => state.translation.header);
  const glossaryHeader = useRuntimeStore((state) => state.glossary.header);
  const glossaryReviewHeader = useRuntimeStore(
    (state) => state.glossary_review.header,
  );
  const [modelOpen, setModelOpen] = useState(false);
  const [recipeOpen, setRecipeOpen] = useState(false);
  const [reasoningOpen, setReasoningOpen] = useState(false);
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
  const {
    input,
    setInput,
    copiedMessageId,
    expandedMessages,
    messagesRef,
    processExpanded,
    sendMessage,
    resendMessage,
    copyMessage,
    toggleMessageExpanded,
    toggleProcessExpanded,
  } = useChatMessageActions({
    sendWorkspaceMessage,
    setError,
  });
  const {
    adjustingDraft,
    draftAdjustment,
    draftConfirmed,
    setDraftAdjustment,
    setDraftConfirmed,
    applyDraft,
    discardDraft,
    startAdjustDraft,
    cancelDraftAdjustment,
    submitDraftAdjustment,
  } = useDraftControls({
    workspace,
    applyWorkspaceDraft,
    discardWorkspaceDraft,
    reviseDraft,
  });
  const {
    editingConvId,
    editingConvTitle,
    newMemory,
    editingMemoryIndex,
    editingMemoryText,
    historyCollapsed,
    setEditingConvTitle,
    setNewMemory,
    setEditingMemoryText,
    toggleHistoryCollapsed,
    createConversation,
    switchConversation,
    deleteConversation,
    startRename,
    commitRename,
    cancelRename,
    addMemory,
    deleteMemory,
    startEditMemory,
    commitMemoryEdit,
    cancelMemoryEdit,
  } = useHistoryPaneState({
    workspace,
    createWorkspaceConversation,
    switchWorkspaceConversation,
    renameConversation,
    deleteWorkspaceConversation,
    updateMemory,
    deleteWorkspaceMemory,
  });

  useEffect(() => {
    const container = messagesRef.current;
    if (!container) return;
    container.scrollTo({
      top: container.scrollHeight,
      behavior: "smooth",
    });
  }, [busy, workspace?.messages.length, workspace?.pending_draft?.id]);

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
          onToggleCollapsed={toggleHistoryCollapsed}
          onCreateConversation={() => void createConversation()}
          onSwitchConversation={switchConversation}
          onStartRename={startRename}
          onCommitRename={commitRename}
          onCancelRename={cancelRename}
          onEditingConversationTitleChange={setEditingConvTitle}
          onDeleteConversation={deleteConversation}
          onNewMemoryChange={setNewMemory}
          onAddMemory={addMemory}
          onStartEditMemory={startEditMemory}
          onEditingMemoryTextChange={setEditingMemoryText}
          onCommitMemoryEdit={commitMemoryEdit}
          onCancelMemoryEdit={cancelMemoryEdit}
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
            <ActiveTaskBar
              activeTask={activeTask}
              taskStatus={matchedActiveTaskHeader?.status}
              t={t}
              messages={messages}
              onOpenDashboard={() => openActiveTaskDashboard(activeTask.kind)}
            />
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
            onToggleProcessExpanded={toggleProcessExpanded}
          />

          {workspace?.pending_draft ? (
            <DraftSection
              draft={workspace.pending_draft}
              busy={busy}
              t={t}
              adjustingDraft={adjustingDraft}
              draftAdjustment={draftAdjustment}
              draftConfirmed={draftConfirmed}
              onDraftAdjustmentChange={setDraftAdjustment}
              onDraftConfirmedChange={setDraftConfirmed}
              onApplyDraft={applyDraft}
              onDiscardDraft={discardDraft}
              onStartAdjustDraft={startAdjustDraft}
              onCancelDraftAdjustment={cancelDraftAdjustment}
              onSubmitDraftAdjustment={submitDraftAdjustment}
            />
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

function pickActiveRecipe(workspace: AgentWorkspace | null): AgentRecipe | null {
  if (!workspace?.active_recipe_id) return null;
  return (
    workspace.recipes.find((recipe) => recipe.id === workspace.active_recipe_id) ??
    null
  );
}
