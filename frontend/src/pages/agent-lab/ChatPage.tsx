import { useEffect, useRef, useState } from "react";
import { agentBridge } from "@/bridge/client";
import type {
  AgentActionDraft,
  AgentInventory,
  AgentRecipe,
  AgentTaskKind,
  AgentWorkspace,
  AgentWorkspaceResponse,
  PromptKind,
  ThinkingLevel,
} from "@/bridge/types";
import { Pill } from "@/components/Pill";
import { useMessages } from "@/locales";
import { useModelProfilesStore } from "@/store/useModelProfilesStore";
import { usePromptPresetsStore } from "@/store/usePromptPresetsStore";
import {
  useRuntimeStore,
  type RunKind,
} from "@/store/useRuntimeStore";
import { useTaskStore } from "@/store/useTaskStore";
import styles from "./ChatPage.module.css";

const THINKING_LEVELS: ReadonlyArray<ThinkingLevel> = [
  "off",
  "low",
  "medium",
  "high",
];
const COLLAPSE_MESSAGE_CHARS = 900;
const DRAFT_TEXT_PREVIEW_CHARS = 520;

export function ChatPage() {
  const messages = useMessages();
  const t = messages.agentLab;
  const navigate = useTaskStore((state) => state.navigate);
  const [workspace, setWorkspace] = useState<AgentWorkspace | null>(null);
  const [inventory, setInventory] = useState<AgentInventory | null>(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
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

  const applyResponse = (response: AgentWorkspaceResponse) => {
    setWorkspace(response.workspace);
    setInventory(response.inventory);
    syncRuntimeTasksFromResponse(response);
  };

  const run = async (action: () => Promise<AgentWorkspaceResponse>) => {
    setBusy(true);
    setError(null);
    try {
      const response = await action();
      applyResponse(response);
      await refreshPromptStoresFromResult(response);
      await refreshModelProfilesFromResult(response);
    } catch (err) {
      setError(
        `${t.saveFailed} ${err instanceof Error ? err.message : ""}`.trim(),
      );
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    void (async () => {
      try {
        applyResponse(await agentBridge.readWorkspace());
        setError(null);
      } catch (err) {
        setError(
          `${t.loadFailed} ${err instanceof Error ? err.message : ""}`.trim(),
        );
      }
    })();
  }, [t.loadFailed]);

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
    setBusy(true);
    setError(null);
    if (restoreOnError) {
      setInput("");
    }
    try {
      applyResponse(await agentBridge.sendMessage(text));
    } catch (err) {
      setError(
        `${t.saveFailed} ${err instanceof Error ? err.message : ""}`.trim(),
      );
      if (restoreOnError) {
        setInput(text);
      }
    } finally {
      setBusy(false);
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
    setAdjustingDraft(false);
    setDraftAdjustment("");
    setDraftConfirmed(false);
    void run(() => agentBridge.applyDraft(workspace.pending_draft!.id));
  };

  const discardDraft = () => {
    if (!workspace?.pending_draft) return;
    setAdjustingDraft(false);
    setDraftAdjustment("");
    void run(() => agentBridge.discardDraft(workspace.pending_draft!.id));
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
    await run(() => agentBridge.reviseDraft(draftId, adjustment));
  };

  const createConversation = () => run(() => agentBridge.createConversation());

  const switchConversation = (id: string) => {
    if (id === workspace?.active_conversation_id) return;
    void run(() => agentBridge.switchConversation(id));
  };

  const deleteConversation = (id: string) =>
    void run(() => agentBridge.deleteConversation(id));

  const startRename = (id: string, title: string) => {
    setEditingConvId(id);
    setEditingConvTitle(title);
  };

  const commitRename = async () => {
    const id = editingConvId;
    const title = editingConvTitle.trim();
    setEditingConvId(null);
    if (!id || !title) return;
    await run(() => agentBridge.renameConversation(id, title));
  };

  const addMemory = async () => {
    const text = newMemory.trim();
    if (!text) return;
    setNewMemory("");
    await run(() =>
      agentBridge.updateMemory([...(workspace?.memories ?? []), text]),
    );
  };

  const deleteMemory = (memory: string) =>
    void run(() => agentBridge.deleteMemory(memory));

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
    await run(() => agentBridge.updateMemory(next));
  };

  const switchRecipe = (id: string) => {
    if (!id) return;
    setRecipeOpen(false);
    void run(() => agentBridge.applyRecipe(id));
  };

  const updateWorkflowModel = (value: string) => {
    setModelOpen(false);
    void run(() =>
      agentBridge.updateWorkspace({ workflow_model_id: value || null }),
    );
  };

  const updateWorkflowThinking = (value: ThinkingLevel) =>
    void run(() => {
      setReasoningOpen(false);
      return agentBridge.updateWorkspace({ workflow_thinking_level: value });
    });

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
  const profileLookup = new Map(
    (inventory?.profiles ?? []).map((profile) => [profile.id, profile]),
  );
  const workflowProfile = workspace?.workflow_model_id
    ? profileLookup.get(workspace.workflow_model_id)
    : null;
  const workflowThinkingEnabled = Boolean(workflowProfile?.supports_thinking);
  const thinkingLevel = workspace?.workflow_thinking_level ?? "off";
  const workflowModelLabel = workflowProfile
    ? shortLabel(workflowProfile.display_name)
    : t.noModel;
  const activeRecipeLabel = activeRecipe
    ? shortLabel(activeRecipe.name)
    : t.activeRecipeNone;

  return (
    <div className={styles.page}>
      {error ? <div className={styles.error}>{error}</div> : null}

      <div
        className={`${styles.chatShell} ${
          historyCollapsed ? styles.historyCollapsedShell : ""
        }`.trim()}
      >
        <aside
          className={`${styles.historyPane} ${
            historyCollapsed ? styles.historyPaneCollapsed : ""
          }`.trim()}
        >
          {historyCollapsed ? (
            <div className={styles.collapsedHistory}>
              <button
                type="button"
                className={styles.historyToggle}
                aria-label={t.expandHistory}
                onClick={() => setHistoryCollapsed((collapsed) => !collapsed)}
              >
                +
              </button>
              <span className={styles.collapsedCount}>
                {workspace?.conversations.length ?? 0}
              </span>
            </div>
          ) : (
            <>
              <div className={styles.paneHeader}>
                <div>
                  <h2>{t.conversationsTitle}</h2>
                  <p>{t.conversationsSub}</p>
                </div>
                <button
                  type="button"
                  className={styles.historyToggle}
                  aria-label={t.collapseHistory}
                  onClick={() =>
                    setHistoryCollapsed((collapsed) => !collapsed)
                  }
                >
                  −
                </button>
              </div>

              <button
                type="button"
                className={styles.newConversationInline}
                disabled={busy}
                onClick={() => void createConversation()}
              >
                {t.newConversation}
              </button>

              <div className={styles.convList}>
                {(workspace?.conversations ?? []).map((conversation) => {
                  const isActive =
                    conversation.id === workspace?.active_conversation_id;
                  const isEditing = conversation.id === editingConvId;
                  return (
                    <div
                      key={conversation.id}
                      className={`${styles.convItem} ${
                        isActive ? styles.convActive : ""
                      }`.trim()}
                    >
                      {isEditing ? (
                        <input
                          className={styles.inlineInput}
                          value={editingConvTitle}
                          autoFocus
                          disabled={busy}
                          onChange={(event) =>
                            setEditingConvTitle(event.target.value)
                          }
                          onBlur={() => void commitRename()}
                          onKeyDown={(event) => {
                            if (event.key === "Enter") {
                              event.preventDefault();
                              void commitRename();
                            } else if (event.key === "Escape") {
                              setEditingConvId(null);
                            }
                          }}
                        />
                      ) : (
                        <button
                          type="button"
                          className={styles.convTitle}
                          disabled={busy}
                          onClick={() => switchConversation(conversation.id)}
                        >
                          <span className={styles.convName}>
                            {conversation.title || t.untitledConversation}
                          </span>
                          <span className={styles.convMeta}>
                            {conversation.message_count}
                          </span>
                        </button>
                      )}
                      {isEditing ? null : (
                        <div className={styles.convActions}>
                          <button
                            type="button"
                            className={styles.linkButton}
                            disabled={busy}
                            onClick={() =>
                              startRename(conversation.id, conversation.title)
                            }
                          >
                            {t.renameConversation}
                          </button>
                          <button
                            type="button"
                            className={styles.linkButton}
                            disabled={busy}
                            onClick={() => deleteConversation(conversation.id)}
                          >
                            {t.deleteConversation}
                          </button>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              <div className={styles.memoryBox}>
                <div className={styles.paneSectionTitle}>
                  <h2>{t.memoryTitle}</h2>
                  <p>{t.memorySub}</p>
                </div>
                {workspace?.memories.length ? (
                  <ul className={styles.memoryList}>
                    {workspace.memories.map((memory, index) => (
                      <li
                        key={`${index}-${memory}`}
                        className={styles.memoryItem}
                      >
                        {editingMemoryIndex === index ? (
                          <input
                            className={styles.inlineInput}
                            value={editingMemoryText}
                            autoFocus
                            disabled={busy}
                            onChange={(event) =>
                              setEditingMemoryText(event.target.value)
                            }
                            onBlur={() => void commitMemoryEdit()}
                            onKeyDown={(event) => {
                              if (event.key === "Enter") {
                                event.preventDefault();
                                void commitMemoryEdit();
                              } else if (event.key === "Escape") {
                                setEditingMemoryIndex(null);
                              }
                            }}
                          />
                        ) : (
                          <>
                            <span className={styles.memoryText}>{memory}</span>
                            <div className={styles.memoryActions}>
                              <button
                                type="button"
                                className={styles.linkButton}
                                disabled={busy}
                                onClick={() => startEditMemory(index, memory)}
                              >
                                {t.editMemory}
                              </button>
                              <button
                                type="button"
                                className={styles.linkButton}
                                disabled={busy}
                                onClick={() => deleteMemory(memory)}
                              >
                                {t.deleteMemory}
                              </button>
                            </div>
                          </>
                        )}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div className={styles.empty}>{t.memoryEmpty}</div>
                )}
                <div className={styles.memoryAdd}>
                  <input
                    className={styles.inlineInput}
                    value={newMemory}
                    placeholder={t.memoryAddPlaceholder}
                    disabled={busy}
                    onChange={(event) => setNewMemory(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") {
                        event.preventDefault();
                        void addMemory();
                      }
                    }}
                  />
                  <Pill
                    disabled={busy || !newMemory.trim()}
                    onClick={() => void addMemory()}
                  >
                    {t.memoryAdd}
                  </Pill>
                </div>
              </div>
            </>
          )}
        </aside>

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

          <div className={styles.messages} ref={messagesRef}>
            {(workspace?.messages ?? []).map((message) => {
              const isLong = message.content.length > COLLAPSE_MESSAGE_CHARS;
              const isExpanded = expandedMessages.has(message.id);
              const body =
                isLong && !isExpanded
                  ? `${message.content
                      .slice(0, COLLAPSE_MESSAGE_CHARS)
                      .trimEnd()}…`
                  : message.content;
              return (
                <div
                  key={message.id}
                  className={`${styles.message} ${
                    message.role === "user"
                      ? styles.userMessage
                      : styles.agentMessage
                  }`}
                >
                  <div className={styles.messageRole}>{message.role}</div>
                  <div className={styles.messageBody}>{body}</div>
                  <div className={styles.messageActions}>
                    {isLong ? (
                      <button
                        type="button"
                        className={styles.messageActionButton}
                        onClick={() => toggleMessageExpanded(message.id)}
                      >
                        {isExpanded ? t.collapseMessage : t.expandMessage}
                      </button>
                    ) : null}
                    {message.role === "user" ? (
                      <button
                        type="button"
                        className={styles.messageActionButton}
                        disabled={busy}
                        onClick={() => resendMessage(message.content)}
                      >
                        {t.resendMessage}
                      </button>
                    ) : null}
                    <button
                      type="button"
                      className={styles.messageActionButton}
                      disabled={busy}
                      onClick={() => void copyMessage(message.id, message.content)}
                    >
                      {copiedMessageId === message.id
                        ? t.copiedMessage
                        : t.copyMessage}
                    </button>
                  </div>
                </div>
              );
            })}
            {(workspace?.messages ?? []).length <= 1 ? (
              <div className={styles.quickActions}>
                {t.quickActions.map((action) => (
                  <button
                    key={action}
                    type="button"
                    disabled={busy}
                    onClick={() => setInput(action)}
                  >
                    {action}
                  </button>
                ))}
              </div>
            ) : null}
            {busy ? (
              <div
                className={`${styles.message} ${styles.agentMessage} ${styles.thinkingMessage}`}
                aria-live="polite"
              >
                <div className={styles.messageRole}>{t.thinkingTitle}</div>
                <div className={styles.thinkingLead}>
                  <span className={styles.spinner} aria-hidden="true" />
                  <span>{t.thinkingLead}</span>
                </div>
                <button
                  type="button"
                  className={styles.processToggle}
                  onClick={() => setProcessExpanded((expanded) => !expanded)}
                >
                  {processExpanded ? t.hideProcessDetails : t.showProcessDetails}
                </button>
                {processExpanded ? (
                  <>
                    <div className={styles.thinkingSteps}>
                      {t.thinkingSteps.map((step) => (
                        <span key={step}>{step}</span>
                      ))}
                    </div>
                    <p className={styles.processDisclosure}>
                      {t.processDisclosure}
                    </p>
                  </>
                ) : null}
              </div>
            ) : null}
            <div className={styles.messagesEnd} />
          </div>

          {workspace?.pending_draft ? (
            <div className={styles.draft}>
              <div>
                <h3>{workspace.pending_draft.title}</h3>
                <p>{workspace.pending_draft.summary}</p>
              </div>
              <DraftConfirmationChecklist draft={workspace.pending_draft} />
              <div className={styles.payloadLabel}>{t.draftPayload}</div>
              <DraftPreview draft={workspace.pending_draft} />
              <div className={styles.draftConfirmBox}>
                <label className={styles.draftConfirmControl}>
                  <input
                    type="checkbox"
                    checked={draftConfirmed}
                    disabled={busy}
                    onChange={(event) => setDraftConfirmed(event.target.checked)}
                  />
                  <span>我已检查草案内容，授权 Agent 执行这些动作。</span>
                </label>
                <p>
                  {!draftConfirmed
                    ? "勾选后才能应用；未勾选、放弃或调整都不会写入配置或启动任务。"
                    : "已确认。点击「应用」后会按草案内容执行。"}
                </p>
              </div>
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

          <div className={styles.composer}>
            <textarea
              value={input}
              placeholder={t.inputPlaceholder}
              disabled={busy}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key !== "Enter") return;
                if (event.shiftKey || event.nativeEvent.isComposing) return;
                event.preventDefault();
                void sendMessage();
              }}
            />
            <div className={styles.composerBar}>
              <div className={styles.composerControls}>
                <div className={styles.compactMenu}>
                  <button
                    type="button"
                    className={styles.compactButton}
                    disabled={busy || !workspace || !inventory}
                    aria-label={t.workflowModel}
                    aria-expanded={modelOpen}
                    aria-haspopup="menu"
                    onClick={() => setModelOpen((open) => !open)}
                  >
                    <span>{workflowModelLabel}</span>
                    <span aria-hidden="true">⌄</span>
                  </button>
                  {modelOpen ? (
                    <div className={styles.compactPopup} role="menu">
                      <button
                        type="button"
                        role="menuitemradio"
                        aria-checked={!workspace?.workflow_model_id}
                        className={styles.compactOption}
                        onClick={() => updateWorkflowModel("")}
                      >
                        <span>{t.noModel}</span>
                        {!workspace?.workflow_model_id ? (
                          <span aria-hidden="true">✓</span>
                        ) : null}
                      </button>
                      {(inventory?.profiles ?? []).map((profile) => (
                        <button
                          key={profile.id}
                          type="button"
                          role="menuitemradio"
                          aria-checked={profile.id === workspace?.workflow_model_id}
                          className={styles.compactOption}
                          onClick={() => updateWorkflowModel(profile.id)}
                        >
                          <span>{shortLabel(profile.display_name)}</span>
                          {profile.id === workspace?.workflow_model_id ? (
                            <span aria-hidden="true">✓</span>
                          ) : null}
                        </button>
                      ))}
                    </div>
                  ) : null}
                </div>
                <div className={styles.reasoningMenu}>
                  <button
                    type="button"
                    className={styles.reasoningButton}
                    disabled={busy || !workflowThinkingEnabled}
                    aria-expanded={reasoningOpen}
                    aria-haspopup="menu"
                    onClick={() => setReasoningOpen((open) => !open)}
                  >
                    {messages.model.thinking[thinkingLevel]}
                    <span aria-hidden="true">⌄</span>
                  </button>
                  {reasoningOpen && workflowThinkingEnabled ? (
                    <div className={styles.reasoningPopup} role="menu">
                      <div className={styles.reasoningHeading}>
                        {t.workflowThinkingLevel}
                      </div>
                      {THINKING_LEVELS.map((level) => (
                        <button
                          key={level}
                          type="button"
                          role="menuitemradio"
                          aria-checked={level === thinkingLevel}
                          className={styles.reasoningOption}
                          onClick={() => updateWorkflowThinking(level)}
                        >
                          <span>{messages.model.thinking[level]}</span>
                          {level === thinkingLevel ? (
                            <span aria-hidden="true">✓</span>
                          ) : null}
                        </button>
                      ))}
                    </div>
                  ) : null}
                </div>
                <div className={styles.compactMenu}>
                  <button
                    type="button"
                    className={styles.compactButton}
                    disabled={busy || !workspace}
                    aria-label={t.activeRecipeTitle}
                    aria-expanded={recipeOpen}
                    aria-haspopup="menu"
                    onClick={() => setRecipeOpen((open) => !open)}
                  >
                    <span>{activeRecipeLabel}</span>
                    <span aria-hidden="true">⌄</span>
                  </button>
                  {recipeOpen ? (
                    <div className={styles.compactPopup} role="menu">
                      {(workspace?.recipes ?? []).length ? (
                        (workspace?.recipes ?? []).map((recipe) => (
                          <button
                            key={recipe.id}
                            type="button"
                            role="menuitemradio"
                            aria-checked={recipe.id === activeRecipe?.id}
                            className={styles.compactOption}
                            onClick={() => switchRecipe(recipe.id)}
                          >
                            <span>{shortLabel(recipe.name)}</span>
                            {recipe.id === activeRecipe?.id ? (
                              <span aria-hidden="true">✓</span>
                            ) : null}
                          </button>
                        ))
                      ) : (
                        <div className={styles.compactEmpty}>
                          {t.activeRecipeNone}
                        </div>
                      )}
                    </div>
                  ) : null}
                </div>
                <Pill
                  disabled={busy || !input.trim()}
                  onClick={() => void sendMessage()}
                >
                  {busy ? t.sending : t.send}
                </Pill>
              </div>
            </div>
          </div>
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

function shortLabel(label: string): string {
  const trimmed = label.trim();
  if (trimmed.length <= 14) return trimmed;
  return `${trimmed.slice(0, 13)}…`;
}

async function refreshPromptStoresFromResult(
  response: AgentWorkspaceResponse,
): Promise<void> {
  const kinds = promptKindsFromResult(response.result);
  if (!kinds.size) return;
  const store = usePromptPresetsStore.getState();
  await Promise.all([...kinds].map((kind) => store.refresh(kind)));
}

function promptKindsFromResult(
  result: Record<string, unknown> | undefined,
): Set<PromptKind> {
  const kinds = new Set<PromptKind>();
  collectPromptKinds(result, kinds);
  return kinds;
}

function collectPromptKinds(value: unknown, kinds: Set<PromptKind>): void {
  if (Array.isArray(value)) {
    value.forEach((item) => collectPromptKinds(item, kinds));
    return;
  }
  if (!isRecord(value)) return;
  const preset = value.preset;
  if (isRecord(preset) && isPromptKind(preset.kind)) {
    kinds.add(preset.kind);
  }
  Object.values(value).forEach((item) => collectPromptKinds(item, kinds));
}

async function refreshModelProfilesFromResult(
  response: AgentWorkspaceResponse,
): Promise<void> {
  if (!resultIncludesModelProfileChange(response.result)) return;
  await useModelProfilesStore.getState().refresh();
}

function resultIncludesModelProfileChange(value: unknown): boolean {
  if (Array.isArray(value)) {
    return value.some((item) => resultIncludesModelProfileChange(item));
  }
  if (!isRecord(value)) return false;
  if (
    value.kind === "create_model_profile" ||
    value.kind === "update_model_profile"
  ) {
    return true;
  }
  if (isRecord(value.profile) && typeof value.profile.id === "string") {
    return true;
  }
  return Object.values(value).some((item) =>
    resultIncludesModelProfileChange(item),
  );
}

function syncRuntimeTasksFromResponse(response: AgentWorkspaceResponse): void {
  const tasks = startedTasksFromResult(response.result);
  const activeTask = response.workspace.active_task;
  if (activeTask) {
    tasks.push({
      kind: activeTask.kind,
      taskId: activeTask.task_id,
    });
  }
  if (!tasks.length) return;

  const runtime = useRuntimeStore.getState();
  const seen = new Set<string>();
  for (const task of tasks) {
    const runKind = runKindFromAgentTaskKind(task.kind);
    if (!runKind) continue;
    const key = `${runKind}:${task.taskId}`;
    if (seen.has(key)) continue;
    seen.add(key);
    runtime.setActiveTaskId(runKind, task.taskId);
    void runtime.pollSnapshot(runKind);
  }
}

function startedTasksFromResult(
  result: Record<string, unknown> | undefined,
): Array<{ kind: AgentTaskKind; taskId: string }> {
  const tasks: Array<{ kind: AgentTaskKind; taskId: string }> = [];
  collectStartedTasks(result, tasks);
  return tasks;
}

function collectStartedTasks(
  value: unknown,
  tasks: Array<{ kind: AgentTaskKind; taskId: string }>,
): void {
  if (Array.isArray(value)) {
    value.forEach((item) => collectStartedTasks(item, tasks));
    return;
  }
  if (!isRecord(value)) return;

  const task = value.task;
  if (isRecord(task) && isAgentTaskKind(task.kind)) {
    const taskId = task.task_id;
    if (typeof taskId === "string" && taskId.trim()) {
      tasks.push({ kind: task.kind, taskId });
    }
  }

  Object.values(value).forEach((item) => collectStartedTasks(item, tasks));
}

function runKindFromAgentTaskKind(kind: AgentTaskKind): RunKind {
  return kind;
}

function isAgentTaskKind(value: unknown): value is AgentTaskKind {
  return (
    value === "translation" ||
    value === "glossary" ||
    value === "glossary_review"
  );
}

function isPromptKind(value: unknown): value is PromptKind {
  return (
    value === "translation" ||
    value === "glossary" ||
    value === "glossary_review"
  );
}

function DraftPreview({ draft }: { draft: AgentActionDraft }) {
  if (draft.kind === "compound_config_update") {
    const actions = Array.isArray(draft.payload.actions)
      ? draft.payload.actions.filter(isRecord)
      : [];
    return (
      <div className={styles.draftPreview}>
        <div className={styles.draftSummary}>
          <span className={styles.draftBadge}>{formatDraftKind(draft.kind)}</span>
          <span>
            {actions.length
              ? `包含 ${actions.length} 个待确认修改`
              : "复合配置修改"}
          </span>
        </div>
        {actions.length ? (
          <div className={styles.draftActionList}>
            {actions.map((action, index) => (
              <DraftActionPreview key={index} action={action} index={index} />
            ))}
          </div>
        ) : (
          <DraftFields payload={draft.payload} />
        )}
        <RawDraftDetails payload={draft.payload} />
      </div>
    );
  }

  return (
    <div className={styles.draftPreview}>
      <div className={styles.draftSummary}>
        <span className={styles.draftBadge}>{formatDraftKind(draft.kind)}</span>
        <span>{draft.title}</span>
      </div>
      <DraftFields payload={draft.payload} />
      <RawDraftDetails payload={draft.payload} />
    </div>
  );
}

function DraftConfirmationChecklist({ draft }: { draft: AgentActionDraft }) {
  const rows = draftConfirmationRows(draft);
  return (
    <div className={styles.draftChecklist} aria-label="草案确认要点">
      {rows.map((row) => (
        <div key={row} className={styles.draftChecklistRow}>
          <span className={styles.draftCheckMark} aria-hidden="true" />
          <span>{row}</span>
        </div>
      ))}
    </div>
  );
}

function draftConfirmationRows(draft: AgentActionDraft): string[] {
  const rows: string[] = [];
  const actionCount = draftActionCount(draft);
  if (actionCount > 1) {
    rows.push(`多步草案：包含 ${actionCount} 个动作，会按显示顺序执行。`);
  }
  if (draftStartsTask(draft)) {
    rows.push("任务启动：点击「应用」后才会启动任务，并在对应 dashboard 显示进度。");
  } else {
    rows.push("配置写入：点击「应用」后才会保存；未应用前不会改动配置。");
  }
  rows.push("安全确认：点击「放弃」不会写入；点击「调整」会让 Agent 重写草案。");
  return rows;
}

function draftActionCount(draft: AgentActionDraft): number {
  if (draft.kind !== "compound_config_update") return 1;
  const actions = draft.payload.actions;
  return Array.isArray(actions) ? actions.filter(isRecord).length : 1;
}

function draftStartsTask(draft: AgentActionDraft): boolean {
  if (isStartTaskKind(draft.kind)) return true;
  if (draft.kind !== "compound_config_update") return false;
  const actions = draft.payload.actions;
  if (!Array.isArray(actions)) return false;
  return actions.some(
    (action) => isRecord(action) && isStartTaskKind(String(action.kind || "")),
  );
}

function isStartTaskKind(kind: string): boolean {
  return (
    kind === "start_glossary_task" ||
    kind === "start_glossary_review_task" ||
    kind === "start_translation_task"
  );
}

function DraftActionPreview({
  action,
  index,
}: {
  action: Record<string, unknown>;
  index: number;
}) {
  const payload = isRecord(action.payload) ? action.payload : {};
  const kind = typeof action.kind === "string" ? action.kind : "";
  const title =
    typeof action.title === "string" && action.title.trim()
      ? action.title
      : formatDraftKind(kind);
  const summary =
    typeof action.summary === "string" && action.summary.trim()
      ? action.summary
      : "";
  return (
    <div className={styles.draftActionItem}>
      <div className={styles.draftActionHeader}>
        <span className={styles.draftActionIndex}>{index + 1}</span>
        <div>
          <strong>{title}</strong>
          {summary ? <p>{summary}</p> : null}
        </div>
      </div>
      <DraftFields payload={payload} />
    </div>
  );
}

function DraftFields({ payload }: { payload: Record<string, unknown> }) {
  const entries = visibleDraftEntries(payload);
  if (!entries.length) {
    return <div className={styles.draftEmpty}>没有可展示的配置字段。</div>;
  }
  return (
    <dl className={styles.draftFields}>
      {entries.map(([key, value]) => (
        <div key={key} className={styles.draftField}>
          <dt>{formatDraftField(key)}</dt>
          <dd>{renderDraftValue(key, value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function visibleDraftEntries(
  payload: Record<string, unknown>,
): Array<[string, unknown]> {
  return Object.entries(payload).filter(([key, value]) => {
    if (key === "actions") return false;
    return value !== undefined;
  });
}

function renderDraftValue(key: string, value: unknown): JSX.Element {
  if (typeof value === "string") {
    return renderDraftString(key, value);
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return <span>{formatPrimitive(value)}</span>;
  }
  if (value === null) {
    return <span className={styles.draftMuted}>未设置</span>;
  }
  if (Array.isArray(value)) {
    if (!value.length) return <span className={styles.draftMuted}>空列表</span>;
    return (
      <ul className={styles.draftMiniList}>
        {value.map((item, index) => (
          <li key={index}>{renderInlineValue(item)}</li>
        ))}
      </ul>
    );
  }
  if (isRecord(value)) {
    const entries = Object.entries(value);
    if (!entries.length) return <span className={styles.draftMuted}>无</span>;
    return (
      <dl className={styles.draftNestedFields}>
        {entries.map(([childKey, childValue]) => (
          <div key={childKey}>
            <dt>{formatDraftField(childKey)}</dt>
            <dd>{renderInlineValue(childValue)}</dd>
          </div>
        ))}
      </dl>
    );
  }
  return <span>{String(value)}</span>;
}

function renderDraftString(key: string, value: string): JSX.Element {
  const text = value.trim();
  if (!text) return <span className={styles.draftMuted}>空</span>;
  const isLong =
    key === "system_prompt" ||
    key === "prompt" ||
    key === "content" ||
    text.length > DRAFT_TEXT_PREVIEW_CHARS;
  if (!isLong) return <span>{text}</span>;
  const preview =
    text.length > DRAFT_TEXT_PREVIEW_CHARS
      ? `${text.slice(0, DRAFT_TEXT_PREVIEW_CHARS).trimEnd()}…`
      : text;
  return (
    <details className={styles.draftTextDetails}>
      <summary>{preview}</summary>
      <div>{text}</div>
    </details>
  );
}

function renderInlineValue(value: unknown): string {
  if (typeof value === "string") return value || "未设置";
  if (typeof value === "number" || typeof value === "boolean") {
    return formatPrimitive(value);
  }
  if (value === null || value === undefined) return "未设置";
  if (Array.isArray(value)) return value.map(renderInlineValue).join("、");
  if (isRecord(value)) {
    return Object.entries(value)
      .map(([key, child]) => `${formatDraftField(key)}：${renderInlineValue(child)}`)
      .join("；");
  }
  return String(value);
}

function formatPrimitive(value: number | boolean): string {
  if (typeof value === "boolean") return value ? "是" : "否";
  return String(value);
}

function RawDraftDetails({ payload }: { payload: Record<string, unknown> }) {
  return (
    <details className={styles.draftRawDetails}>
      <summary>查看技术详情</summary>
      <pre>{JSON.stringify(payload, null, 2)}</pre>
    </details>
  );
}

function formatDraftKind(kind: string): string {
  return (
    {
      create_prompt_preset: "创建 Prompt 预设",
      update_prompt_preset: "更新 Prompt 预设",
      create_recipe: "创建预设配置",
      update_recipe: "更新预设配置",
      apply_recipe: "应用预设配置",
      delete_recipe: "删除预设配置",
      create_model_profile: "创建模型配置",
      update_model_profile: "更新模型配置",
      update_workspace: "更新工作区配置",
      update_memory: "更新记忆",
      add_memory: "添加记忆",
      delete_memory: "删除记忆",
      compound_config_update: "复合配置修改",
      start_glossary_task: "启动术语提取任务",
      start_glossary_review_task: "启动术语审查任务",
      start_translation_task: "启动翻译任务",
    }[kind] ?? kind
  );
}

function formatDraftField(key: string): string {
  return (
    {
      kind: "类型",
      id: "ID",
      name: "名称",
      title: "标题",
      description: "说明",
      system_prompt: "系统提示词",
      prompt: "提示词",
      content: "内容",
      enabled: "启用",
      profile_id: "模型 ID",
      recipe_id: "预设 ID",
      patch: "修改内容",
      stage_model_ids: "阶段模型",
      stage_prompt_ids: "阶段 Prompt",
      workflow_model_id: "工作模型",
      workflow_thinking_level: "思考强度",
      translation: "翻译",
      glossary: "术语提取",
      glossary_review: "术语审查",
      term_extract: "术语提取",
      term_review: "术语审查",
      input_dir: "输入目录",
      output_dir: "输出目录",
      input_folder: "输入目录",
      output_folder: "输出目录",
      source_language: "源语言",
      target_language: "目标语言",
      novel_background: "小说背景",
      glossary_task_id: "术语任务 ID",
      glossary_review_task_id: "术语审查任务 ID",
      concurrency_limit: "并发数",
      rpm_limit: "每分钟请求数",
      tpm_limit: "每分钟 Token 数",
      retry_attempts: "重试次数",
      model_id: "模型名称",
      provider_format: "接口类型",
      base_url: "接口地址",
    }[key] ?? key
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function pickActiveRecipe(workspace: AgentWorkspace | null): AgentRecipe | null {
  if (!workspace?.active_recipe_id) return null;
  return (
    workspace.recipes.find((recipe) => recipe.id === workspace.active_recipe_id) ??
    null
  );
}
