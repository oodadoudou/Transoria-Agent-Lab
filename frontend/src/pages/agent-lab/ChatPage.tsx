import { useEffect, useRef, useState } from "react";
import { agentBridge } from "@/bridge/client";
import type {
  AgentInventory,
  AgentRecipe,
  AgentWorkspace,
  AgentWorkspaceResponse,
  ThinkingLevel,
} from "@/bridge/types";
import { Pill } from "@/components/Pill";
import { useMessages } from "@/locales";
import { useTaskStore } from "@/store/useTaskStore";
import styles from "./ChatPage.module.css";

const THINKING_LEVELS: ReadonlyArray<ThinkingLevel> = [
  "off",
  "low",
  "medium",
  "high",
];
const COLLAPSE_MESSAGE_CHARS = 900;

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
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const messagesRef = useRef<HTMLDivElement | null>(null);
  const [expandedMessages, setExpandedMessages] = useState<ReadonlySet<string>>(
    () => new Set(),
  );

  const applyResponse = (response: AgentWorkspaceResponse) => {
    setWorkspace(response.workspace);
    setInventory(response.inventory);
  };

  const run = async (action: () => Promise<AgentWorkspaceResponse>) => {
    setBusy(true);
    setError(null);
    try {
      applyResponse(await action());
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
    if (!workspace?.pending_draft) return;
    setAdjustingDraft(false);
    setDraftAdjustment("");
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
              <div className={styles.payloadLabel}>{t.draftPayload}</div>
              <pre>{JSON.stringify(workspace.pending_draft.payload, null, 2)}</pre>
              <div className={styles.actions}>
                <Pill disabled={busy} onClick={applyDraft}>
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

function pickActiveRecipe(workspace: AgentWorkspace | null): AgentRecipe | null {
  if (!workspace?.active_recipe_id) return null;
  return (
    workspace.recipes.find((recipe) => recipe.id === workspace.active_recipe_id) ??
    null
  );
}
