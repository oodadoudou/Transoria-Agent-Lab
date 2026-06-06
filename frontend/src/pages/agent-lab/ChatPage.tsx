import { useEffect, useState } from "react";
import { agentBridge } from "@/bridge/client";
import type {
  AgentInventory,
  AgentModelSlot,
  AgentPromptSlot,
  AgentRecipe,
  AgentWorkspace,
  AgentWorkspaceResponse,
} from "@/bridge/types";
import { Panel } from "@/components/Panel";
import { Pill } from "@/components/Pill";
import { useMessages } from "@/locales";
import { useTaskStore } from "@/store/useTaskStore";
import styles from "./ChatPage.module.css";

const MODEL_SLOTS: ReadonlyArray<AgentModelSlot> = [
  "translation",
  "term_extract",
  "term_review",
];

const PROMPT_SLOTS: ReadonlyArray<AgentPromptSlot> = [
  "translation",
  "term_extract",
  "term_review",
];

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

  const sendMessage = async () => {
    const text = input.trim();
    if (!text) return;
    setBusy(true);
    setError(null);
    setInput("");
    try {
      applyResponse(await agentBridge.sendMessage(text));
    } catch (err) {
      setError(
        `${t.saveFailed} ${err instanceof Error ? err.message : ""}`.trim(),
      );
      setInput(text);
    } finally {
      setBusy(false);
    }
  };

  const applyDraft = () => {
    if (!workspace?.pending_draft) return;
    void run(() => agentBridge.applyDraft(workspace.pending_draft!.id));
  };

  const discardDraft = () => {
    if (!workspace?.pending_draft) return;
    void run(() => agentBridge.discardDraft(workspace.pending_draft!.id));
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
    void run(() => agentBridge.applyRecipe(id));
  };

  const updateWorkflowModel = (value: string) =>
    void run(() =>
      agentBridge.updateWorkspace({ workflow_model_id: value || null }),
    );

  const activeRecipe = pickActiveRecipe(workspace);
  const profileLookup = new Map(
    (inventory?.profiles ?? []).map((profile) => [profile.id, profile]),
  );
  const promptLookup = new Map(
    Object.values(inventory?.prompts ?? {})
      .flat()
      .map((prompt) => [prompt.id, prompt]),
  );

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div>
          <h1>{t.title}</h1>
          <p>{t.sub}</p>
        </div>
      </div>

      {error ? <div className={styles.error}>{error}</div> : null}

      <div className={styles.grid}>
        <div className={styles.sideStack}>
          <Panel label={t.conversationsTitle} subtitle={t.conversationsSub}>
            <div className={styles.stack}>
              <Pill disabled={busy} onClick={() => void createConversation()}>
                {t.newConversation}
              </Pill>
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
            </div>
          </Panel>

          <Panel label={t.activeRecipeTitle} subtitle={t.activeRecipeSub}>
            <div className={styles.stack}>
              <label className={styles.selectField}>
                <span>{t.activeRecipeTitle}</span>
                <select
                  value={activeRecipe?.id ?? ""}
                  disabled={busy || !workspace}
                  onChange={(event) => switchRecipe(event.target.value)}
                >
                  <option value="">{t.activeRecipeNone}</option>
                  {(workspace?.recipes ?? []).map((recipe) => (
                    <option key={recipe.id} value={recipe.id}>
                      {recipe.name}
                    </option>
                  ))}
                </select>
              </label>

              <div className={styles.stageSummary}>
                {MODEL_SLOTS.map((slot) => {
                  const modelId = workspace?.stage_model_ids[slot] ?? null;
                  const profile = modelId ? profileLookup.get(modelId) : null;
                  return (
                    <div key={`m-${slot}`} className={styles.stageRow}>
                      <span className={styles.stageLabel}>
                        {t.stageModel[slot]}
                      </span>
                      <span className={styles.stageValue}>
                        {profile?.display_name ?? t.stageEmptyModel}
                      </span>
                    </div>
                  );
                })}
                {PROMPT_SLOTS.map((slot) => {
                  const promptId = workspace?.stage_prompt_ids[slot] ?? null;
                  const prompt = promptId ? promptLookup.get(promptId) : null;
                  return (
                    <div key={`p-${slot}`} className={styles.stageRow}>
                      <span className={styles.stageLabel}>
                        {t.stagePrompt[slot]}
                      </span>
                      <span className={styles.stageValue}>
                        {prompt?.name ?? t.stageEmptyPrompt}
                      </span>
                    </div>
                  );
                })}
              </div>

              <button
                type="button"
                className={styles.linkButton}
                onClick={() =>
                  navigate({ module: "agent-lab", page: "recipes" })
                }
              >
                {t.activeRecipeManageHere}
              </button>
            </div>
          </Panel>

          <Panel label={t.configTitle} subtitle={t.configSub}>
            <div className={styles.stack}>
              <label className={styles.selectField}>
                <span>{t.workflowModel}</span>
                <select
                  value={workspace?.workflow_model_id ?? ""}
                  disabled={busy || !workspace || !inventory}
                  onChange={(event) => updateWorkflowModel(event.target.value)}
                >
                  <option value="">{t.noModel}</option>
                  {(inventory?.profiles ?? []).map((profile) => (
                    <option key={profile.id} value={profile.id}>
                      {profile.api_key_configured
                        ? profile.display_name
                        : `${profile.display_name} (${t.modelNotConfigured})`}
                    </option>
                  ))}
                </select>
              </label>
              <div className={styles.boundary}>{t.confirmationBoundary}</div>
            </div>
          </Panel>
        </div>

        <Panel
          label={t.chatTitle}
          subtitle={t.chatSub}
          className={styles.chatPanel}
        >
          <div className={styles.messages}>
            {(workspace?.messages ?? []).map((message) => (
              <div
                key={message.id}
                className={`${styles.message} ${
                  message.role === "user"
                    ? styles.userMessage
                    : styles.agentMessage
                }`}
              >
                <div className={styles.messageRole}>{message.role}</div>
                <div className={styles.messageBody}>{message.content}</div>
              </div>
            ))}
          </div>
          <div className={styles.composer}>
            <textarea
              value={input}
              placeholder={t.inputPlaceholder}
              disabled={busy}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                  event.preventDefault();
                  void sendMessage();
                }
              }}
            />
            <Pill
              disabled={busy || !input.trim()}
              onClick={() => void sendMessage()}
            >
              {busy ? t.sending : t.send}
            </Pill>
          </div>
        </Panel>

        <div className={styles.sideStack}>
          <Panel label={t.draftTitle} subtitle={t.draftSub}>
            {workspace?.pending_draft ? (
              <div className={styles.draft}>
                <h3>{workspace.pending_draft.title}</h3>
                <p>{workspace.pending_draft.summary}</p>
                <div className={styles.payloadLabel}>{t.draftPayload}</div>
                <pre>
                  {JSON.stringify(workspace.pending_draft.payload, null, 2)}
                </pre>
                <div className={styles.actions}>
                  <Pill disabled={busy} onClick={applyDraft}>
                    {t.applyDraft}
                  </Pill>
                  <Pill
                    variant="ghost"
                    disabled={busy}
                    onClick={discardDraft}
                  >
                    {t.discardDraft}
                  </Pill>
                </div>
              </div>
            ) : (
              <div className={styles.empty}>{t.noDraft}</div>
            )}
          </Panel>

          <Panel label={t.memoryTitle} subtitle={t.memorySub}>
            <div className={styles.stack}>
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
          </Panel>

          <Panel label={t.inventoryTitle} subtitle={t.inventorySub}>
            <div className={styles.profileList}>
              {(inventory?.profiles ?? []).map((profile) => (
                <div key={profile.id} className={styles.profile}>
                  <div className={styles.profileName}>
                    {profile.display_name}
                  </div>
                  <div className={styles.profileMeta}>{profile.model_id}</div>
                  <div className={styles.limits}>
                    <span>
                      {t.concurrency}: {profile.concurrency_limit}
                    </span>
                    <span>
                      {t.rpm}: {profile.rpm_limit}
                    </span>
                    <span>
                      {t.tpm}: {profile.tpm_limit}
                    </span>
                    <span>
                      {t.retry}: {profile.retry_attempts}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}

function pickActiveRecipe(workspace: AgentWorkspace | null): AgentRecipe | null {
  if (!workspace) return null;
  for (const recipe of workspace.recipes) {
    if (
      sameSlot(recipe.stage_model_ids, workspace.stage_model_ids) &&
      sameSlot(recipe.stage_prompt_ids, workspace.stage_prompt_ids)
    ) {
      return recipe;
    }
  }
  return null;
}

function sameSlot(
  a: Record<string, string | null>,
  b: Record<string, string | null>,
): boolean {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  for (const key of keys) {
    if ((a[key] ?? null) !== (b[key] ?? null)) return false;
  }
  return true;
}
