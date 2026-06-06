import { useEffect, useMemo, useState } from "react";
import { agentBridge } from "@/bridge/client";
import type {
  AgentInventory,
  AgentInventoryPrompt,
  AgentModelSlot,
  AgentPromptSlot,
  AgentWorkspace,
  AgentWorkspaceResponse,
  PromptKind,
} from "@/bridge/types";
import { Panel } from "@/components/Panel";
import { Pill } from "@/components/Pill";
import { useMessages } from "@/locales";
import styles from "./WorkspacePage.module.css";

const MODEL_SLOTS: Array<{ slot: AgentModelSlot; labelKey: string }> = [
  { slot: "translation", labelKey: "translationModel" },
  { slot: "term_extract", labelKey: "termExtractModel" },
  { slot: "term_review", labelKey: "termReviewModel" },
];

const PROMPT_SLOTS: Array<{
  slot: AgentPromptSlot;
  labelKey: string;
  kind: PromptKind;
}> = [
  { slot: "translation", labelKey: "translationPrompt", kind: "translation" },
  { slot: "term_extract", labelKey: "termExtractPrompt", kind: "glossary" },
  {
    slot: "term_review",
    labelKey: "termReviewPrompt",
    kind: "glossary_review",
  },
];

const EMPTY_MODEL_SLOTS: Record<AgentModelSlot, string | null> = {
  translation: null,
  term_extract: null,
  term_review: null,
};

const EMPTY_PROMPT_SLOTS: Record<AgentPromptSlot, string | null> = {
  translation: null,
  term_extract: null,
  term_review: null,
};

export function WorkspacePage() {
  const messages = useMessages();
  const t = messages.agentLab;
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
  const [newRecipeName, setNewRecipeName] = useState("");
  const [editingRecipeId, setEditingRecipeId] = useState<string | null>(null);
  const [editingRecipeName, setEditingRecipeName] = useState("");

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

  const load = async () => {
    applyResponse(await agentBridge.readWorkspace());
  };

  useEffect(() => {
    void load().catch((err: unknown) => {
      setError(
        `${t.loadFailed} ${err instanceof Error ? err.message : ""}`.trim(),
      );
    });
  }, [t.loadFailed]);

  const updateWorkspace = async (
    patch: Partial<
      Pick<
        AgentWorkspace,
        "workflow_model_id" | "stage_model_ids" | "stage_prompt_ids"
      >
    >,
  ) => {
    setBusy(true);
    setError(null);
    try {
      const response = await agentBridge.updateWorkspace(patch);
      setWorkspace(response.workspace);
      setInventory(response.inventory);
    } catch (err) {
      setError(
        `${t.saveFailed} ${err instanceof Error ? err.message : ""}`.trim(),
      );
    } finally {
      setBusy(false);
    }
  };

  const sendMessage = async () => {
    const text = input.trim();
    if (!text) return;
    setBusy(true);
    setError(null);
    setInput("");
    try {
      const response = await agentBridge.sendMessage(text);
      setWorkspace(response.workspace);
      setInventory(response.inventory);
    } catch (err) {
      setError(
        `${t.saveFailed} ${err instanceof Error ? err.message : ""}`.trim(),
      );
      setInput(text);
    } finally {
      setBusy(false);
    }
  };

  const applyDraft = async () => {
    if (!workspace?.pending_draft) return;
    setBusy(true);
    setError(null);
    try {
      const response = await agentBridge.applyDraft(workspace.pending_draft.id);
      setWorkspace(response.workspace);
      setInventory(response.inventory);
    } catch (err) {
      setError(
        `${t.saveFailed} ${err instanceof Error ? err.message : ""}`.trim(),
      );
    } finally {
      setBusy(false);
    }
  };

  const discardDraft = async () => {
    if (!workspace?.pending_draft) return;
    setBusy(true);
    setError(null);
    try {
      const response = await agentBridge.discardDraft(
        workspace.pending_draft.id,
      );
      setWorkspace(response.workspace);
      setInventory(response.inventory);
    } catch (err) {
      setError(
        `${t.saveFailed} ${err instanceof Error ? err.message : ""}`.trim(),
      );
    } finally {
      setBusy(false);
    }
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

  const saveCurrentAsRecipe = async () => {
    const name = newRecipeName.trim();
    if (!name) return;
    setNewRecipeName("");
    await run(() =>
      agentBridge.createRecipe({
        name,
        stage_model_ids: workspace?.stage_model_ids,
        stage_prompt_ids: workspace?.stage_prompt_ids,
      }),
    );
  };

  const applyRecipe = (id: string) =>
    void run(() => agentBridge.applyRecipe(id));

  const updateRecipeStages = (id: string) =>
    void run(() =>
      agentBridge.updateRecipe(id, {
        stage_model_ids: workspace?.stage_model_ids,
        stage_prompt_ids: workspace?.stage_prompt_ids,
      }),
    );

  const deleteRecipe = (id: string) =>
    void run(() => agentBridge.deleteRecipe(id));

  const startRecipeRename = (id: string, name: string) => {
    setEditingRecipeId(id);
    setEditingRecipeName(name);
  };

  const commitRecipeRename = async () => {
    const id = editingRecipeId;
    const name = editingRecipeName.trim();
    setEditingRecipeId(null);
    if (!id || !name) return;
    await run(() => agentBridge.updateRecipe(id, { name }));
  };

  const promptOptions = useMemo(() => {
    const empty: Record<PromptKind, AgentInventoryPrompt[]> = {
      translation: [],
      glossary: [],
      glossary_review: [],
    };
    if (!inventory) return empty;
    return {
      ...empty,
      ...inventory.prompts,
    };
  }, [inventory]);

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

          <Panel label={t.configTitle} subtitle={t.configSub}>
            <div className={styles.stack}>
              <SelectField
                label={t.workflowModel}
                value={workspace?.workflow_model_id ?? ""}
                emptyLabel={t.noModel}
                disabled={busy || !workspace || !inventory}
                options={(inventory?.profiles ?? []).map((profile) => ({
                  value: profile.id,
                  label: formatProfile(
                    profile.display_name,
                    profile.api_key_configured,
                    t.modelNotConfigured,
                  ),
                }))}
                onChange={(value) =>
                  void updateWorkspace({ workflow_model_id: value || null })
                }
              />

              <div className={styles.boundary}>{t.confirmationBoundary}</div>

              {MODEL_SLOTS.map(({ slot, labelKey }) => (
                <SelectField
                  key={slot}
                  label={t[labelKey as keyof typeof t] as string}
                  value={workspace?.stage_model_ids[slot] ?? ""}
                  emptyLabel={t.noModel}
                  disabled={busy || !workspace || !inventory}
                  options={(inventory?.profiles ?? []).map((profile) => ({
                    value: profile.id,
                    label: formatProfile(
                      profile.display_name,
                      profile.api_key_configured,
                      t.modelNotConfigured,
                    ),
                  }))}
                  onChange={(value) =>
                    void updateWorkspace({
                      stage_model_ids: {
                        ...(workspace?.stage_model_ids ?? EMPTY_MODEL_SLOTS),
                        [slot]: value || null,
                      },
                    })
                  }
                />
              ))}

              {PROMPT_SLOTS.map(({ slot, labelKey, kind }) => (
                <SelectField
                  key={slot}
                  label={t[labelKey as keyof typeof t] as string}
                  value={workspace?.stage_prompt_ids[slot] ?? ""}
                  emptyLabel={t.noPrompt}
                  disabled={busy || !workspace || !inventory}
                  options={promptOptions[kind].map((prompt) => ({
                    value: prompt.id,
                    label: prompt.name,
                  }))}
                  onChange={(value) =>
                    void updateWorkspace({
                      stage_prompt_ids: {
                        ...(workspace?.stage_prompt_ids ?? EMPTY_PROMPT_SLOTS),
                        [slot]: value || null,
                      },
                    })
                  }
                />
              ))}
            </div>
          </Panel>

          <Panel label={t.recipesTitle} subtitle={t.recipesSub}>
            <div className={styles.stack}>
              <div className={styles.memoryAdd}>
                <input
                  className={styles.inlineInput}
                  value={newRecipeName}
                  placeholder={t.recipeNamePlaceholder}
                  disabled={busy || !workspace}
                  onChange={(event) => setNewRecipeName(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      event.preventDefault();
                      void saveCurrentAsRecipe();
                    }
                  }}
                />
                <Pill
                  disabled={busy || !newRecipeName.trim()}
                  onClick={() => void saveCurrentAsRecipe()}
                >
                  {t.recipeSaveCurrent}
                </Pill>
              </div>
              {workspace?.recipes.length ? (
                <div className={styles.convList}>
                  {workspace.recipes.map((recipe) => (
                    <div key={recipe.id} className={styles.convItem}>
                      {editingRecipeId === recipe.id ? (
                        <input
                          className={styles.inlineInput}
                          value={editingRecipeName}
                          autoFocus
                          disabled={busy}
                          onChange={(event) =>
                            setEditingRecipeName(event.target.value)
                          }
                          onBlur={() => void commitRecipeRename()}
                          onKeyDown={(event) => {
                            if (event.key === "Enter") {
                              event.preventDefault();
                              void commitRecipeRename();
                            } else if (event.key === "Escape") {
                              setEditingRecipeId(null);
                            }
                          }}
                        />
                      ) : (
                        <div className={styles.convName}>{recipe.name}</div>
                      )}
                      <div className={styles.convActions}>
                        <button
                          type="button"
                          className={styles.linkButton}
                          disabled={busy}
                          onClick={() => applyRecipe(recipe.id)}
                        >
                          {t.recipeApply}
                        </button>
                        <button
                          type="button"
                          className={styles.linkButton}
                          disabled={busy}
                          onClick={() => updateRecipeStages(recipe.id)}
                        >
                          {t.recipeUpdateStages}
                        </button>
                        <button
                          type="button"
                          className={styles.linkButton}
                          disabled={busy}
                          onClick={() =>
                            startRecipeRename(recipe.id, recipe.name)
                          }
                        >
                          {t.recipeRename}
                        </button>
                        <button
                          type="button"
                          className={styles.linkButton}
                          disabled={busy}
                          onClick={() => deleteRecipe(recipe.id)}
                        >
                          {t.recipeDelete}
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className={styles.empty}>{t.recipeEmpty}</div>
              )}
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
                  <Pill disabled={busy} onClick={() => void applyDraft()}>
                    {t.applyDraft}
                  </Pill>
                  <Pill
                    variant="ghost"
                    disabled={busy}
                    onClick={() => void discardDraft()}
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

function SelectField({
  label,
  value,
  emptyLabel,
  disabled,
  options,
  onChange,
}: {
  label: string;
  value: string;
  emptyLabel: string;
  disabled: boolean;
  options: Array<{ value: string; label: string }>;
  onChange: (value: string) => void;
}) {
  return (
    <label className={styles.selectField}>
      <span>{label}</span>
      <select
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
      >
        <option value="">{emptyLabel}</option>
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function formatProfile(
  name: string,
  configured: boolean,
  missingLabel: string,
) {
  return configured ? name : `${name} (${missingLabel})`;
}
