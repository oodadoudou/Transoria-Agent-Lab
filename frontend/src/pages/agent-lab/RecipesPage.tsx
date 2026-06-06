import { useEffect, useMemo, useState } from "react";
import { agentBridge } from "@/bridge/client";
import type {
  AgentInventory,
  AgentInventoryPrompt,
  AgentModelSlot,
  AgentPromptSlot,
  AgentRecipe,
  AgentRecipeInput,
  AgentWorkspace,
  AgentWorkspaceResponse,
  PromptKind,
} from "@/bridge/types";
import { Panel } from "@/components/Panel";
import { Pill } from "@/components/Pill";
import { useMessages } from "@/locales";
import styles from "./RecipesPage.module.css";

const MODEL_SLOTS: ReadonlyArray<AgentModelSlot> = [
  "translation",
  "term_extract",
  "term_review",
];

const PROMPT_SLOTS: ReadonlyArray<{
  slot: AgentPromptSlot;
  kind: PromptKind;
}> = [
  { slot: "translation", kind: "translation" },
  { slot: "term_extract", kind: "glossary" },
  { slot: "term_review", kind: "glossary_review" },
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

interface ModalState {
  mode: "create" | "edit";
  recipeId: string | null;
  name: string;
  description: string;
  stage_model_ids: Record<AgentModelSlot, string | null>;
  stage_prompt_ids: Record<AgentPromptSlot, string | null>;
}

export function RecipesPage() {
  const messages = useMessages();
  const t = messages.agentLab;
  const [workspace, setWorkspace] = useState<AgentWorkspace | null>(null);
  const [inventory, setInventory] = useState<AgentInventory | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modal, setModal] = useState<ModalState | null>(null);

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

  const promptOptions = useMemo(() => {
    const empty: Record<PromptKind, AgentInventoryPrompt[]> = {
      translation: [],
      glossary: [],
      glossary_review: [],
    };
    if (!inventory) return empty;
    return { ...empty, ...inventory.prompts };
  }, [inventory]);

  const activeRecipeId = useMemo(
    () => pickActiveRecipeId(workspace),
    [workspace],
  );

  const openCreate = () =>
    setModal({
      mode: "create",
      recipeId: null,
      name: "",
      description: "",
      stage_model_ids: { ...EMPTY_MODEL_SLOTS },
      stage_prompt_ids: { ...EMPTY_PROMPT_SLOTS },
    });

  const openEdit = (recipe: AgentRecipe) =>
    setModal({
      mode: "edit",
      recipeId: recipe.id,
      name: recipe.name,
      description: recipe.description,
      stage_model_ids: { ...recipe.stage_model_ids },
      stage_prompt_ids: { ...recipe.stage_prompt_ids },
    });

  const closeModal = () => setModal(null);

  const saveModal = async () => {
    if (!modal) return;
    const name = modal.name.trim();
    if (!name) return;
    const payload: AgentRecipeInput = {
      name,
      description: modal.description.trim() || undefined,
      stage_model_ids: modal.stage_model_ids,
      stage_prompt_ids: modal.stage_prompt_ids,
    };
    if (modal.mode === "create") {
      await run(() => agentBridge.createRecipe(payload));
    } else if (modal.recipeId) {
      await run(() => agentBridge.updateRecipe(modal.recipeId!, payload));
    }
    setModal(null);
  };

  const applyRecipe = (id: string) =>
    void run(() => agentBridge.applyRecipe(id));

  const deleteRecipe = (id: string) => {
    if (!window.confirm(t.recipeDeleteConfirm)) return;
    void run(() => agentBridge.deleteRecipe(id));
  };

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
          <h1>{t.recipesPageTitle}</h1>
          <p>{t.recipesPageSub}</p>
        </div>
        <Pill disabled={busy || !workspace} onClick={openCreate}>
          {t.recipeNew}
        </Pill>
      </div>

      {error ? <div className={styles.error}>{error}</div> : null}

      <Panel label={t.activeStageTitle} subtitle={t.activeStageSub}>
        <div className={styles.stageSummary}>
          {MODEL_SLOTS.map((slot) => {
            const modelId = workspace?.stage_model_ids[slot] ?? null;
            const profile = modelId ? profileLookup.get(modelId) : null;
            return (
              <div key={`m-${slot}`} className={styles.stageRow}>
                <span className={styles.stageLabel}>{t.stageModel[slot]}</span>
                <span className={styles.stageValue}>
                  {profile?.display_name ?? t.stageEmptyModel}
                </span>
              </div>
            );
          })}
          {PROMPT_SLOTS.map(({ slot }) => {
            const promptId = workspace?.stage_prompt_ids[slot] ?? null;
            const prompt = promptId ? promptLookup.get(promptId) : null;
            return (
              <div key={`p-${slot}`} className={styles.stageRow}>
                <span className={styles.stageLabel}>{t.stagePrompt[slot]}</span>
                <span className={styles.stageValue}>
                  {formatPromptChoice(t, slot, prompt)}
                </span>
              </div>
            );
          })}
        </div>
      </Panel>

      <Panel label={t.recipesSectionTitle} subtitle={t.recipesSectionSub}>
        {workspace?.recipes.length ? (
          <div className={styles.recipeList}>
            {workspace.recipes.map((recipe) => {
              const isActive = recipe.id === activeRecipeId;
              return (
                <div
                  key={recipe.id}
                  className={`${styles.recipeItem} ${
                    isActive ? styles.recipeActive : ""
                  }`.trim()}
                >
                  <div className={styles.recipeMain}>
                    <div className={styles.recipeTitleRow}>
                      <span className={styles.recipeName}>{recipe.name}</span>
                      {isActive ? (
                        <span className={styles.activeBadge}>
                          {t.recipeApplied}
                        </span>
                      ) : null}
                    </div>
                    {recipe.description ? (
                      <p className={styles.recipeDescription}>
                        {recipe.description}
                      </p>
                    ) : null}
                    <div className={styles.recipeSlots}>
                      {MODEL_SLOTS.map((slot) => {
                        const profile = profileLookup.get(
                          recipe.stage_model_ids[slot] ?? "",
                        );
                        return (
                          <span
                            key={`m-${slot}`}
                            className={styles.recipeSlot}
                          >
                            <span className={styles.recipeSlotLabel}>
                              {t.stageModel[slot]}
                            </span>
                            <span>
                              {profile?.display_name ?? t.stageEmptyModel}
                            </span>
                          </span>
                        );
                      })}
                      {PROMPT_SLOTS.map(({ slot }) => {
                        const prompt = promptLookup.get(
                          recipe.stage_prompt_ids[slot] ?? "",
                        );
                        return (
                          <span
                            key={`p-${slot}`}
                            className={styles.recipeSlot}
                          >
                            <span className={styles.recipeSlotLabel}>
                              {t.stagePrompt[slot]}
                            </span>
                            <span>{formatPromptChoice(t, slot, prompt)}</span>
                          </span>
                        );
                      })}
                    </div>
                  </div>
                  <div className={styles.recipeActions}>
                    <button
                      type="button"
                      className={styles.linkButton}
                      disabled={busy || isActive}
                      onClick={() => applyRecipe(recipe.id)}
                    >
                      {t.recipeApply}
                    </button>
                    <button
                      type="button"
                      className={styles.linkButton}
                      disabled={busy}
                      onClick={() => openEdit(recipe)}
                    >
                      {t.recipeEdit}
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
              );
            })}
          </div>
        ) : (
          <div className={styles.empty}>{t.recipeEmpty}</div>
        )}
      </Panel>

      {modal ? (
        <RecipeModal
          state={modal}
          busy={busy}
          inventory={inventory}
          promptOptions={promptOptions}
          onChange={setModal}
          onSave={() => void saveModal()}
          onCancel={closeModal}
        />
      ) : null}
    </div>
  );
}

function RecipeModal({
  state,
  busy,
  inventory,
  promptOptions,
  onChange,
  onSave,
  onCancel,
}: {
  state: ModalState;
  busy: boolean;
  inventory: AgentInventory | null;
  promptOptions: Record<PromptKind, AgentInventoryPrompt[]>;
  onChange: (next: ModalState) => void;
  onSave: () => void;
  onCancel: () => void;
}) {
  const messages = useMessages();
  const t = messages.agentLab;
  const title =
    state.mode === "create" ? t.recipeModalTitleCreate : t.recipeModalTitleEdit;
  const profiles = inventory?.profiles ?? [];

  const setModel = (slot: AgentModelSlot, value: string) =>
    onChange({
      ...state,
      stage_model_ids: { ...state.stage_model_ids, [slot]: value || null },
    });

  const setPrompt = (slot: AgentPromptSlot, value: string) =>
    onChange({
      ...state,
      stage_prompt_ids: { ...state.stage_prompt_ids, [slot]: value || null },
    });

  return (
    <div className={styles.modalBackdrop} role="dialog" aria-modal="true">
      <div className={styles.modal}>
        <div className={styles.modalHeader}>
          <h2>{title}</h2>
        </div>
        <div className={styles.modalBody}>
          <label className={styles.field}>
            <span>{t.recipeFieldName}</span>
            <input
              className={styles.input}
              value={state.name}
              placeholder={t.recipeFieldNamePlaceholder}
              disabled={busy}
              autoFocus
              onChange={(event) =>
                onChange({ ...state, name: event.target.value })
              }
            />
          </label>
          <label className={styles.field}>
            <span>{t.recipeFieldDescription}</span>
            <textarea
              className={styles.textarea}
              value={state.description}
              placeholder={t.recipeFieldDescriptionPlaceholder}
              disabled={busy}
              onChange={(event) =>
                onChange({ ...state, description: event.target.value })
              }
            />
          </label>
          {MODEL_SLOTS.map((slot) => (
            <label key={`m-${slot}`} className={styles.field}>
              <span>{t.stageModel[slot]}</span>
              <select
                value={state.stage_model_ids[slot] ?? ""}
                disabled={busy}
                onChange={(event) => setModel(slot, event.target.value)}
              >
                <option value="">{t.stageEmptyModel}</option>
                {profiles.map((profile) => (
                  <option key={profile.id} value={profile.id}>
                    {profile.api_key_configured
                      ? profile.display_name
                      : `${profile.display_name} (${t.modelNotConfigured})`}
                  </option>
                ))}
              </select>
            </label>
          ))}
          {PROMPT_SLOTS.map(({ slot, kind }) => (
            <label key={`p-${slot}`} className={styles.field}>
              <span>{t.stagePrompt[slot]}</span>
              <select
                value={state.stage_prompt_ids[slot] ?? ""}
                disabled={busy}
                onChange={(event) => setPrompt(slot, event.target.value)}
              >
                <option value="">{t.stageEmptyPrompt}</option>
                {promptOptions[kind].map((prompt) => (
                  <option key={prompt.id} value={prompt.id}>
                    {formatPromptChoice(t, slot, prompt)}
                  </option>
                ))}
              </select>
            </label>
          ))}
        </div>
        <div className={styles.modalActions}>
          <Pill variant="ghost" disabled={busy} onClick={onCancel}>
            {t.recipeCancel}
          </Pill>
          <Pill disabled={busy || !state.name.trim()} onClick={onSave}>
            {t.recipeSave}
          </Pill>
        </div>
      </div>
    </div>
  );
}

function pickActiveRecipeId(workspace: AgentWorkspace | null): string | null {
  if (!workspace) return null;
  for (const recipe of workspace.recipes) {
    if (
      sameSlot(recipe.stage_model_ids, workspace.stage_model_ids) &&
      sameSlot(recipe.stage_prompt_ids, workspace.stage_prompt_ids)
    ) {
      return recipe.id;
    }
  }
  return null;
}

function formatPromptChoice(
  t: ReturnType<typeof useMessages>["agentLab"],
  slot: AgentPromptSlot,
  prompt: AgentInventoryPrompt | null | undefined,
): string {
  if (!prompt) return t.stageEmptyPrompt;
  return `${t.stagePrompt[slot]} · ${prompt.name}`;
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
