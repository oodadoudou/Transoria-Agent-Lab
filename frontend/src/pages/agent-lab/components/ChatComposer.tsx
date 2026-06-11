import type {
  AgentInventory,
  AgentRecipe,
  AgentWorkspace,
  ThinkingLevel,
} from "@/bridge/types";
import { Pill } from "@/components/Pill";
import type { useMessages } from "@/locales";
import styles from "../ChatPage.module.css";

type AgentLabMessages = ReturnType<typeof useMessages>["agentLab"];
type ThinkingMessages = ReturnType<typeof useMessages>["model"]["thinking"];

const THINKING_LEVELS: ReadonlyArray<ThinkingLevel> = [
  "off",
  "low",
  "medium",
  "high",
];

interface ChatComposerProps {
  input: string;
  busy: boolean;
  workspace: AgentWorkspace | null;
  inventory: AgentInventory | null;
  activeRecipe: AgentRecipe | null;
  workflowModelLabel: string;
  activeRecipeLabel: string;
  workflowThinkingEnabled: boolean;
  thinkingLevel: ThinkingLevel;
  t: AgentLabMessages;
  thinkingLabels: ThinkingMessages;
  modelOpen: boolean;
  recipeOpen: boolean;
  reasoningOpen: boolean;
  onInputChange: (value: string) => void;
  onSend: () => void | Promise<void>;
  onToggleModelOpen: () => void;
  onToggleRecipeOpen: () => void;
  onToggleReasoningOpen: () => void;
  onUpdateWorkflowModel: (value: string) => void;
  onUpdateWorkflowThinking: (value: ThinkingLevel) => void;
  onSwitchRecipe: (id: string) => void;
}

export function ChatComposer({
  input,
  busy,
  workspace,
  inventory,
  activeRecipe,
  workflowModelLabel,
  activeRecipeLabel,
  workflowThinkingEnabled,
  thinkingLevel,
  t,
  thinkingLabels,
  modelOpen,
  recipeOpen,
  reasoningOpen,
  onInputChange,
  onSend,
  onToggleModelOpen,
  onToggleRecipeOpen,
  onToggleReasoningOpen,
  onUpdateWorkflowModel,
  onUpdateWorkflowThinking,
  onSwitchRecipe,
}: ChatComposerProps) {
  return (
    <div className={styles.composer}>
      <textarea
        value={input}
        placeholder={t.inputPlaceholder}
        disabled={busy}
        onChange={(event) => onInputChange(event.target.value)}
        onKeyDown={(event) => {
          if (event.key !== "Enter") return;
          if (event.shiftKey || event.nativeEvent.isComposing) return;
          event.preventDefault();
          void onSend();
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
              onClick={onToggleModelOpen}
            >
              <span>{shortLabel(workflowModelLabel)}</span>
              <span aria-hidden="true">⌄</span>
            </button>
            {modelOpen ? (
              <div className={styles.compactPopup} role="menu">
                <button
                  type="button"
                  role="menuitemradio"
                  aria-checked={!workspace?.workflow_model_id}
                  className={styles.compactOption}
                  onClick={() => onUpdateWorkflowModel("")}
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
                    onClick={() => onUpdateWorkflowModel(profile.id)}
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
              onClick={onToggleReasoningOpen}
            >
              {thinkingLabels[thinkingLevel]}
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
                    onClick={() => onUpdateWorkflowThinking(level)}
                  >
                    <span>{thinkingLabels[level]}</span>
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
              onClick={onToggleRecipeOpen}
            >
              <span>{shortLabel(activeRecipeLabel)}</span>
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
                      onClick={() => onSwitchRecipe(recipe.id)}
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

          <Pill disabled={busy || !input.trim()} onClick={() => void onSend()}>
            {busy ? t.sending : t.send}
          </Pill>
        </div>
      </div>
    </div>
  );
}

function shortLabel(label: string): string {
  const trimmed = label.trim();
  if (trimmed.length <= 14) return trimmed;
  return `${trimmed.slice(0, 13)}…`;
}
