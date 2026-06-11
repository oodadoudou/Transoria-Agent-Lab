import type { AgentActionDraft } from "@/bridge/types";
import { Pill } from "@/components/Pill";
import type { useMessages } from "@/locales";
import styles from "../ChatPage.module.css";
import { DraftConfirmationChecklist, DraftPreview } from "./DraftPreview";

type AgentLabMessages = ReturnType<typeof useMessages>["agentLab"];

interface DraftSectionProps {
  draft: AgentActionDraft;
  busy: boolean;
  t: AgentLabMessages;
  adjustingDraft: boolean;
  draftAdjustment: string;
  draftConfirmed: boolean;
  onDraftAdjustmentChange: (value: string) => void;
  onDraftConfirmedChange: (value: boolean) => void;
  onApplyDraft: () => void;
  onDiscardDraft: () => void;
  onStartAdjustDraft: () => void;
  onCancelDraftAdjustment: () => void;
  onSubmitDraftAdjustment: () => void | Promise<void>;
}

export function DraftSection({
  draft,
  busy,
  t,
  adjustingDraft,
  draftAdjustment,
  draftConfirmed,
  onDraftAdjustmentChange,
  onDraftConfirmedChange,
  onApplyDraft,
  onDiscardDraft,
  onStartAdjustDraft,
  onCancelDraftAdjustment,
  onSubmitDraftAdjustment,
}: DraftSectionProps) {
  return (
    <div className={styles.draft}>
      <div>
        <h3>{draft.title}</h3>
        <p>{draft.summary}</p>
      </div>
      <DraftConfirmationChecklist draft={draft} />
      <div className={styles.payloadLabel}>{t.draftPayload}</div>
      <DraftPreview draft={draft} />
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
          onChange={(event) => onDraftConfirmedChange(event.target.checked)}
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
          {draftConfirmed ? t.draftConfirmReadyLabel : t.draftConfirmLabel}
        </span>
      </label>
      <div className={styles.actions}>
        <Pill disabled={busy || !draftConfirmed} onClick={onApplyDraft}>
          {t.applyDraft}
        </Pill>
        <Pill variant="ghost" disabled={busy} onClick={onDiscardDraft}>
          {t.discardDraft}
        </Pill>
        <Pill variant="ghost" disabled={busy} onClick={onStartAdjustDraft}>
          {t.adjustDraft}
        </Pill>
      </div>
      {adjustingDraft ? (
        <div className={styles.adjustBox}>
          <label htmlFor="agent-draft-adjustment">{t.adjustDraftTitle}</label>
          <textarea
            id="agent-draft-adjustment"
            value={draftAdjustment}
            autoFocus
            placeholder={t.adjustDraftPlaceholder}
            disabled={busy}
            onChange={(event) => onDraftAdjustmentChange(event.target.value)}
            onKeyDown={(event) => {
              if (event.key !== "Enter") return;
              if (event.shiftKey || event.nativeEvent.isComposing) {
                return;
              }
              event.preventDefault();
              void onSubmitDraftAdjustment();
            }}
          />
          <div className={styles.adjustActions}>
            <Pill
              disabled={busy || !draftAdjustment.trim()}
              onClick={() => void onSubmitDraftAdjustment()}
            >
              {t.submitAdjustment}
            </Pill>
            <Pill
              variant="ghost"
              disabled={busy}
              onClick={onCancelDraftAdjustment}
            >
              {t.cancelAdjustment}
            </Pill>
          </div>
        </div>
      ) : null}
    </div>
  );
}
