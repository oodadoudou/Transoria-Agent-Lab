import { useEffect, useState } from "react";
import type { AgentWorkspace } from "@/bridge/types";

interface UseDraftControlsOptions {
  workspace: AgentWorkspace | null;
  applyWorkspaceDraft: (draftId: string) => Promise<boolean>;
  discardWorkspaceDraft: (draftId: string) => Promise<boolean>;
  reviseDraft: (draftId: string, adjustment: string) => Promise<boolean>;
}

export function useDraftControls({
  workspace,
  applyWorkspaceDraft,
  discardWorkspaceDraft,
  reviseDraft,
}: UseDraftControlsOptions) {
  const [adjustingDraft, setAdjustingDraft] = useState(false);
  const [draftAdjustment, setDraftAdjustment] = useState("");
  const [draftConfirmed, setDraftConfirmed] = useState(false);

  useEffect(() => {
    setDraftConfirmed(false);
  }, [workspace?.pending_draft?.id]);

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

  return {
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
  };
}
