import { useEffect, useState } from "react";
import { agentBridge } from "@/bridge/client";
import type {
  AgentInventory,
  AgentWorkspace,
  AgentWorkspaceResponse,
} from "@/bridge/types";
import {
  refreshModelProfilesFromResult,
  refreshPromptStoresFromResult,
  syncRuntimeTasksFromResponse,
} from "./chatRuntime";

type WorkspacePatch = Partial<
  Pick<
    AgentWorkspace,
    | "workflow_model_id"
    | "workflow_thinking_level"
    | "stage_model_ids"
    | "stage_prompt_ids"
  >
>;

interface UseAgentWorkspaceOptions {
  loadFailedText: string;
  saveFailedText: string;
}

export function useAgentWorkspace({
  loadFailedText,
  saveFailedText,
}: UseAgentWorkspaceOptions) {
  const [workspace, setWorkspace] = useState<AgentWorkspace | null>(null);
  const [inventory, setInventory] = useState<AgentInventory | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const applyResponse = (response: AgentWorkspaceResponse) => {
    setWorkspace(response.workspace);
    setInventory(response.inventory);
    syncRuntimeTasksFromResponse(response);
  };

  const run = async (
    action: () => Promise<AgentWorkspaceResponse>,
  ): Promise<boolean> => {
    setBusy(true);
    setError(null);
    try {
      const response = await action();
      applyResponse(response);
      await refreshPromptStoresFromResult(response);
      await refreshModelProfilesFromResult(response);
      return true;
    } catch (err) {
      setError(formatError(saveFailedText, err));
      return false;
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
        setError(formatError(loadFailedText, err));
      }
    })();
  }, [loadFailedText]);

  const sendMessage = async (rawText: string): Promise<boolean> => {
    const text = rawText.trim();
    if (!text) return false;
    setBusy(true);
    setError(null);
    try {
      applyResponse(await agentBridge.sendMessage(text));
      return true;
    } catch (err) {
      setError(formatError(saveFailedText, err));
      return false;
    } finally {
      setBusy(false);
    }
  };

  return {
    workspace,
    inventory,
    busy,
    error,
    setError,
    sendMessage,
    applyDraft: (draftId: string) =>
      run(() => agentBridge.applyDraft(draftId)),
    discardDraft: (draftId: string) =>
      run(() => agentBridge.discardDraft(draftId)),
    reviseDraft: (draftId: string, adjustment: string) =>
      run(() => agentBridge.reviseDraft(draftId, adjustment)),
    createConversation: () => run(() => agentBridge.createConversation()),
    switchConversation: (conversationId: string) =>
      run(() => agentBridge.switchConversation(conversationId)),
    renameConversation: (conversationId: string, title: string) =>
      run(() => agentBridge.renameConversation(conversationId, title)),
    deleteConversation: (conversationId: string) =>
      run(() => agentBridge.deleteConversation(conversationId)),
    updateMemory: (memories: string[]) =>
      run(() => agentBridge.updateMemory(memories)),
    deleteMemory: (memory: string) =>
      run(() => agentBridge.deleteMemory(memory)),
    applyRecipe: (recipeId: string) =>
      run(() => agentBridge.applyRecipe(recipeId)),
    updateWorkspace: (patch: WorkspacePatch) =>
      run(() => agentBridge.updateWorkspace(patch)),
  };
}

function formatError(prefix: string, err: unknown): string {
  return `${prefix} ${err instanceof Error ? err.message : ""}`.trim();
}
