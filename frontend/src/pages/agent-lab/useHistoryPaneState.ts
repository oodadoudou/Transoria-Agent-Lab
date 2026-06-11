import { useState } from "react";
import type { AgentWorkspace } from "@/bridge/types";

interface UseHistoryPaneStateOptions {
  workspace: AgentWorkspace | null;
  createWorkspaceConversation: () => Promise<boolean>;
  switchWorkspaceConversation: (conversationId: string) => Promise<boolean>;
  renameConversation: (
    conversationId: string,
    title: string,
  ) => Promise<boolean>;
  deleteWorkspaceConversation: (conversationId: string) => Promise<boolean>;
  updateMemory: (memories: string[]) => Promise<boolean>;
  deleteWorkspaceMemory: (memory: string) => Promise<boolean>;
}

export function useHistoryPaneState({
  workspace,
  createWorkspaceConversation,
  switchWorkspaceConversation,
  renameConversation,
  deleteWorkspaceConversation,
  updateMemory,
  deleteWorkspaceMemory,
}: UseHistoryPaneStateOptions) {
  const [editingConvId, setEditingConvId] = useState<string | null>(null);
  const [editingConvTitle, setEditingConvTitle] = useState("");
  const [newMemory, setNewMemory] = useState("");
  const [editingMemoryIndex, setEditingMemoryIndex] = useState<number | null>(
    null,
  );
  const [editingMemoryText, setEditingMemoryText] = useState("");
  const [historyCollapsed, setHistoryCollapsed] = useState(false);

  const createConversation = () => createWorkspaceConversation();

  const switchConversation = (id: string) => {
    if (id === workspace?.active_conversation_id) return;
    void switchWorkspaceConversation(id);
  };

  const deleteConversation = (id: string) =>
    void deleteWorkspaceConversation(id);

  const startRename = (id: string, title: string) => {
    setEditingConvId(id);
    setEditingConvTitle(title);
  };

  const commitRename = async () => {
    const id = editingConvId;
    const title = editingConvTitle.trim();
    setEditingConvId(null);
    if (!id || !title) return;
    await renameConversation(id, title);
  };

  const addMemory = async () => {
    const text = newMemory.trim();
    if (!text) return;
    setNewMemory("");
    await updateMemory([...(workspace?.memories ?? []), text]);
  };

  const deleteMemory = (memory: string) => void deleteWorkspaceMemory(memory);

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
    await updateMemory(next);
  };

  return {
    editingConvId,
    editingConvTitle,
    newMemory,
    editingMemoryIndex,
    editingMemoryText,
    historyCollapsed,
    setEditingConvTitle,
    setNewMemory,
    setEditingMemoryText,
    toggleHistoryCollapsed: () =>
      setHistoryCollapsed((collapsed) => !collapsed),
    createConversation,
    switchConversation,
    deleteConversation,
    startRename,
    commitRename,
    cancelRename: () => setEditingConvId(null),
    addMemory,
    deleteMemory,
    startEditMemory,
    commitMemoryEdit,
    cancelMemoryEdit: () => setEditingMemoryIndex(null),
  };
}
