import type { useMessages } from "@/locales";
import type { AgentWorkspace } from "@/bridge/types";
import { Pill } from "@/components/Pill";
import styles from "../ChatPage.module.css";

type AgentLabMessages = ReturnType<typeof useMessages>["agentLab"];

interface HistoryPaneProps {
  workspace: AgentWorkspace | null;
  busy: boolean;
  t: AgentLabMessages;
  collapsed: boolean;
  editingConversationId: string | null;
  editingConversationTitle: string;
  newMemory: string;
  editingMemoryIndex: number | null;
  editingMemoryText: string;
  onToggleCollapsed: () => void;
  onCreateConversation: () => void;
  onSwitchConversation: (id: string) => void;
  onStartRename: (id: string, title: string) => void;
  onCommitRename: () => void | Promise<void>;
  onCancelRename: () => void;
  onEditingConversationTitleChange: (value: string) => void;
  onDeleteConversation: (id: string) => void;
  onNewMemoryChange: (value: string) => void;
  onAddMemory: () => void | Promise<void>;
  onStartEditMemory: (index: number, memory: string) => void;
  onEditingMemoryTextChange: (value: string) => void;
  onCommitMemoryEdit: () => void | Promise<void>;
  onCancelMemoryEdit: () => void;
  onDeleteMemory: (memory: string) => void;
}

export function HistoryPane({
  workspace,
  busy,
  t,
  collapsed,
  editingConversationId,
  editingConversationTitle,
  newMemory,
  editingMemoryIndex,
  editingMemoryText,
  onToggleCollapsed,
  onCreateConversation,
  onSwitchConversation,
  onStartRename,
  onCommitRename,
  onCancelRename,
  onEditingConversationTitleChange,
  onDeleteConversation,
  onNewMemoryChange,
  onAddMemory,
  onStartEditMemory,
  onEditingMemoryTextChange,
  onCommitMemoryEdit,
  onCancelMemoryEdit,
  onDeleteMemory,
}: HistoryPaneProps) {
  return (
    <aside
      className={`${styles.historyPane} ${
        collapsed ? styles.historyPaneCollapsed : ""
      }`.trim()}
    >
      {collapsed ? (
        <div className={styles.collapsedHistory}>
          <button
            type="button"
            className={styles.historyToggle}
            aria-label={t.expandHistory}
            onClick={onToggleCollapsed}
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
              onClick={onToggleCollapsed}
            >
              −
            </button>
          </div>

          <button
            type="button"
            className={styles.newConversationInline}
            disabled={busy}
            onClick={onCreateConversation}
          >
            {t.newConversation}
          </button>

          <div className={styles.convList}>
            {(workspace?.conversations ?? []).map((conversation) => {
              const isActive =
                conversation.id === workspace?.active_conversation_id;
              const isEditing = conversation.id === editingConversationId;
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
                      value={editingConversationTitle}
                      autoFocus
                      disabled={busy}
                      onChange={(event) =>
                        onEditingConversationTitleChange(event.target.value)
                      }
                      onBlur={() => void onCommitRename()}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") {
                          event.preventDefault();
                          void onCommitRename();
                        } else if (event.key === "Escape") {
                          onCancelRename();
                        }
                      }}
                    />
                  ) : (
                    <button
                      type="button"
                      className={styles.convTitle}
                      disabled={busy}
                      onClick={() => onSwitchConversation(conversation.id)}
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
                          onStartRename(conversation.id, conversation.title)
                        }
                      >
                        {t.renameConversation}
                      </button>
                      <button
                        type="button"
                        className={styles.linkButton}
                        disabled={busy}
                        onClick={() => onDeleteConversation(conversation.id)}
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
                  <li key={`${index}-${memory}`} className={styles.memoryItem}>
                    {editingMemoryIndex === index ? (
                      <input
                        className={styles.inlineInput}
                        value={editingMemoryText}
                        autoFocus
                        disabled={busy}
                        onChange={(event) =>
                          onEditingMemoryTextChange(event.target.value)
                        }
                        onBlur={() => void onCommitMemoryEdit()}
                        onKeyDown={(event) => {
                          if (event.key === "Enter") {
                            event.preventDefault();
                            void onCommitMemoryEdit();
                          } else if (event.key === "Escape") {
                            onCancelMemoryEdit();
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
                            onClick={() => onStartEditMemory(index, memory)}
                          >
                            {t.editMemory}
                          </button>
                          <button
                            type="button"
                            className={styles.linkButton}
                            disabled={busy}
                            onClick={() => onDeleteMemory(memory)}
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
                onChange={(event) => onNewMemoryChange(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    void onAddMemory();
                  }
                }}
              />
              <Pill disabled={busy || !newMemory.trim()} onClick={onAddMemory}>
                {t.memoryAdd}
              </Pill>
            </div>
          </div>
        </>
      )}
    </aside>
  );
}
