import type { Ref } from "react";
import type { AgentWorkspace } from "@/bridge/types";
import styles from "../ChatPage.module.css";

const COLLAPSE_MESSAGE_CHARS = 900;

type ChatMessage = AgentWorkspace["messages"][number];

type ChatMessageListText = {
  collapseMessage: string;
  expandMessage: string;
  resendMessage: string;
  copyMessage: string;
  copiedMessage: string;
  quickActions: readonly string[];
  thinkingTitle: string;
  thinkingLead: string;
  hideProcessDetails: string;
  showProcessDetails: string;
  thinkingSteps: readonly string[];
  processDisclosure: string;
};

type ChatMessageListProps = {
  messages: readonly ChatMessage[];
  busy: boolean;
  copiedMessageId: string | null;
  expandedMessages: ReadonlySet<string>;
  messagesRef: Ref<HTMLDivElement>;
  processExpanded: boolean;
  t: ChatMessageListText;
  onCopyMessage: (id: string, content: string) => void | Promise<void>;
  onQuickAction: (text: string) => void;
  onResendMessage: (content: string) => void;
  onToggleMessageExpanded: (id: string) => void;
  onToggleProcessExpanded: () => void;
};

export function ChatMessageList({
  messages,
  busy,
  copiedMessageId,
  expandedMessages,
  messagesRef,
  processExpanded,
  t,
  onCopyMessage,
  onQuickAction,
  onResendMessage,
  onToggleMessageExpanded,
  onToggleProcessExpanded,
}: ChatMessageListProps) {
  return (
    <div className={styles.messages} ref={messagesRef}>
      {messages.map((message) => (
        <ChatMessageItem
          key={message.id}
          message={message}
          busy={busy}
          copied={copiedMessageId === message.id}
          expanded={expandedMessages.has(message.id)}
          t={t}
          onCopyMessage={onCopyMessage}
          onResendMessage={onResendMessage}
          onToggleMessageExpanded={onToggleMessageExpanded}
        />
      ))}
      {messages.length <= 1 ? (
        <div className={styles.quickActions}>
          {t.quickActions.map((action) => (
            <button
              key={action}
              type="button"
              disabled={busy}
              onClick={() => onQuickAction(action)}
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
            onClick={onToggleProcessExpanded}
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
              <p className={styles.processDisclosure}>{t.processDisclosure}</p>
            </>
          ) : null}
        </div>
      ) : null}
      <div className={styles.messagesEnd} />
    </div>
  );
}

function ChatMessageItem({
  message,
  busy,
  copied,
  expanded,
  t,
  onCopyMessage,
  onResendMessage,
  onToggleMessageExpanded,
}: {
  message: ChatMessage;
  busy: boolean;
  copied: boolean;
  expanded: boolean;
  t: ChatMessageListText;
  onCopyMessage: (id: string, content: string) => void | Promise<void>;
  onResendMessage: (content: string) => void;
  onToggleMessageExpanded: (id: string) => void;
}) {
  const isLong = message.content.length > COLLAPSE_MESSAGE_CHARS;
  const body =
    isLong && !expanded
      ? `${message.content.slice(0, COLLAPSE_MESSAGE_CHARS).trimEnd()}…`
      : message.content;

  return (
    <div
      className={`${styles.message} ${
        message.role === "user" ? styles.userMessage : styles.agentMessage
      }`}
    >
      <div className={styles.messageRole}>{message.role}</div>
      <div className={styles.messageBody}>{body}</div>
      <div className={styles.messageActions}>
        {isLong ? (
          <button
            type="button"
            className={styles.messageActionButton}
            onClick={() => onToggleMessageExpanded(message.id)}
          >
            {expanded ? t.collapseMessage : t.expandMessage}
          </button>
        ) : null}
        {message.role === "user" ? (
          <button
            type="button"
            className={styles.messageActionButton}
            disabled={busy}
            onClick={() => onResendMessage(message.content)}
          >
            {t.resendMessage}
          </button>
        ) : null}
        <button
          type="button"
          className={styles.messageActionButton}
          disabled={busy}
          onClick={() => void onCopyMessage(message.id, message.content)}
        >
          {copied ? t.copiedMessage : t.copyMessage}
        </button>
      </div>
    </div>
  );
}
