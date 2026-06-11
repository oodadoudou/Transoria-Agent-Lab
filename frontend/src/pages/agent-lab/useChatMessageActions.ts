import { useRef, useState } from "react";

type SendWorkspaceMessage = (text: string) => Promise<boolean>;

interface UseChatMessageActionsOptions {
  sendWorkspaceMessage: SendWorkspaceMessage;
  setError: (message: string | null) => void;
}

export function useChatMessageActions({
  sendWorkspaceMessage,
  setError,
}: UseChatMessageActionsOptions) {
  const [input, setInput] = useState("");
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [processExpanded, setProcessExpanded] = useState(false);
  const messagesRef = useRef<HTMLDivElement | null>(null);
  const [expandedMessages, setExpandedMessages] = useState<ReadonlySet<string>>(
    () => new Set(),
  );

  const sendText = async (rawText: string, restoreOnError = false) => {
    const text = rawText.trim();
    if (!text) return;
    if (restoreOnError) {
      setInput("");
    }
    const ok = await sendWorkspaceMessage(text);
    if (!ok && restoreOnError) {
      setInput(text);
    }
  };

  const sendMessage = async () => {
    await sendText(input, true);
  };

  const resendMessage = (content: string) => {
    void sendText(content);
  };

  const copyMessage = async (id: string, content: string) => {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(content);
      } else {
        fallbackCopy(content);
      }
      setCopiedMessageId(id);
      window.setTimeout(() => {
        setCopiedMessageId((current) => (current === id ? null : current));
      }, 1400);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const toggleMessageExpanded = (id: string) => {
    setExpandedMessages((current) => {
      const next = new Set(current);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  return {
    input,
    setInput,
    copiedMessageId,
    expandedMessages,
    messagesRef,
    processExpanded,
    sendMessage,
    resendMessage,
    copyMessage,
    toggleMessageExpanded,
    toggleProcessExpanded: () => setProcessExpanded((expanded) => !expanded),
  };
}

function fallbackCopy(text: string): void {
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "true");
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  textarea.select();
  try {
    document.execCommand("copy");
  } finally {
    document.body.removeChild(textarea);
  }
}
