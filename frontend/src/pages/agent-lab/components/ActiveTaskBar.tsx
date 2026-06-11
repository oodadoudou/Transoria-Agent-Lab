import type { AgentActiveTask } from "@/bridge/types";
import type { useMessages } from "@/locales";
import styles from "../ChatPage.module.css";

type AgentLabMessages = ReturnType<typeof useMessages>["agentLab"];
type LocaleMessages = ReturnType<typeof useMessages>;

interface ActiveTaskBarProps {
  activeTask: AgentActiveTask;
  taskStatus: string | undefined;
  t: AgentLabMessages;
  messages: LocaleMessages;
  onOpenDashboard: () => void;
}

export function ActiveTaskBar({
  activeTask,
  taskStatus,
  t,
  messages,
  onOpenDashboard,
}: ActiveTaskBarProps) {
  return (
    <div className={styles.activeTaskBar}>
      <div className={styles.activeTaskMain}>
        <span className={styles.activeTaskBadge}>{t.activeTaskTitle}</span>
        <div>
          <strong>{t.activeTaskKind[activeTask.kind]}</strong>
          <p>
            {t.activeTaskStatus}: {formatTaskStatus(taskStatus, messages)}
            <span aria-hidden="true"> · </span>
            ID {activeTask.task_id}
            <span aria-hidden="true"> · </span>
            {t.activeTaskStartedAt} {formatDateTime(activeTask.started_at)}
          </p>
        </div>
      </div>
      <button
        type="button"
        className={styles.activeTaskButton}
        onClick={onOpenDashboard}
      >
        {t.activeTaskOpenDashboard}
      </button>
    </div>
  );
}

function formatDateTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function formatTaskStatus(
  status: string | undefined,
  messages: LocaleMessages,
): string {
  switch (status) {
    case "running":
      return messages.status.running;
    case "stopping":
    case "pausing":
      return messages.status.stopping;
    case "failed":
      return messages.status.failed;
    case "completed":
      return messages.status.completed;
    case "stopped":
    case "paused":
      return messages.status.stopped;
    case "pending":
    default:
      return messages.status.running;
  }
}
