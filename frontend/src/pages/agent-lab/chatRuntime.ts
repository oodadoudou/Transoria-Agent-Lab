import type {
  AgentTaskKind,
  AgentWorkspaceResponse,
  PromptKind,
} from "@/bridge/types";
import { useModelProfilesStore } from "@/store/useModelProfilesStore";
import { usePromptPresetsStore } from "@/store/usePromptPresetsStore";
import { useRuntimeStore, type RunKind } from "@/store/useRuntimeStore";

export async function refreshPromptStoresFromResult(
  response: AgentWorkspaceResponse,
): Promise<void> {
  const kinds = promptKindsFromResult(response.result);
  if (!kinds.size) return;
  const store = usePromptPresetsStore.getState();
  await Promise.all([...kinds].map((kind) => store.refresh(kind)));
}

export async function refreshModelProfilesFromResult(
  response: AgentWorkspaceResponse,
): Promise<void> {
  if (!resultIncludesModelProfileChange(response.result)) return;
  await useModelProfilesStore.getState().refresh();
}

export function syncRuntimeTasksFromResponse(
  response: AgentWorkspaceResponse,
): void {
  const tasks = startedTasksFromResult(response.result);
  const activeTask = response.workspace.active_task;
  if (activeTask) {
    tasks.push({
      kind: activeTask.kind,
      taskId: activeTask.task_id,
    });
  }
  if (!tasks.length) return;

  const runtime = useRuntimeStore.getState();
  const seen = new Set<string>();
  for (const task of tasks) {
    const runKind = runKindFromAgentTaskKind(task.kind);
    if (!runKind) continue;
    const key = `${runKind}:${task.taskId}`;
    if (seen.has(key)) continue;
    seen.add(key);
    runtime.setActiveTaskId(runKind, task.taskId);
    void runtime.pollSnapshot(runKind);
  }
}

function promptKindsFromResult(
  result: Record<string, unknown> | undefined,
): Set<PromptKind> {
  const kinds = new Set<PromptKind>();
  collectPromptKinds(result, kinds);
  return kinds;
}

function collectPromptKinds(value: unknown, kinds: Set<PromptKind>): void {
  if (Array.isArray(value)) {
    value.forEach((item) => collectPromptKinds(item, kinds));
    return;
  }
  if (!isRecord(value)) return;
  const preset = value.preset;
  if (isRecord(preset) && isPromptKind(preset.kind)) {
    kinds.add(preset.kind);
  }
  Object.values(value).forEach((item) => collectPromptKinds(item, kinds));
}

function resultIncludesModelProfileChange(value: unknown): boolean {
  if (Array.isArray(value)) {
    return value.some((item) => resultIncludesModelProfileChange(item));
  }
  if (!isRecord(value)) return false;
  if (
    value.kind === "create_model_profile" ||
    value.kind === "update_model_profile"
  ) {
    return true;
  }
  if (isRecord(value.profile) && typeof value.profile.id === "string") {
    return true;
  }
  return Object.values(value).some((item) =>
    resultIncludesModelProfileChange(item),
  );
}

function startedTasksFromResult(
  result: Record<string, unknown> | undefined,
): Array<{ kind: AgentTaskKind; taskId: string }> {
  const tasks: Array<{ kind: AgentTaskKind; taskId: string }> = [];
  collectStartedTasks(result, tasks);
  return tasks;
}

function collectStartedTasks(
  value: unknown,
  tasks: Array<{ kind: AgentTaskKind; taskId: string }>,
): void {
  if (Array.isArray(value)) {
    value.forEach((item) => collectStartedTasks(item, tasks));
    return;
  }
  if (!isRecord(value)) return;

  const task = value.task;
  if (isRecord(task) && isAgentTaskKind(task.kind)) {
    const taskId = task.task_id;
    if (typeof taskId === "string" && taskId.trim()) {
      tasks.push({ kind: task.kind, taskId });
    }
  }

  Object.values(value).forEach((item) => collectStartedTasks(item, tasks));
}

function runKindFromAgentTaskKind(kind: AgentTaskKind): RunKind {
  return kind;
}

function isAgentTaskKind(value: unknown): value is AgentTaskKind {
  return (
    value === "translation" ||
    value === "glossary" ||
    value === "glossary_review"
  );
}

function isPromptKind(value: unknown): value is PromptKind {
  return (
    value === "translation" ||
    value === "glossary" ||
    value === "glossary_review"
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
