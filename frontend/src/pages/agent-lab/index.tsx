import type { AgentLabPage } from "@/store/useTaskStore";
import { WorkspacePage } from "./WorkspacePage";

export function AgentLabModule({ page }: { page: AgentLabPage }) {
  if (page === "workspace") return <WorkspacePage />;
  return <WorkspacePage />;
}
