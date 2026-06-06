import type { AgentLabPage } from "@/store/useTaskStore";
import { ChatPage } from "./ChatPage";
import { RecipesPage } from "./RecipesPage";

export function AgentLabModule({ page }: { page: AgentLabPage }) {
  switch (page) {
    case "chat":
      return <ChatPage />;
    case "recipes":
      return <RecipesPage />;
  }
}
