import type { AgentActionDraft } from "@/bridge/types";
import styles from "../ChatPage.module.css";

const DRAFT_TEXT_PREVIEW_CHARS = 520;

export function DraftPreview({ draft }: { draft: AgentActionDraft }) {
  if (draft.kind === "compound_config_update") {
    const actions = Array.isArray(draft.payload.actions)
      ? draft.payload.actions.filter(isRecord)
      : [];
    return (
      <div className={styles.draftPreview}>
        <div className={styles.draftSummary}>
          <span className={styles.draftBadge}>{formatDraftKind(draft.kind)}</span>
          <span>
            {actions.length
              ? `包含 ${actions.length} 个待确认修改`
              : "复合配置修改"}
          </span>
        </div>
        {actions.length ? (
          <div className={styles.draftActionList}>
            {actions.map((action, index) => (
              <DraftActionPreview key={index} action={action} index={index} />
            ))}
          </div>
        ) : (
          <DraftFields payload={draft.payload} />
        )}
        <RawDraftDetails payload={draft.payload} />
      </div>
    );
  }

  return (
    <div className={styles.draftPreview}>
      <div className={styles.draftSummary}>
        <span className={styles.draftBadge}>{formatDraftKind(draft.kind)}</span>
        <span>{draft.title}</span>
      </div>
      <DraftFields payload={draft.payload} />
      <RawDraftDetails payload={draft.payload} />
    </div>
  );
}

export function DraftConfirmationChecklist({
  draft,
}: {
  draft: AgentActionDraft;
}) {
  const rows = draftConfirmationRows(draft);
  return (
    <div className={styles.draftChecklist} aria-label="草案确认要点">
      {rows.map((row) => (
        <div key={row} className={styles.draftChecklistRow}>
          <span className={styles.draftCheckMark} aria-hidden="true" />
          <span>{row}</span>
        </div>
      ))}
    </div>
  );
}

function draftConfirmationRows(draft: AgentActionDraft): string[] {
  const rows: string[] = [];
  const actionCount = draftActionCount(draft);
  if (actionCount > 1) {
    rows.push(`多步草案：包含 ${actionCount} 个动作，会按显示顺序执行。`);
  }
  if (draftStartsTask(draft)) {
    rows.push("任务启动：点击「应用」后才会启动任务，并在对应 dashboard 显示进度。");
  } else {
    rows.push("配置写入：点击「应用」后才会保存；未应用前不会改动配置。");
  }
  rows.push("安全确认：点击「放弃」不会写入；点击「调整」会让 Agent 重写草案。");
  return rows;
}

function draftActionCount(draft: AgentActionDraft): number {
  if (draft.kind !== "compound_config_update") return 1;
  const actions = draft.payload.actions;
  return Array.isArray(actions) ? actions.filter(isRecord).length : 1;
}

function draftStartsTask(draft: AgentActionDraft): boolean {
  if (isStartTaskKind(draft.kind)) return true;
  if (draft.kind !== "compound_config_update") return false;
  const actions = draft.payload.actions;
  if (!Array.isArray(actions)) return false;
  return actions.some(
    (action) => isRecord(action) && isStartTaskKind(String(action.kind || "")),
  );
}

function isStartTaskKind(kind: string): boolean {
  return (
    kind === "start_glossary_task" ||
    kind === "start_glossary_review_task" ||
    kind === "start_translation_task"
  );
}

function DraftActionPreview({
  action,
  index,
}: {
  action: Record<string, unknown>;
  index: number;
}) {
  const payload = isRecord(action.payload) ? action.payload : {};
  const kind = typeof action.kind === "string" ? action.kind : "";
  const title =
    typeof action.title === "string" && action.title.trim()
      ? action.title
      : formatDraftKind(kind);
  const summary =
    typeof action.summary === "string" && action.summary.trim()
      ? action.summary
      : "";
  return (
    <div className={styles.draftActionItem}>
      <div className={styles.draftActionHeader}>
        <span className={styles.draftActionIndex}>{index + 1}</span>
        <div>
          <strong>{title}</strong>
          {summary ? <p>{summary}</p> : null}
        </div>
      </div>
      <DraftFields payload={payload} />
    </div>
  );
}

function DraftFields({ payload }: { payload: Record<string, unknown> }) {
  const entries = visibleDraftEntries(payload);
  if (!entries.length) {
    return <div className={styles.draftEmpty}>没有可展示的配置字段。</div>;
  }
  return (
    <dl className={styles.draftFields}>
      {entries.map(([key, value]) => (
        <div key={key} className={styles.draftField}>
          <dt>{formatDraftField(key)}</dt>
          <dd>{renderDraftValue(key, value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function visibleDraftEntries(
  payload: Record<string, unknown>,
): Array<[string, unknown]> {
  return Object.entries(payload).filter(([key, value]) => {
    if (key === "actions") return false;
    return value !== undefined;
  });
}

function renderDraftValue(key: string, value: unknown): JSX.Element {
  if (typeof value === "string") {
    return renderDraftString(key, value);
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return <span>{formatPrimitive(value)}</span>;
  }
  if (value === null) {
    return <span className={styles.draftMuted}>未设置</span>;
  }
  if (Array.isArray(value)) {
    if (!value.length) return <span className={styles.draftMuted}>空列表</span>;
    return (
      <ul className={styles.draftMiniList}>
        {value.map((item, index) => (
          <li key={index}>{renderInlineValue(item)}</li>
        ))}
      </ul>
    );
  }
  if (isRecord(value)) {
    const entries = Object.entries(value);
    if (!entries.length) return <span className={styles.draftMuted}>无</span>;
    return (
      <dl className={styles.draftNestedFields}>
        {entries.map(([childKey, childValue]) => (
          <div key={childKey}>
            <dt>{formatDraftField(childKey)}</dt>
            <dd>{renderInlineValue(childValue)}</dd>
          </div>
        ))}
      </dl>
    );
  }
  return <span>{String(value)}</span>;
}

function renderDraftString(key: string, value: string): JSX.Element {
  const text = value.trim();
  if (!text) return <span className={styles.draftMuted}>空</span>;
  const isLong =
    key === "system_prompt" ||
    key === "prompt" ||
    key === "content" ||
    text.length > DRAFT_TEXT_PREVIEW_CHARS;
  if (!isLong) return <span>{text}</span>;
  const preview =
    text.length > DRAFT_TEXT_PREVIEW_CHARS
      ? `${text.slice(0, DRAFT_TEXT_PREVIEW_CHARS).trimEnd()}…`
      : text;
  return (
    <details className={styles.draftTextDetails}>
      <summary>{preview}</summary>
      <div>{text}</div>
    </details>
  );
}

function renderInlineValue(value: unknown): string {
  if (typeof value === "string") return value || "未设置";
  if (typeof value === "number" || typeof value === "boolean") {
    return formatPrimitive(value);
  }
  if (value === null || value === undefined) return "未设置";
  if (Array.isArray(value)) return value.map(renderInlineValue).join("、");
  if (isRecord(value)) {
    return Object.entries(value)
      .map(
        ([key, child]) =>
          `${formatDraftField(key)}：${renderInlineValue(child)}`,
      )
      .join("；");
  }
  return String(value);
}

function formatPrimitive(value: number | boolean): string {
  if (typeof value === "boolean") return value ? "是" : "否";
  return String(value);
}

function RawDraftDetails({ payload }: { payload: Record<string, unknown> }) {
  return (
    <details className={styles.draftRawDetails}>
      <summary>查看技术详情</summary>
      <pre>{JSON.stringify(payload, null, 2)}</pre>
    </details>
  );
}

function formatDraftKind(kind: string): string {
  return (
    {
      create_prompt_preset: "创建 Prompt 预设",
      update_prompt_preset: "更新 Prompt 预设",
      create_recipe: "创建预设配置",
      update_recipe: "更新预设配置",
      apply_recipe: "应用预设配置",
      delete_recipe: "删除预设配置",
      create_model_profile: "创建模型配置",
      update_model_profile: "更新模型配置",
      update_workspace: "更新工作区配置",
      update_memory: "更新记忆",
      add_memory: "添加记忆",
      delete_memory: "删除记忆",
      compound_config_update: "复合配置修改",
      start_glossary_task: "启动术语提取任务",
      start_glossary_review_task: "启动术语审查任务",
      start_translation_task: "启动翻译任务",
    }[kind] ?? kind
  );
}

function formatDraftField(key: string): string {
  return (
    {
      kind: "类型",
      id: "ID",
      name: "名称",
      title: "标题",
      description: "说明",
      system_prompt: "系统提示词",
      prompt: "提示词",
      content: "内容",
      enabled: "启用",
      profile_id: "模型 ID",
      recipe_id: "预设 ID",
      patch: "修改内容",
      stage_model_ids: "阶段模型",
      stage_prompt_ids: "阶段 Prompt",
      workflow_model_id: "工作模型",
      workflow_thinking_level: "思考强度",
      translation: "翻译",
      glossary: "术语提取",
      glossary_review: "术语审查",
      term_extract: "术语提取",
      term_review: "术语审查",
      input_dir: "输入目录",
      output_dir: "输出目录",
      input_folder: "输入目录",
      output_folder: "输出目录",
      source_language: "源语言",
      target_language: "目标语言",
      novel_background: "小说背景",
      glossary_task_id: "术语任务 ID",
      glossary_review_task_id: "术语审查任务 ID",
      concurrency_limit: "并发数",
      rpm_limit: "每分钟请求数",
      tpm_limit: "每分钟 Token 数",
      retry_attempts: "重试次数",
      model_id: "模型名称",
      provider_format: "接口类型",
      base_url: "接口地址",
    }[key] ?? key
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
