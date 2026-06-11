"""Direct model-profile response builders for Agent Lab chat."""

from __future__ import annotations

from collections.abc import Callable
from typing import Mapping

from transoria.agent.schemas import MODEL_SLOTS, AgentActionDraft, AgentWorkspaceState
from transoria.bridge.handlers.agent_inventory import (
    model_profile_has_placeholder_fields,
)
from transoria.bridge.handlers.agent_model_intents import (
    extract_model_copy_display_name,
    extract_provider_model_id_override,
    find_existing_upgrade_profile,
    infer_provider_model_id_from_copy_request,
    is_same_model_family,
    looks_like_model_profile_copy_continuation,
    looks_like_model_profile_copy_request,
    model_profile_copy_payload,
    requests_different_provider_model_id,
    resolve_model_profile_from_copy_request,
)
from transoria.model_profiles import ModelProfileStore

ExcludedRequestPredicate = Callable[[str], bool]


def direct_model_profile_copy_response(
    *,
    user_message: str,
    conversation_context: list[Mapping[str, object]],
    profile_store: ModelProfileStore,
) -> tuple[str, AgentActionDraft | None] | None:
    context_text = _conversation_context_text(conversation_context, role="user")
    current_request = looks_like_model_profile_copy_request(user_message)
    contextual_confirmation = (
        looks_like_model_profile_copy_request(context_text)
        and looks_like_model_profile_copy_continuation(user_message)
    )
    if not current_request and not contextual_confirmation:
        return None
    combined_text = "\n".join((context_text, user_message))
    request_text = combined_text if contextual_confirmation else user_message

    source = resolve_model_profile_from_copy_request(
        request_text,
        profile_store=profile_store,
    )
    if source is None:
        return (
            "我理解你想基于已有模型配置复制创建一个新配置，但没有在当前模型库中唯一匹配到源模型。请明确要复制哪个模型配置名称。",
            None,
        )

    display_name = extract_model_copy_display_name(user_message)
    if not display_name:
        display_name = extract_model_copy_display_name(request_text)
    if not display_name:
        return (
            f"我可以基于 {source.display_name} 复制创建新模型配置。请明确新配置的显示名称。",
            None,
        )

    provider_model_id = extract_provider_model_id_override(user_message)
    if provider_model_id is None:
        provider_model_id = extract_provider_model_id_override(request_text)
    if provider_model_id is None:
        provider_model_id = infer_provider_model_id_from_copy_request(
            source,
            display_name=display_name,
            text=request_text,
        )
    if requests_different_provider_model_id(user_message) and not provider_model_id:
        return (
            (
                f"我可以复用 {source.display_name} 的 provider_format、base_url、并发/限速和 API key 状态，"
                f"并把新配置显示名称设为 {display_name}。但你明确要求不要复用相同的模型 ID，"
                "请给出要写入 provider 的准确 model_id。"
            ),
            None,
        )

    profile_payload = model_profile_copy_payload(
        source,
        display_name=display_name,
        provider_model_id=provider_model_id,
    )
    model_id_note = (
        f"provider model_id 改为 {provider_model_id}"
        if provider_model_id
        else f"provider model_id 保持为 {source.model_id}"
    )
    draft = AgentActionDraft.create(
        kind="create_model_profile",
        title=f"复制模型配置为 {display_name}",
        summary=(
            f"基于 {source.display_name} 创建新模型配置 {display_name}，"
            f"{model_id_note}，其余运行参数保持一致。"
        ),
        payload={"profile": profile_payload},
    )
    return (
        (
            f"我会基于 {source.display_name} 复制创建新模型配置 {display_name}。"
            f"{model_id_note}；provider_format、base_url、并发/限速、重试、思考配置和已配置的 API key 状态保持一致。"
            "请在草案中确认后应用。"
        ),
        draft,
    )


def direct_vague_model_profile_guidance_response(
    *,
    user_message: str,
    profile_store: ModelProfileStore,
    excluded_request: ExcludedRequestPredicate | None = None,
) -> tuple[str, AgentActionDraft | None] | None:
    if not looks_like_vague_model_profile_request(
        user_message,
        excluded_request=excluded_request,
    ):
        return None
    usable_profiles = [
        profile
        for profile in profile_store.load()
        if profile.api_keys and not model_profile_has_placeholder_fields(profile)
    ]
    examples = ""
    if usable_profiles:
        labels = "、".join(profile.display_name for profile in usable_profiles[:5])
        examples = f"\n\n当前可以复用的已有模型配置：{labels}。"
    return (
        (
            "我可以帮你配置模型，但不能替你猜 provider model_id、base_url 或 API key。"
            "如果你不知道具体信息，请选择一种方式：\n"
            "1. 说“按照某个已有模型复制一个”，并告诉我要改成的准确 provider model_id；\n"
            "2. 直接提供接口类型、base_url、provider model_id 和 API key，我会生成确认草案；\n"
            "3. 如果只是想提高质量，可以让我先查看现有模型，切换到已经配置好的更强模型。\n\n"
            "任何写入都会先生成草案，API key 在预览里会遮罩。"
            f"{examples}"
        ),
        None,
    )


def direct_model_upgrade_response(
    *,
    user_message: str,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    excluded_request: ExcludedRequestPredicate | None = None,
) -> tuple[str, AgentActionDraft | None] | None:
    if not looks_like_model_upgrade_request(
        user_message,
        excluded_request=excluded_request,
    ):
        return None
    target = find_existing_upgrade_profile(
        user_message,
        state=state,
        profile_store=profile_store,
    )
    if target is None:
        return (
            (
                "我理解你想换成更强的模型配置，但当前模型库里没有唯一可复用的更高质量配置。"
                "请告诉我要使用的准确 provider model_id，或者先在模型页添加一个可用模型；"
                "如果你只是想先用现有配置，我也可以帮你把当前流程切到已有模型。"
            ),
            None,
        )

    stage_models = {slot: target.id for slot in MODEL_SLOTS}
    payload: dict[str, object] = {"stage_model_ids": stage_models}
    current_workflow = (
        profile_store.get(state.workflow_model_id) if state.workflow_model_id else None
    )
    if current_workflow is None or is_same_model_family(current_workflow, target):
        payload["workflow_model_id"] = target.id
    draft = AgentActionDraft.create(
        kind="update_workspace",
        title=f"切换到更强模型 {target.display_name}",
        summary=(
            f"将翻译、术语提取和术语审查阶段切换到现有模型 {target.display_name}。"
            "如果当前工作模型属于同一模型系列，也会同步切换工作模型。"
        ),
        payload=payload,
    )
    return (
        (
            f"我在现有模型库里找到了更适合质量测试的模型 {target.display_name}。"
            "我不会直接保存；下面是把翻译流程各阶段切换到这个模型的确认草案。"
        ),
        draft,
    )


def looks_like_vague_model_profile_request(
    text: str,
    *,
    excluded_request: ExcludedRequestPredicate | None = None,
) -> bool:
    normalized = text.lower()
    if looks_like_model_profile_copy_request(text) or (
        excluded_request is not None and excluded_request(text)
    ):
        return False
    has_model = any(marker in normalized for marker in ("模型", "model", "profile"))
    has_create_or_config = any(
        marker in normalized
        for marker in (
            "新增",
            "添加",
            "加一个",
            "新建",
            "创建",
            "配置",
            "接入",
            "add",
            "create",
            "configure",
        )
    )
    has_uncertainty = any(
        marker in normalized
        for marker in (
            "不知道",
            "不清楚",
            "不会",
            "不懂",
            "不了解",
            "随便",
            "你帮我",
            "帮我配",
            "帮我配置",
            "不确定",
        )
    )
    return has_model and has_create_or_config and has_uncertainty


def looks_like_model_upgrade_request(
    text: str,
    *,
    excluded_request: ExcludedRequestPredicate | None = None,
) -> bool:
    normalized = text.lower()
    if looks_like_model_profile_copy_request(text) or (
        excluded_request is not None and excluded_request(text)
    ):
        return False
    has_create_or_config = any(
        marker in normalized
        for marker in (
            "新增",
            "添加",
            "加一个",
            "新建",
            "创建",
            "配置",
            "接入",
            "add",
            "create",
            "configure",
        )
    )
    has_uncertainty = any(
        marker in normalized
        for marker in (
            "不知道",
            "不清楚",
            "不会",
            "不懂",
            "不了解",
            "随便",
            "你帮我",
            "帮我配",
            "帮我配置",
            "不确定",
        )
    )
    has_existing_inventory_intent = any(
        marker in normalized
        for marker in (
            "现有配置",
            "已有配置",
            "当前配置",
            "模型库",
            "已经配置",
            "已配置",
            "inventory",
        )
    )
    if has_create_or_config and has_uncertainty and not has_existing_inventory_intent:
        return False
    has_model = any(marker in normalized for marker in ("模型", "model", "profile"))
    has_quality_intent = any(
        marker in normalized
        for marker in (
            "更厉害",
            "更强",
            "更好",
            "高质量",
            "高规格",
            "高级",
            "pro",
            "质量",
            "升级",
            "换成",
            "切换",
            "better",
            "stronger",
            "upgrade",
        )
    )
    has_agent_help = any(
        marker in normalized
        for marker in (
            "不知道",
            "不清楚",
            "不会",
            "你帮我",
            "帮我看",
            "看看",
            "帮我",
            "配置",
            "修改方案",
            "方案",
        )
    )
    return has_model and has_quality_intent and has_agent_help


def _conversation_context_text(
    conversation_context: list[Mapping[str, object]],
    *,
    role: str | None = None,
) -> str:
    parts: list[str] = []
    for item in conversation_context:
        if role is not None and item.get("role") != role:
            continue
        content = str(item.get("content") or "").strip()
        if content:
            parts.append(content)
    return "\n".join(parts)
