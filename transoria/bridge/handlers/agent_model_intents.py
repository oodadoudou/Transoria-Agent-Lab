"""Model-profile intent helpers for the Agent Lab bridge."""

from __future__ import annotations

import re

from transoria.agent.schemas import AgentWorkspaceState
from transoria.llm.config import ModelConfig, ThinkingLevel
from transoria.model_profiles import ModelProfileStore

MAX_MODEL_TITLE_LENGTH = 120


def looks_like_model_profile_copy_request(text: str) -> bool:
    normalized = text.lower()
    has_model = any(marker in normalized for marker in ("模型", "model", "profile"))
    has_copy = any(
        marker in normalized
        for marker in (
            "按照",
            "基于",
            "复制",
            "克隆",
            "参照",
            "照着",
            "一样的",
            "same",
            "copy",
            "clone",
            "duplicate",
        )
    )
    has_create = any(
        marker in normalized
        for marker in (
            "创建",
            "新建",
            "配置一个",
            "新模型",
            "新配置",
            "复制一个",
            "复制一套",
            "add",
            "create",
        )
    )
    return has_model and has_copy and has_create


def looks_like_model_profile_copy_continuation(text: str) -> bool:
    normalized = text.lower()
    if len(text) > 500:
        return False
    has_confirmation = any(
        marker in normalized
        for marker in (
            "对的",
            "是的",
            "可以",
            "确认",
            "创建一个新的",
            "用这个",
            "不要用相同",
            "改为",
            "改成",
            "model_id",
            "模型 id",
            "模型id",
            "provider format",
            "base_url",
            "接口模型",
        )
    )
    return has_confirmation


def find_existing_upgrade_profile(
    user_message: str,
    *,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
) -> ModelConfig | None:
    requested_families = model_family_tokens_from_text(user_message)
    current_profiles = current_workspace_profiles(
        state,
        profile_store=profile_store,
    )
    if not requested_families:
        for profile in current_profiles:
            requested_families.update(model_family_tokens_for_profile(profile))

    comparable_current_profiles = [
        profile
        for profile in current_profiles
        if not requested_families
        or (model_family_tokens_for_profile(profile) & requested_families)
    ]
    current_best = max(
        (model_quality_score(profile) for profile in comparable_current_profiles),
        default=-100,
    )
    candidates: list[tuple[int, str, ModelConfig]] = []
    for profile in profile_store.load():
        if not profile.api_keys:
            continue
        families = model_family_tokens_for_profile(profile)
        if requested_families and not (families & requested_families):
            continue
        score = model_quality_score(profile)
        if score <= current_best and requested_families:
            continue
        candidates.append((score, profile.display_name.lower(), profile))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_score = candidates[0][0]
    best = [profile for score, _, profile in candidates if score == best_score]
    if len(best) != 1:
        return None
    if best_score < 10:
        return None
    return best[0]


def current_workspace_profiles(
    state: AgentWorkspaceState,
    *,
    profile_store: ModelProfileStore,
) -> list[ModelConfig]:
    ids = [state.workflow_model_id, *state.stage_model_ids.values()]
    profiles: list[ModelConfig] = []
    seen: set[str] = set()
    for profile_id in ids:
        if not profile_id or profile_id in seen:
            continue
        profile = profile_store.get(profile_id)
        if profile is None:
            continue
        seen.add(profile.id)
        profiles.append(profile)
    return profiles


def is_same_model_family(left: ModelConfig, right: ModelConfig) -> bool:
    return bool(model_family_tokens_for_profile(left) & model_family_tokens_for_profile(right))


def model_family_tokens_for_profile(profile: ModelConfig) -> set[str]:
    return model_family_tokens_from_text(
        " ".join((profile.id, profile.display_name, profile.model_id))
    )


def model_family_tokens_from_text(text: str) -> set[str]:
    normalized = text.lower()
    aliases = {
        "deepseek": ("deepseek", "deep seek", "深度求索"),
        "gemini": ("gemini", "google"),
        "claude": ("claude", "anthropic"),
        "gpt": ("gpt", "openai", "chatgpt"),
        "qwen": ("qwen", "通义", "千问"),
        "kimi": ("kimi", "moonshot"),
    }
    result: set[str] = set()
    for family, markers in aliases.items():
        if any(marker in normalized for marker in markers):
            result.add(family)
    return result


def model_quality_score(profile: ModelConfig) -> int:
    label = f"{profile.id} {profile.display_name} {profile.model_id}".lower()
    score = 0
    if "pro" in label:
        score += 40
    if "reasoner" in label or "reasoning" in label:
        score += 36
    if "opus" in label:
        score += 34
    if "sonnet" in label:
        score += 24
    if "max" in label or "ultra" in label:
        score += 18
    if profile.thinking_level is not ThinkingLevel.OFF:
        score += 8
    if "flash" in label:
        score -= 16
    if any(marker in label for marker in ("mini", "lite", "nano", "haiku")):
        score -= 20
    return score


def model_profile_copy_payload(
    source: ModelConfig,
    *,
    display_name: str,
    provider_model_id: str | None,
) -> dict[str, object]:
    body = source.to_dict()
    body.pop("id", None)
    body.pop("api_keys", None)
    body["copy_from_profile_id"] = source.id
    body["display_name"] = display_name
    if provider_model_id:
        body["model_id"] = provider_model_id
    return body


def resolve_model_profile_from_copy_request(
    text: str,
    *,
    profile_store: ModelProfileStore,
) -> ModelConfig | None:
    labels = extract_model_copy_source_labels(text)
    for label in labels:
        resolved = resolve_model_profile_by_label(label, profile_store)
        if resolved is not None:
            return resolved
        fuzzy = resolve_model_profile_by_fuzzy_text(label, profile_store)
        if fuzzy is not None:
            return fuzzy
    return resolve_model_profile_by_fuzzy_text(text, profile_store)


def requested_existing_model_for_task(
    text: str,
    *,
    profile_store: ModelProfileStore,
) -> ModelConfig | None:
    for label in extract_task_model_reference_labels(text):
        resolved = resolve_model_profile_by_label(label, profile_store)
        if resolved is not None:
            return resolved
        fuzzy = resolve_model_profile_by_fuzzy_text(label, profile_store)
        if fuzzy is not None:
            return fuzzy
    return None


def extract_task_model_reference_labels(text: str) -> list[str]:
    labels: list[str] = []
    patterns = (
        r"(?:按|按照|用|使用|采用|基于|走)\s*(?:我(?:已经)?配置(?:好)?的|已配置的|配置好的|现有的|已有的)?\s*([A-Za-z0-9_.\-\s]+?)\s*(?:的)?(?:模型配置|配置|模型|profile)",
        r"(?:模型配置|阶段模型|工作模型|模型|profile)\s*(?:用|使用|选择|切到|切换到|换成|设为|设置为)\s*([A-Za-z0-9_.\-\s]+)",
        r"(?:with|using|use|based on)\s+([A-Za-z0-9_.\-\s]+?)\s+(?:model|profile|config)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            label = match.group(1).strip(" \t\n\r:：，。,.「」『』“”\"'")
            if label and is_usable_task_model_label(label):
                labels.append(label)
    return labels


def is_usable_task_model_label(label: str) -> bool:
    normalized = label.strip().lower()
    if not normalized:
        return False
    return normalized not in {
        "当前",
        "当前工作",
        "当前工作区",
        "工作",
        "默认",
        "现在",
        "current",
        "default",
        "workflow",
    }


def extract_model_copy_source_labels(text: str) -> list[str]:
    labels: list[str] = []
    patterns = (
        r"(?:按照|基于|参照|照着|复制|克隆|从)\s*([A-Za-z0-9_.\-\s]+?)\s*(?:的配置|配置|模型|profile)",
        r"(?:copy|clone|duplicate|from|based on)\s+([A-Za-z0-9_.\-\s]+?)(?:\s+profile|\s+model|$)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            label = match.group(1).strip(" \t\n\r:：，。,.")
            if label:
                labels.append(label)
    return labels


def resolve_model_profile_by_fuzzy_text(
    text: str,
    profile_store: ModelProfileStore,
) -> ModelConfig | None:
    text_tokens = lookup_tokens(text)
    if not text_tokens:
        return None
    scored: list[tuple[int, ModelConfig]] = []
    normalized_text = normalize_lookup_label(text)
    for profile in profile_store.load():
        labels = (profile.id, profile.display_name, profile.model_id)
        score = 0
        for label in labels:
            normalized_label = normalize_lookup_label(label)
            if normalized_label and normalized_label in normalized_text:
                score += 6
            score += len(text_tokens & lookup_tokens(label))
        if score:
            scored.append((score, profile))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored[0][0] < 2:
        return None
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]


def lookup_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", value.lower())
        if token
        not in {
            "api",
            "base",
            "config",
            "format",
            "id",
            "key",
            "model",
            "profile",
            "provider",
            "url",
            "一样",
            "一样的",
            "不同",
            "新的",
            "模型",
            "相同",
            "配置",
        }
    }


def extract_model_copy_display_name(text: str) -> str:
    patterns = (
        r"(?:模型(?:显示)?(?:名称|名字|名)|显示(?:名称|名字)|display_name|display name)\s*(?:叫做|叫|改为|改成|设为|设置为|为|=|:|：)\s*[「『“\"']?(.+?)[」』”\"']?(?:[，。,.]|$)",
        r"(?:配置|创建|新建)(?:一个|一套)?\s*[「『“\"']?(.+?)[」』”\"']?(?:的)?(?:新)?模型(?:配置)?",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip(" \t\n\r「」『』“”\"'")
            if value:
                return value[:MAX_MODEL_TITLE_LENGTH]
    return ""


def extract_provider_model_id_override(text: str) -> str | None:
    patterns = (
        r"(?:model[_\s-]?id|模型\s*ID|接口模型|provider\s+model(?:\s+id)?)\s*(?:改为|改成|设置为|设为|使用|用|为|=|:|：)\s*[`\"']?([^`\"'，。,\s]+)",
        r"[`\"']([A-Za-z0-9_.:/\-]+)[`\"']\s*(?:这个)?\s*(?:model[_\s-]?id|模型\s*ID|接口模型)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if value:
                return value[:MAX_MODEL_TITLE_LENGTH]
    return None


def infer_provider_model_id_from_copy_request(
    source: ModelConfig,
    *,
    display_name: str,
    text: str,
) -> str | None:
    if not requests_different_provider_model_id(text):
        return None
    source_model_id = source.model_id.strip()
    if not source_model_id:
        return None
    target = display_name.lower()
    if "pro" in target and re.search(r"(?:^|[-_.])flash(?:$|[-_.])", source_model_id):
        return re.sub(
            r"(?i)(?:^|(?<=[-_.]))flash(?=$|[-_.])",
            "pro",
            source_model_id,
            count=1,
        )[:MAX_MODEL_TITLE_LENGTH]
    if "flash" in target and re.search(r"(?:^|[-_.])pro(?:$|[-_.])", source_model_id):
        return re.sub(
            r"(?i)(?:^|(?<=[-_.]))pro(?=$|[-_.])",
            "flash",
            source_model_id,
            count=1,
        )[:MAX_MODEL_TITLE_LENGTH]
    return None


def requests_different_provider_model_id(text: str) -> bool:
    normalized = text.lower()
    return bool(
        re.search(
            r"(?:不要|不|不能|别).{0,8}(?:相同|一样).{0,8}(?:model[_\s-]?id|模型\s*id)",
            normalized,
        )
        or re.search(
            r"(?:model[_\s-]?id|模型\s*id|接口模型).{0,12}(?:新的|不同|不一样|改为|改成|换成)",
            normalized,
        )
    )


def resolve_model_profile_by_label(
    label: str,
    profile_store: ModelProfileStore,
) -> ModelConfig | None:
    normalized = normalize_lookup_label(label)
    matches = [
        profile
        for profile in profile_store.load()
        if normalize_lookup_label(profile.id) == normalized
        or normalize_lookup_label(profile.display_name) == normalized
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def normalize_lookup_label(value: str) -> str:
    return value.strip().lower().replace(" ", "").replace("·", "")
