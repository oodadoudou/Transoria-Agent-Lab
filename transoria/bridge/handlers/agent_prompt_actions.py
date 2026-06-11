"""Prompt preset action helpers for Agent Lab."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from transoria.bridge.errors import BridgeError
from transoria.bridge.handlers.agent_inventory import (
    generate_prompt_id,
    prompt_store_for,
)
from transoria.prompts import PromptKind, PromptPreset

PROMPT_KIND_ALIASES = {
    "translation": PromptKind.TRANSLATION,
    "translation_prompt": PromptKind.TRANSLATION,
    "translate_prompt": PromptKind.TRANSLATION,
    "translate": PromptKind.TRANSLATION,
    "翻译": PromptKind.TRANSLATION,
    "翻译prompt": PromptKind.TRANSLATION,
    "翻译提示词": PromptKind.TRANSLATION,
    "翻译预设": PromptKind.TRANSLATION,
    "glossary": PromptKind.GLOSSARY,
    "term_extract": PromptKind.GLOSSARY,
    "term_extraction": PromptKind.GLOSSARY,
    "glossary_extraction": PromptKind.GLOSSARY,
    "glossary_prompt": PromptKind.GLOSSARY,
    "term_extract_prompt": PromptKind.GLOSSARY,
    "术语": PromptKind.GLOSSARY,
    "术语提取": PromptKind.GLOSSARY,
    "术语提取prompt": PromptKind.GLOSSARY,
    "术语提取提示词": PromptKind.GLOSSARY,
    "术语提取预设": PromptKind.GLOSSARY,
    "glossary_review": PromptKind.GLOSSARY_REVIEW,
    "glossary_review_prompt": PromptKind.GLOSSARY_REVIEW,
    "term_review": PromptKind.GLOSSARY_REVIEW,
    "term_review_prompt": PromptKind.GLOSSARY_REVIEW,
    "term_audit": PromptKind.GLOSSARY_REVIEW,
    "术语审核": PromptKind.GLOSSARY_REVIEW,
    "术语审查": PromptKind.GLOSSARY_REVIEW,
    "术语审核prompt": PromptKind.GLOSSARY_REVIEW,
    "术语审查prompt": PromptKind.GLOSSARY_REVIEW,
    "术语审核提示词": PromptKind.GLOSSARY_REVIEW,
    "术语审查提示词": PromptKind.GLOSSARY_REVIEW,
    "术语审核预设": PromptKind.GLOSSARY_REVIEW,
    "术语审查预设": PromptKind.GLOSSARY_REVIEW,
}


def create_prompt_preset(
    payload: Mapping[str, object],
    *,
    cache_root: Path,
) -> PromptPreset:
    kind, name, description, system_prompt, enabled = coerce_prompt_preset_payload(
        payload
    )
    store = prompt_store_for(cache_root, kind)
    existing = list(store.load())
    preset_id = generate_prompt_id(name, kind, existing)
    preset = PromptPreset(
        id=preset_id,
        name=name,
        kind=kind,
        system_prompt=system_prompt,
        suffix_prompt="",
        thinking_prompt="",
        description=description,
        enabled=enabled,
        is_system=False,
    )
    store.save([*existing, preset])
    return preset


def update_prompt_preset(
    payload: Mapping[str, object],
    *,
    cache_root: Path,
) -> PromptPreset:
    preset_id, patch = coerce_prompt_update_payload(payload)
    kind, presets, index = resolve_prompt_for_update(
        preset_id,
        patch,
        cache_root=cache_root,
    )
    updated = replace(presets[index], **patch)
    presets[index] = updated
    prompt_store_for(cache_root, kind).save(presets)
    return updated


def coerce_prompt_update_payload(
    payload: Mapping[str, object],
) -> tuple[str, dict[str, object]]:
    preset_id = str(payload.get("id") or payload.get("preset_id") or "").strip()
    if not preset_id:
        raise BridgeError.invalid_argument("id is required.", field="id")
    raw_patch = payload.get("patch")
    if raw_patch is None:
        raw_patch = {
            key: payload[key]
            for key in ("name", "system_prompt", "description", "enabled")
            if key in payload
        }
    if not isinstance(raw_patch, Mapping):
        raise BridgeError.invalid_argument(
            "patch object is required.",
            field="patch",
        )
    patch = coerce_prompt_patch(raw_patch)
    if not patch:
        raise BridgeError.invalid_argument(
            "patch must include at least one editable field.",
            field="patch",
        )
    return preset_id, patch


def coerce_prompt_patch(patch: Mapping[str, object]) -> dict[str, object]:
    valid = {"name", "system_prompt", "description", "enabled"}
    coerced: dict[str, object] = {}
    for key, value in patch.items():
        if key not in valid:
            raise BridgeError.invalid_argument(
                f"field {key!r} cannot be updated.",
                field=key,
            )
        if key == "enabled":
            if not isinstance(value, bool):
                raise BridgeError.invalid_argument(
                    "enabled must be a boolean.",
                    field=key,
                )
            coerced[key] = value
            continue
        if not isinstance(value, str):
            raise BridgeError.invalid_argument(f"{key} must be a string.", field=key)
        text = value.strip() if key in {"name", "system_prompt"} else value
        if key in {"name", "system_prompt"} and not text:
            raise BridgeError.invalid_argument(f"{key} must not be empty.", field=key)
        coerced[key] = text
    return coerced


def resolve_prompt_for_update(
    preset_id: str,
    patch: Mapping[str, object],
    *,
    cache_root: Path,
) -> tuple[PromptKind, list[PromptPreset], int]:
    for kind in PromptKind:
        store = prompt_store_for(cache_root, kind)
        presets = list(store.load())
        for index, preset in enumerate(presets):
            if preset.id != preset_id:
                continue
            if preset.is_system:
                raise BridgeError.invalid_argument(
                    "system prompt presets are read-only; duplicate to edit.",
                    details={"reason": "is_system"},
                )
            return kind, presets, index
    raise BridgeError.not_found(f"prompt preset {preset_id!r} does not exist.")


def coerce_prompt_preset_payload(
    payload: Mapping[str, object],
) -> tuple[PromptKind, str, str, str, bool]:
    kind = coerce_prompt_kind(
        _first_present(
            payload,
            (
                "kind",
                "type",
                "prompt_kind",
                "promptKind",
                "stage",
                "category",
                "类型",
                "类别",
                "用途",
            ),
        ),
        fallback_text=" ".join(
            str(payload.get(key) or "")
            for key in ("name", "description", "system_prompt", "prompt", "content")
        ),
    )
    name = str(
        _first_present(payload, ("name", "display_name", "displayName", "名称", "名字"))
        or ""
    ).strip()
    system_prompt = prompt_body_from_payload(payload)
    if not name:
        raise BridgeError.invalid_argument("name is required.", field="name")
    if not system_prompt:
        raise BridgeError.invalid_argument(
            "system_prompt is required.",
            field="system_prompt",
        )
    return (
        kind,
        name,
        str(payload.get("description") or "").strip(),
        system_prompt,
        bool(payload.get("enabled", True)),
    )


def coerce_prompt_kind(value: object, *, fallback_text: str = "") -> PromptKind:
    raw_kind = str(value or "").strip()
    normalized = (
        (raw_kind or fallback_text).lower()
        .replace(" ", "")
        .replace("-", "_")
        .replace("·", "")
        .replace("：", "")
        .replace(":", "")
    )
    kind = PROMPT_KIND_ALIASES.get(raw_kind) or PROMPT_KIND_ALIASES.get(normalized)
    if kind is not None:
        return kind
    if any(marker in normalized for marker in ("review", "audit", "审核", "审查")):
        return PromptKind.GLOSSARY_REVIEW
    if any(marker in normalized for marker in ("translation", "translate", "翻译")):
        return PromptKind.TRANSLATION
    if any(marker in normalized for marker in ("glossary", "term", "术语")):
        return PromptKind.GLOSSARY
    raise BridgeError.invalid_argument(
        "prompt kind must be translation, glossary, or glossary_review.",
        field="kind",
    )


def prompt_body_from_payload(payload: Mapping[str, object]) -> str:
    for key in ("system_prompt", "systemPrompt", "prompt", "content", "body"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _first_present(payload: Mapping[str, object], keys: tuple[str, ...]) -> object | None:
    for key in keys:
        if key in payload:
            return payload[key]
    return None
