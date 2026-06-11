"""Model profile action helpers for Agent Lab bridge handlers."""

from __future__ import annotations

from secrets import token_hex
from typing import Mapping, Protocol

from transoria.bridge.errors import BridgeError
from transoria.bridge.handlers.agent_inventory import looks_like_placeholder_value
from transoria.llm.config import ModelConfig, ProviderFormat, ThinkingLevel
from transoria.model_profiles import ModelProfileStore


class ModelProfileReader(Protocol):
    def get(self, profile_id: str) -> ModelConfig | None: ...


def model_profile_from_draft_payload(
    payload: Mapping[str, object],
    *,
    profile_store: ModelProfileReader | None = None,
) -> ModelConfig:
    raw_profile = payload.get("profile")
    body = dict(raw_profile) if isinstance(raw_profile, Mapping) else dict(payload)
    source_profile_id = str(
        body.pop("copy_from_profile_id", None)
        or body.pop("source_profile_id", None)
        or ""
    ).strip()
    copied_api_keys: tuple[str, ...] = ()
    if source_profile_id:
        if profile_store is None:
            raise BridgeError.invalid_argument(
                "copy_from_profile_id requires profile store access.",
                field="copy_from_profile_id",
            )
        source = profile_store.get(source_profile_id)
        if source is None:
            raise BridgeError.not_found(
                f"profile {source_profile_id!r} does not exist.",
                details={"id": source_profile_id},
            )
        source_body = source.to_dict()
        source_body.pop("id", None)
        source_body.pop("api_keys", None)
        source_body.update(body)
        body = source_body
        copied_api_keys = source.api_keys
    body.setdefault("id", generate_model_profile_id(body))
    for field in ("display_name", "model_id"):
        if looks_like_placeholder_value(body.get(field)):
            raise BridgeError.invalid_argument(
                f"{field} contains placeholder text.",
                field=field,
            )
    api_keys = body.pop("api_keys", None)
    try:
        profile = ModelConfig.from_dict(body)
    except (KeyError, TypeError, ValueError) as exc:
        raise BridgeError.invalid_argument(str(exc)) from exc
    keys = (
        copied_api_keys
        if api_keys is None and source_profile_id
        else coerce_api_keys(api_keys)
    )
    return profile.with_api_keys(keys)


def coerce_model_profile_update_payload(
    payload: Mapping[str, object],
) -> tuple[str, dict[str, object], object | None]:
    profile_id = str(payload.get("profile_id") or payload.get("id") or "").strip()
    if not profile_id:
        raise BridgeError.invalid_argument("profile_id is required.", field="profile_id")
    raw_patch = payload.get("patch")
    if raw_patch is None:
        raw_patch = {
            key: value for key, value in payload.items() if key not in {"profile_id", "id"}
        }
    if not isinstance(raw_patch, Mapping):
        raise BridgeError.invalid_argument(
            "patch object is required.",
            field="patch",
        )
    patch = dict(raw_patch)
    api_keys = patch.pop("api_keys", None)
    if not patch and api_keys is None:
        raise BridgeError.invalid_argument(
            "profile update must include at least one field.",
            field="patch",
        )
    return profile_id, patch, api_keys


def coerce_model_profile_patch(patch: Mapping[str, object]) -> dict[str, object]:
    valid_fields = set(ModelConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    unknown = set(patch) - valid_fields
    if unknown:
        raise BridgeError.invalid_argument(
            f"unknown profile field(s): {sorted(unknown)!r}",
            details={"unknown_fields": sorted(unknown)},
        )
    coerced: dict[str, object] = {}
    for key, value in patch.items():
        if key in {"display_name", "model_id"} and looks_like_placeholder_value(value):
            raise BridgeError.invalid_argument(
                f"{key} contains placeholder text.",
                field=key,
            )
        if key == "provider_format" and isinstance(value, str):
            try:
                coerced[key] = ProviderFormat(value)
            except ValueError as exc:
                raise BridgeError.invalid_argument(str(exc), field=key) from exc
        elif key == "thinking_level" and isinstance(value, str):
            try:
                coerced[key] = ThinkingLevel(value)
            except ValueError as exc:
                raise BridgeError.invalid_argument(str(exc), field=key) from exc
        elif key == "custom_headers" and isinstance(value, list):
            coerced[key] = tuple(
                (str(pair[0]), str(pair[1]))
                for pair in value
                if isinstance(pair, (list, tuple)) and len(pair) == 2
            )
        else:
            coerced[key] = value
    return coerced


def coerce_api_keys(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise BridgeError.invalid_argument(
            "api_keys must be a list of strings.",
            field="api_keys",
        )
    keys: list[str] = []
    for raw in value:
        if not isinstance(raw, str):
            raise BridgeError.invalid_argument(
                "api_keys must be a list of strings.",
                field="api_keys",
            )
        key = raw.strip()
        if key:
            keys.append(key)
    return tuple(keys)


def model_profile_body(
    profile: ModelConfig,
    *,
    profile_store: ModelProfileStore,
) -> dict[str, object]:
    body = profile.to_dict()
    body.pop("api_keys", None)
    body["api_key_configured"] = bool(profile.api_keys)
    body["api_key_status"] = profile_store.api_key_status(profile.id)
    return body


def generate_model_profile_id(body: Mapping[str, object]) -> str:
    seed = str(body.get("display_name") or body.get("model_id") or "profile")
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in seed).strip("-")
    if not slug:
        slug = "profile"
    return f"{slug}-{token_hex(3)}"
