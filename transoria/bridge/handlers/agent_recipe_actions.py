"""Recipe payload helpers for Agent Lab."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from transoria.agent.schemas import (
    MODEL_SLOTS,
    PROMPT_SLOTS,
    AgentRecipe,
    AgentWorkspaceState,
)
from transoria.bridge.errors import BridgeError
from transoria.bridge.handlers.agent_inventory import prompt_store_for
from transoria.llm.config import ModelConfig
from transoria.model_profiles import ModelProfileStore
from transoria.prompts import PromptKind

PROMPT_KIND_BY_SLOT = {
    "translation": PromptKind.TRANSLATION,
    "term_extract": PromptKind.GLOSSARY,
    "term_review": PromptKind.GLOSSARY_REVIEW,
}

MAX_RECIPE_NAME_LENGTH = 120
MAX_RECIPE_DESCRIPTION_LENGTH = 400


def recipe_body_from_payload(payload: Mapping[str, object]) -> dict[str, object]:
    raw_recipe = first_present(
        payload,
        ("recipe", "recipe_config", "recipeConfig", "config", "preset", "预设", "配方"),
    )
    body = dict(raw_recipe) if isinstance(raw_recipe, Mapping) else dict(payload)
    if "name" not in body:
        for key in (
            "recipe_name",
            "recipeName",
            "preset_name",
            "presetName",
            "config_name",
            "configName",
            "configuration_name",
            "configurationName",
            "display_name",
            "displayName",
            "title",
            "名称",
            "名字",
            "配方名称",
            "预设名称",
        ):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                body["name"] = value
                break
    return body


def coerce_recipe_update_payload(
    payload: Mapping[str, object],
    *,
    profile_store: ModelProfileStore | ModelProfileReader,
    cache_root: Path,
) -> tuple[
    str,
    dict[str, str],
    dict[str, str | None] | None,
    dict[str, str | None] | None,
]:
    body = recipe_body_from_payload(payload)
    recipe_id = str(body.get("recipe_id") or body.get("id") or "").strip()
    if not recipe_id:
        raise BridgeError.invalid_argument(
            "recipe_id is required.",
            field="recipe_id",
        )
    fields: dict[str, str] = {}
    if "name" in body:
        name = str(body.get("name") or "").strip()
        if not name:
            raise BridgeError.invalid_argument("name is required.", field="name")
        if len(name) > MAX_RECIPE_NAME_LENGTH:
            raise BridgeError.invalid_argument(
                "name is too long.",
                field="name",
                details={"max_length": MAX_RECIPE_NAME_LENGTH},
            )
        fields["name"] = name
    if "description" in body:
        description = str(body.get("description") or "").strip()
        if len(description) > MAX_RECIPE_DESCRIPTION_LENGTH:
            raise BridgeError.invalid_argument(
                "description is too long.",
                field="description",
                details={"max_length": MAX_RECIPE_DESCRIPTION_LENGTH},
            )
        fields["description"] = description
    stage_models = (
        coerce_model_slots(body.get("stage_model_ids"), profile_store=profile_store)
        if "stage_model_ids" in body
        else None
    )
    stage_prompts = (
        coerce_prompt_slots(body.get("stage_prompt_ids"), cache_root=cache_root)
        if "stage_prompt_ids" in body
        else None
    )
    if not fields and stage_models is None and stage_prompts is None:
        raise BridgeError.invalid_argument(
            "recipe update must include at least one field.",
            field="recipe_id",
        )
    return recipe_id, fields, stage_models, stage_prompts


def require_recipe_from_payload(
    payload: Mapping[str, object],
    *,
    state: AgentWorkspaceState,
) -> AgentRecipe:
    recipe_id = str(payload.get("recipe_id") or payload.get("id") or "").strip()
    if not recipe_id:
        raise BridgeError.invalid_argument(
            "recipe_id is required.",
            field="recipe_id",
        )
    recipe = state.get_recipe(recipe_id)
    if recipe is None:
        raise BridgeError.not_found(
            f"recipe {recipe_id!r} does not exist.",
            details={"recipe_id": recipe_id},
        )
    return recipe


def coerce_recipe_payload(
    payload: Mapping[str, object],
    *,
    profile_store: ModelProfileStore | ModelProfileReader,
    cache_root: Path,
) -> tuple[str, str, dict[str, str | None], dict[str, str | None]]:
    body = recipe_body_from_payload(payload)
    name = str(body.get("name") or "").strip()
    if not name:
        raise BridgeError.invalid_argument("name is required.", field="name")
    if len(name) > MAX_RECIPE_NAME_LENGTH:
        raise BridgeError.invalid_argument(
            "name is too long.",
            field="name",
            details={"max_length": MAX_RECIPE_NAME_LENGTH},
        )
    description = str(body.get("description") or "").strip()
    if len(description) > MAX_RECIPE_DESCRIPTION_LENGTH:
        raise BridgeError.invalid_argument(
            "description is too long.",
            field="description",
            details={"max_length": MAX_RECIPE_DESCRIPTION_LENGTH},
        )
    stage_models = (
        coerce_model_slots(body.get("stage_model_ids"), profile_store=profile_store)
        if "stage_model_ids" in body
        else {}
    )
    stage_prompts = (
        coerce_prompt_slots(body.get("stage_prompt_ids"), cache_root=cache_root)
        if "stage_prompt_ids" in body
        else {}
    )
    return name, description, stage_models, stage_prompts


def coerce_model_slots(
    value: object,
    *,
    profile_store: ModelProfileStore | ModelProfileReader,
) -> dict[str, str | None]:
    if not isinstance(value, Mapping):
        raise BridgeError.invalid_argument(
            "stage_model_ids must be an object.",
            field="stage_model_ids",
        )
    result: dict[str, str | None] = {}
    for slot, raw in value.items():
        slot_name = str(slot)
        if slot_name not in MODEL_SLOTS:
            raise BridgeError.invalid_argument(
                f"unknown model slot: {slot_name!r}",
                field="stage_model_ids",
            )
        result[slot_name] = coerce_model_id(
            raw,
            profile_store=profile_store,
            field=f"stage_model_ids.{slot_name}",
        )
    return result


def coerce_prompt_slots(value: object, *, cache_root: Path) -> dict[str, str | None]:
    if not isinstance(value, Mapping):
        raise BridgeError.invalid_argument(
            "stage_prompt_ids must be an object.",
            field="stage_prompt_ids",
        )
    result: dict[str, str | None] = {}
    for slot, raw in value.items():
        slot_name = str(slot)
        if slot_name not in PROMPT_SLOTS:
            raise BridgeError.invalid_argument(
                f"unknown prompt slot: {slot_name!r}",
                field="stage_prompt_ids",
            )
        prompt_id = optional_str(raw)
        if prompt_id is not None:
            kind = PROMPT_KIND_BY_SLOT[slot_name]
            store = prompt_store_for(cache_root, kind)
            if not any(p.id == prompt_id for p in store.load()):
                raise BridgeError.not_found(
                    f"prompt preset {prompt_id!r} does not exist.",
                    details={"id": prompt_id, "slot": slot_name},
                )
        result[slot_name] = prompt_id
    return result


def coerce_model_id(
    value: object,
    *,
    profile_store: ModelProfileStore | ModelProfileReader,
    field: str,
) -> str | None:
    profile_id = optional_str(value)
    if profile_id is None:
        return None
    if profile_store.get(profile_id) is None:
        raise BridgeError.not_found(
            f"profile {profile_id!r} does not exist.",
            details={"id": profile_id, "field": field},
        )
    return profile_id


def optional_str(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    if value is None:
        return None
    raise BridgeError.invalid_argument("value must be a string or null.")


def first_present(payload: Mapping[str, object], keys: tuple[str, ...]) -> object | None:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


class ModelProfileReader(Protocol):
    def get(self, profile_id: str) -> ModelConfig | None: ...
