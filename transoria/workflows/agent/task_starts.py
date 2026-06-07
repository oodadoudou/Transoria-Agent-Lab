"""Agent Lab wrappers that start existing workflow tasks from explicit drafts."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Mapping

from transoria.agent.schemas import AgentWorkspaceState
from transoria.bridge.errors import BridgeError
from transoria.bridge.task_service import (
    TaskService,
    _coerce_language,
    _coerce_text_preserve_rules,
    _coerce_translation_replacements,
    _count_source_chars,
    _derive_chunk_size,
    _effective_glossary_chunk_token_limit,
    _ensure_output_dir,
    _require_directory,
    _require_distinct_translation_folders,
    _require_input_with_supported_files,
)
from transoria.model_profiles import ModelProfileStore
from transoria.prompts import PromptKind, PromptPresetStore
from transoria.settings import SettingsStore
from transoria.workflows.glossary.config import GlossaryConfig
from transoria.workflows.glossary_review.config import GlossaryReviewConfig
from transoria.workflows.glossary_review.loader import (
    normalize_output_filename,
)
from transoria.workflows.translation.config import TranslationConfig
from transoria.workflows.translation.rules import Glossary


_TRANSLATION_CHUNK_CHAR_BUDGET = 18_000

_MODEL_SLOT_BY_DRAFT_KIND: dict[str, str] = {
    "start_glossary_task": "term_extract",
    "start_glossary_review_task": "term_review",
    "start_translation_task": "translation",
}

_PROMPT_KIND_BY_STAGE: dict[str, PromptKind] = {
    "translation": PromptKind.TRANSLATION,
    "term_extract": PromptKind.GLOSSARY,
    "term_review": PromptKind.GLOSSARY_REVIEW,
}


def start_agent_task(
    *,
    draft_kind: str,
    payload: Mapping[str, object],
    state: AgentWorkspaceState,
    task_service: TaskService,
    settings_store: SettingsStore,
    profile_store: ModelProfileStore,
    cache_root: Path,
    request_id: str,
) -> dict[str, object]:
    if draft_kind == "start_glossary_task":
        config = _build_glossary_config(
            payload=payload,
            state=state,
            settings_store=settings_store,
            profile_store=profile_store,
            cache_root=cache_root,
        )
        return task_service.start_glossary_with_config(config, request_id=request_id)
    if draft_kind == "start_glossary_review_task":
        config = _build_glossary_review_config(
            payload=payload,
            state=state,
            task_service=task_service,
            settings_store=settings_store,
            profile_store=profile_store,
            cache_root=cache_root,
            request_id=request_id,
        )
        return task_service.start_glossary_review_with_config(
            config,
            request_id=request_id,
        )
    if draft_kind == "start_translation_task":
        config = _build_translation_config(
            payload=payload,
            state=state,
            task_service=task_service,
            settings_store=settings_store,
            profile_store=profile_store,
            cache_root=cache_root,
        )
        return task_service.start_translation_with_config(
            config,
            request_id=request_id,
        )
    raise BridgeError.invalid_argument(
        f"unsupported task-start draft kind: {draft_kind!r}",
        details={"kind": draft_kind},
    )


def _build_translation_config(
    *,
    payload: Mapping[str, object],
    state: AgentWorkspaceState,
    task_service: TaskService,
    settings_store: SettingsStore,
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> TranslationConfig:
    settings = settings_store.load_all()
    translation = settings.translation
    input_dir = _require_directory(_required_str(payload, "input_dir"), field="input_dir")
    _require_input_with_supported_files(input_dir, field="input_dir")
    output_dir = _ensure_output_dir(_required_str(payload, "output_dir"), field="output_dir")
    _require_distinct_translation_folders(input_dir, output_dir)
    source_language = _coerce_language(
        _required_str(payload, "source_language"),
        field="source_language",
    )
    target_language = _coerce_language(
        _required_str(payload, "target_language"),
        field="target_language",
    )
    model, preset = _resolve_stage_model_and_prompt(
        draft_kind="start_translation_task",
        state=state,
        profile_store=profile_store,
        cache_root=cache_root,
        timeout_seconds=translation.timeout_seconds,
    )
    glossary = _translation_glossary_from_payload(payload, task_service=task_service)
    if not glossary.entries:
        glossary = Glossary.from_records(translation.translation_glossary)
    return TranslationConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        source_language=source_language,
        target_language=target_language,
        model=model,
        prompt_preset=preset,
        glossary=glossary,
        text_preserve_rules=_coerce_text_preserve_rules(
            translation.text_preserve_rules
        ),
        pre_replacements=_coerce_translation_replacements(
            translation.pre_replacements
        ),
        post_replacements=_coerce_translation_replacements(
            translation.post_replacements
        ),
        bilingual_enabled=translation.bilingual_enabled,
        bilingual_dedup_when_same=translation.bilingual_dedupe_identical,
        bilingual_subfolder=translation.bilingual_subfolder_name
        or "bilingual outputs",
        context_line_count=max(0, int(translation.context_lines)),
        chunk_size=_derive_chunk_size(model.input_token_limit),
        chunk_token_limit=_TRANSLATION_CHUNK_CHAR_BUDGET,
        token_counter=_count_source_chars,
        low_confidence_max_retries=max(
            0, int(translation.low_confidence_max_retries)
        ),
        auto_retry_max_rounds=max(
            0, min(100, int(translation.auto_retry_max_rounds))
        ),
    )


def _build_glossary_config(
    *,
    payload: Mapping[str, object],
    state: AgentWorkspaceState,
    settings_store: SettingsStore,
    profile_store: ModelProfileStore,
    cache_root: Path,
) -> GlossaryConfig:
    settings = settings_store.load_all()
    glossary = settings.glossary
    input_dir = _require_directory(_required_str(payload, "input_dir"), field="input_dir")
    _require_input_with_supported_files(input_dir, field="input_dir")
    output_dir = _ensure_output_dir(_required_str(payload, "output_dir"), field="output_dir")
    model, preset = _resolve_stage_model_and_prompt(
        draft_kind="start_glossary_task",
        state=state,
        profile_store=profile_store,
        cache_root=cache_root,
        timeout_seconds=glossary.timeout_seconds,
    )
    return GlossaryConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        source_language=_coerce_language(
            _required_str(payload, "source_language"),
            field="source_language",
        ),
        target_language=_coerce_language(
            _required_str(payload, "target_language"),
            field="target_language",
        ),
        model=model,
        prompt_preset=preset,
        reference_example_limit=max(0, int(glossary.reference_examples_per_term)),
        max_term_display_length=max(1, int(glossary.max_term_display_length)),
        min_frequency=max(1, int(glossary.minimum_frequency)),
        chunk_token_limit=_effective_glossary_chunk_token_limit(
            int(glossary.chunk_token_limit)
        ),
        allow_src_eq_dst=bool(glossary.keep_identical_src_dst),
        combine_folder_glossary=bool(glossary.merge_folder_glossary),
        normalize_widths=bool(glossary.normalize_widths),
        novel_background=str(
            payload.get("novel_background")
            if isinstance(payload.get("novel_background"), str)
            else glossary.novel_background
        ),
    )


def _build_glossary_review_config(
    *,
    payload: Mapping[str, object],
    state: AgentWorkspaceState,
    task_service: TaskService,
    settings_store: SettingsStore,
    profile_store: ModelProfileStore,
    cache_root: Path,
    request_id: str,
) -> GlossaryReviewConfig:
    settings = settings_store.load_all()
    review = settings.glossary_review
    output_filename = normalize_output_filename(review.output_filename)
    artifact = _glossary_artifact_from_task(payload, task_service=task_service)
    glossary_task_id = _required_str(payload, "glossary_task_id")
    input_dir, xlsx_path, reference_paths = _prepare_glossary_review_input(
        artifact=artifact,
        cache_root=cache_root,
        glossary_task_id=glossary_task_id,
        request_id=request_id,
    )
    model, preset = _resolve_stage_model_and_prompt(
        draft_kind="start_glossary_review_task",
        state=state,
        profile_store=profile_store,
        cache_root=cache_root,
        timeout_seconds=review.timeout_seconds,
    )
    return GlossaryReviewConfig(
        input_dir=input_dir,
        selected_xlsx_path=xlsx_path,
        selected_reference_paths=reference_paths,
        output_filename=output_filename,
        novel_background=str(
            payload.get("novel_background")
            if isinstance(payload.get("novel_background"), str)
            else review.novel_background
        ),
        review_rounds=max(1, int(review.review_rounds)),
        batch_size=max(1, int(review.batch_size)),
        retry_attempts=max(0, int(review.retry_attempts)),
        model=model,
        prompt_preset=preset,
    )


def _resolve_stage_model_and_prompt(
    *,
    draft_kind: str,
    state: AgentWorkspaceState,
    profile_store: ModelProfileStore,
    cache_root: Path,
    timeout_seconds: int,
):
    stage = _MODEL_SLOT_BY_DRAFT_KIND[draft_kind]
    model_id = state.stage_model_ids.get(stage)
    if not model_id:
        raise BridgeError.invalid_argument(
            f"stage model is not configured for {stage}.",
            field=f"stage_model_ids.{stage}",
        )
    model = profile_store.get(model_id)
    if model is None:
        raise BridgeError.not_found(
            f"profile {model_id!r} does not exist.",
            details={"id": model_id, "stage": stage},
        )
    if not model.api_keys:
        raise BridgeError.invalid_argument(
            f"profile {model_id!r} has no API key configured.",
            field=f"stage_model_ids.{stage}",
            details={"reason": "missing_api_key"},
        )
    prompt_id = state.stage_prompt_ids.get(stage)
    if not prompt_id:
        raise BridgeError.invalid_argument(
            f"stage prompt is not configured for {stage}.",
            field=f"stage_prompt_ids.{stage}",
        )
    preset = PromptPresetStore(
        path=cache_root / f"prompts.{_PROMPT_KIND_BY_STAGE[stage].value}.json",
        kind=_PROMPT_KIND_BY_STAGE[stage],
    ).get_active(prompt_id)
    return replace(model, timeout_seconds=float(timeout_seconds)), preset


def _translation_glossary_from_payload(
    payload: Mapping[str, object], *, task_service: TaskService
) -> Glossary:
    review_task_id = payload.get("glossary_review_task_id")
    if isinstance(review_task_id, str) and review_task_id.strip():
        final = task_service.read_glossary_review_final(task_id=review_task_id.strip())
        rows = final.get("rows")
        glossary = Glossary.from_records(rows if isinstance(rows, list) else [])
        if not glossary.entries:
            raise BridgeError.invalid_argument(
                f"glossary review task {review_task_id!r} has no usable final table rows.",
                field="glossary_review_task_id",
                details={"task_id": review_task_id},
            )
        return glossary
    task_id = payload.get("glossary_task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        return Glossary.empty()
    artifact = _glossary_artifact_from_task(payload, task_service=task_service)
    json_path = _required_artifact_path(artifact, "json_path")
    try:
        return Glossary.from_json_file(json_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise BridgeError.invalid_argument(
            f"cannot load glossary JSON from task {task_id!r}: {exc}",
            field="glossary_task_id",
            details={"task_id": task_id, "json_path": json_path},
        ) from exc


def _glossary_artifact_from_task(
    payload: Mapping[str, object], *, task_service: TaskService
) -> Mapping[str, object]:
    task_id = _required_str(payload, "glossary_task_id")
    result = task_service.read_artifacts(kind="glossary", task_id=task_id)
    artifact = result.get("combined_artifact")
    if isinstance(artifact, Mapping):
        return artifact
    per_novel = result.get("per_novel_artifacts")
    if isinstance(per_novel, list):
        for item in per_novel:
            if isinstance(item, Mapping):
                return item
    raise BridgeError.invalid_argument(
        f"glossary task {task_id!r} has no usable glossary artifact.",
        field="glossary_task_id",
        details={"task_id": task_id},
    )


def _required_artifact_path(artifact: Mapping[str, object], key: str) -> str:
    value = artifact.get(key)
    if not isinstance(value, str) or not value.strip():
        raise BridgeError.invalid_argument(
            f"glossary artifact is missing {key}.",
            field="glossary_task_id",
        )
    return value


def _prepare_glossary_review_input(
    *,
    artifact: Mapping[str, object],
    cache_root: Path,
    glossary_task_id: str,
    request_id: str,
) -> tuple[Path, Path, tuple[Path, ...]]:
    source_xlsx = _require_existing_artifact_file(
        artifact,
        "xlsx_path",
        suffix=".xlsx",
    )
    source_refs = _require_existing_artifact_file(
        artifact,
        "references_path",
        suffix=".txt",
    )
    review_input_dir = (
        cache_root
        / "agent_lab"
        / "review_inputs"
        / f"{_safe_cache_token(glossary_task_id)}-{_safe_cache_token(request_id)}"
    )
    try:
        review_input_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise BridgeError.conflict(
            "agent glossary review input cache already exists.",
            details={"input_dir": str(review_input_dir)},
        ) from exc
    xlsx_path = review_input_dir / source_xlsx.name
    refs_path = review_input_dir / source_refs.name
    shutil.copy2(source_xlsx, xlsx_path)
    shutil.copy2(source_refs, refs_path)
    return review_input_dir, xlsx_path, (refs_path,)


def _require_existing_artifact_file(
    artifact: Mapping[str, object],
    key: str,
    *,
    suffix: str,
) -> Path:
    path = Path(_required_artifact_path(artifact, key))
    if not path.is_file() or path.suffix.lower() != suffix:
        raise BridgeError.invalid_argument(
            f"glossary artifact {key} points to an invalid {suffix} file.",
            field="glossary_task_id",
            details={key: str(path)},
        )
    return path


def _required_str(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise BridgeError.invalid_argument(f"{field} is required.", field=field)
    return value.strip()


def _safe_cache_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return token or "task"


__all__ = ["start_agent_task"]
