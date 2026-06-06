"""JSON cache for the editable glossary review final table."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Mapping

from transoria.workflows.glossary_review.loader import load_glossary_xlsx

FINAL_TABLE_CACHE_FILENAME = "final-table.json"
FINAL_TABLE_CACHE_VERSION = 1


def final_table_cache_path(task_dir: Path) -> Path:
    return task_dir / FINAL_TABLE_CACHE_FILENAME


def read_or_create_final_table_cache(
    *,
    task_dir: Path,
    task_id: str,
    source_path: Path,
) -> dict[str, object]:
    path = final_table_cache_path(task_dir)
    if path.exists():
        return _read_cache(path)
    payload = final_table_cache_payload_from_xlsx(
        task_id=task_id,
        source_path=source_path,
    )
    write_final_table_cache(task_dir=task_dir, payload=payload)
    return payload


def final_table_cache_payload_from_xlsx(
    *,
    task_id: str,
    source_path: Path,
) -> dict[str, object]:
    loaded = load_glossary_xlsx(source_path)
    return {
        "version": FINAL_TABLE_CACHE_VERSION,
        "task_id": task_id,
        "source_path": str(source_path),
        "updated_at": _timestamp(),
        "rows": [
            {
                "row_index": row.row_index,
                "src": row.src,
                "dst": row.dst,
                "info": row.info,
                "frequency": row.frequency,
            }
            for row in loaded.rows
        ],
    }


def write_final_table_cache(
    *,
    task_dir: Path,
    payload: Mapping[str, object],
) -> Path:
    path = final_table_cache_path(task_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["version"] = FINAL_TABLE_CACHE_VERSION
    body["updated_at"] = _timestamp()
    _atomic_write_text(
        path,
        json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True),
    )
    return path


def rows_from_final_table_cache(payload: Mapping[str, object]) -> list[dict[str, object]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return []
    normalized: list[dict[str, object]] = []
    for index, raw in enumerate(rows, start=2):
        if not isinstance(raw, Mapping):
            continue
        row_index = raw.get("row_index")
        try:
            row_index_int = int(row_index) if row_index is not None else index
        except (TypeError, ValueError):
            row_index_int = index
        normalized.append(
            {
                "row_index": max(2, row_index_int),
                "src": str(raw.get("src") or "").strip(),
                "dst": str(raw.get("dst") or "").strip(),
                "info": str(raw.get("info") or "").strip(),
                "frequency": _coerce_frequency(raw.get("frequency")),
            }
        )
    return normalized


def _read_cache(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"version": FINAL_TABLE_CACHE_VERSION, "rows": []}
    return payload


def _coerce_frequency(value: object) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
