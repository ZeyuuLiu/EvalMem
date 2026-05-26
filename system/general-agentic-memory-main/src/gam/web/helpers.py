# -*- coding: utf-8 -*-
"""
GAM Web Application - Helper utilities

Shared helper functions, global state, and constants used by route handlers.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from werkzeug.utils import secure_filename

from ..readers import PdfReader, TxtReader


# ---------------------------------------------------------------------------
# Global state for background tasks
# ---------------------------------------------------------------------------
task_queues: Dict[str, Any] = {}
task_results: Dict[str, Any] = {}

# Default base output path
DEFAULT_OUTPUT_BASE = "./temp/web_test"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_timestamp_dir() -> str:
    """生成时间戳目录名：年-月-日-时-分-秒"""
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def parse_int_param(value: str | None, default: int, min_value: int, max_value: int) -> int:
    """Parse integer params with bounds; fallback to *default* on invalid."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed < min_value or parsed > max_value:
        return default
    return parsed


def parse_bool_param(value: str | None, default: bool = True) -> bool:
    """Parse boolean params from string values."""
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def read_uploaded_files(uploaded_files: List[Any]) -> List[Dict[str, str]]:
    """Read uploaded files and return ``filename`` + ``content``."""
    entries: List[Dict[str, str]] = []
    for file in uploaded_files:
        if not file or not getattr(file, "filename", None):
            continue
        filename = secure_filename(file.filename)
        ext = Path(filename).suffix.lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            if ext == '.pdf':
                reader = PdfReader()
                content = reader.read(tmp_path)
            elif ext in ['.txt', '.md']:
                reader = TxtReader()
                content = reader.read(tmp_path)
            else:
                with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
        finally:
            os.unlink(tmp_path)

        entries.append({
            "filename": filename,
            "stem": Path(filename).stem,
            "content": content or "",
        })

    return entries


def next_chunks_dir(base_dir: Path) -> Path:
    """Return next chunks directory path under *base_dir* (chunks_0, chunks_1, …)."""
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
    max_index = -1
    for item in base_dir.iterdir():
        if not item.is_dir():
            continue
        if item.name.startswith("chunks_"):
            suffix = item.name.split("chunks_", 1)[-1]
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))
    return base_dir / f"chunks_{max_index + 1}"
