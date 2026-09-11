"""Task loading and filtering helpers for Gemini runner."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - optional fallback
    yaml = None


def _load_file_data(file_path: Path) -> Any:
    if file_path.suffix.lower() == ".json":
        return json.loads(file_path.read_text(encoding="utf-8"))
    if file_path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML task files.")
        return yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    raise ValueError(f"Unsupported task file format: {file_path}")


def _normalize_task(raw_task: dict[str, Any], source_path: Path, idx: int) -> dict[str, Any]:
    website = raw_task.get("website") if isinstance(raw_task.get("website"), dict) else {}
    task_id = raw_task.get("id") or f"{source_path.stem}-{idx + 1}"
    prompt = raw_task.get("prompt") or raw_task.get("goal") or raw_task.get("description") or ""
    website_id = website.get("id") or source_path.stem
    return {
        "id": task_id,
        "prompt": prompt,
        "goal": raw_task.get("goal") or prompt,
        "website": {
            "id": website_id,
            "name": website.get("name") or website_id,
            "url": website.get("url", ""),
            **website,
        },
        "gt": raw_task.get("gt"),
        "evals": raw_task.get("evals") or [],
        "_task_file": str(source_path.resolve()),
    }


def load_tasks_from_file(file_path: str | Path) -> list[dict[str, Any]]:
    source_path = Path(file_path).expanduser().resolve()
    data = _load_file_data(source_path)
    # The suites in tasks/ use `tasks:`; `test_cases:` is the older spelling.
    # Without the former the whole file parses as one task whose id defaults to
    # "<stem>-1", so a task range silently selects nothing.
    if isinstance(data, dict) and isinstance(data.get("tasks"), list):
        rows = data["tasks"]
    elif isinstance(data, dict) and isinstance(data.get("test_cases"), list):
        rows = data["test_cases"]
    elif isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = [data]
    else:
        raise ValueError(f"Unsupported task payload in {source_path}")
    return [_normalize_task(row, source_path, idx) for idx, row in enumerate(rows)]


def list_tasks(task_source: Path) -> list[dict[str, Any]]:
    if task_source.is_file():
        return load_tasks_from_file(task_source)

    tasks: list[dict[str, Any]] = []
    for pattern in ("**/*.json", "**/*.yaml", "**/*.yml"):
        for task_file in sorted(task_source.glob(pattern)):
            tasks.extend(load_tasks_from_file(task_file))
    return tasks


def filter_tasks_by_application(tasks: list[dict[str, Any]], application: str) -> list[dict[str, Any]]:
    return [task for task in tasks if task.get("website", {}).get("id") == application]


def filter_tasks_by_id(tasks: list[dict[str, Any]], task_id: Any) -> list[dict[str, Any]]:
    """Select tasks by id. Accepts one id or several.

    Several ids are what let a resume re-run an arbitrary set of tasks:
    ``task_range`` only expresses a contiguous slice, and the gaps left by failed
    runs rarely are. Selected tasks keep their order in the task file.

    The argument may arrive as a string *or* as a tuple/list: this runner's CLI is
    Fire, which splits ``--task a,b,c`` into a tuple before the value ever reaches
    here, unlike argparse which passes the raw string through. Handle both, or the
    multi-id form dies with "'tuple' object has no attribute 'split'".
    """
    parts: list[str] = []
    for chunk in (task_id if isinstance(task_id, (list, tuple)) else [task_id]):
        parts.extend(str(chunk).split(","))

    wanted = set()
    for part in parts:
        part = part.strip()
        if not part:
            continue
        wanted.add(part.split(".", 1)[1] if part.startswith("eval.") else part)
    return [task for task in tasks if task.get("id") in wanted]


def select_random_tasks(tasks: list[dict[str, Any]], max_tasks: int | None) -> list[dict[str, Any]]:
    if not max_tasks or max_tasks >= len(tasks):
        shuffled = tasks[:]
        random.shuffle(shuffled)
        return shuffled
    return random.sample(tasks, max_tasks)


def apply_task_range(tasks: list[dict[str, Any]], task_range: str) -> list[dict[str, Any]]:
    if not task_range:
        return tasks
    parts = task_range.split(":")
    if len(parts) == 1:
        idx = int(parts[0])
        return tasks[idx : idx + 1]
    if len(parts) == 2:
        start = int(parts[0]) if parts[0] else None
        end = int(parts[1]) if parts[1] else None
        return tasks[start:end]
    raise ValueError(f"Invalid task_range '{task_range}'.")
