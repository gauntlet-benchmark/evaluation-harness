"""
This module provides functionality to load and manage task configurations.
"""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Repo-root `tasks/`, holding the YAML suites for the five benchmark apps.
TASK_DIRECTORIES = [Path(CURRENT_DIR).parents[4] / "tasks"]


def _iter_task_files(root: Path):
    if root.is_file():
        if root.suffix.lower() in {".json", ".yaml", ".yml"}:
            yield root.resolve()
        return

    if not root.exists():
        return

    for suffix in ("*.json", "*.yaml", "*.yml"):
        for file_path in root.rglob(suffix):
            if file_path.is_file():
                yield file_path.resolve()


def _load_file_data(file_path: Path) -> Any:
    with open(file_path, "r", encoding="utf-8") as file:
        if file_path.suffix.lower() == ".json":
            return json.load(file)
        return yaml.safe_load(file) or {}


def _default_task_id(source_path: Path, case_index: int) -> str:
    return f"{source_path.stem}-{case_index + 1}"


def _normalize_task(raw_task: Dict[str, Any], source_path: Path, case_index: int = 0) -> Dict[str, Any]:
    website = raw_task.get("website") or {}
    if not isinstance(website, dict):
        website = {}

    task_id = raw_task.get("id") or _default_task_id(source_path, case_index)
    goal = raw_task.get("goal") or raw_task.get("prompt") or raw_task.get("description") or ""
    website_id = website.get("id") or source_path.stem

    return {
        "id": task_id,
        "goal": goal,
        "website": {
            "id": website_id,
            "name": website.get("name") or website_id,
            "url": website.get("url", ""),
            **website,
        },
        "difficulty": raw_task.get("difficulty", ""),
        "description": raw_task.get("description", ""),
        "prompt": raw_task.get("prompt", ""),
        "gt": raw_task.get("gt"),
        "llm_judge_gt": raw_task.get("llm_judge_gt"),
        "_task_file": str(source_path.resolve()),
    }


def load_tasks_from_file(file_path: str | Path) -> list[Dict[str, Any]]:
    source_path = Path(file_path).expanduser().resolve()
    raw_data = _load_file_data(source_path)

    if isinstance(raw_data, dict) and isinstance(raw_data.get("tasks"), list):
        return [
            _normalize_task(task_case or {}, source_path, index)
            for index, task_case in enumerate(raw_data["tasks"])
        ]

    if isinstance(raw_data, list):
        return [
            _normalize_task(task_case or {}, source_path, index)
            for index, task_case in enumerate(raw_data)
        ]

    if isinstance(raw_data, dict):
        return [_normalize_task(raw_data, source_path, 0)]

    return []


def _find_task_by_id(task_id: str) -> Dict[str, Any]:
    normalized_task_id = task_id.split(".", 1)[1] if task_id.startswith("eval.") else task_id

    for task_dir in TASK_DIRECTORIES:
        for file_path in _iter_task_files(Path(task_dir)):
            for task_config in load_tasks_from_file(file_path):
                candidate_id = task_config.get("id", "")
                namespaced_id = f"{file_path.stem}.{candidate_id}"
                if normalized_task_id in {candidate_id, namespaced_id}:
                    return task_config

    raise FileNotFoundError(f"Task configuration file not found for task id: {task_id}")


@dataclass
class Task:
    id: str
    start_url: str
    goal: str
    difficulty: str = ""
    description: str = ""

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


class TaskConfig:
    def __init__(self, input_source: str, task_source: str | None = None, is_path: bool = False) -> None:
        source_path = Path(input_source)
        if task_source:
            tasks = load_tasks_from_file(task_source)
            matching_tasks = [task for task in tasks if task.get("id") == input_source]
            if not matching_tasks:
                raise FileNotFoundError(f"Task id {input_source} not found in task file: {task_source}")
            self.config_json = matching_tasks[0]
            self.id = self.config_json.get("id", "")
        elif source_path.exists() and source_path.suffix.lower() in {".json", ".yaml", ".yml"}:
            tasks = load_tasks_from_file(source_path)
            if len(tasks) != 1:
                raise ValueError(
                    f"Task file {source_path} contains {len(tasks)} tasks. Load it by task id instead of direct file path."
                )
            self.config_json = tasks[0]
            self.id = self.config_json.get("id", "")
        else:
            self.id = input_source
            self.config_json = _find_task_by_id(self.id)

        if not self.is_valid_config():
            raise ValueError(f"Invalid task configuration for task ID: {self.id}")

        url = self.config_json.get("website", {}).get("url", "")

        task_fields = self.config_json.copy()
        for key in ("website", "prompt", "gt", "llm_judge_gt", "_task_file"):
            task_fields.pop(key, None)

        self.task = Task(start_url=url, **task_fields)

    def to_json(self) -> Dict[str, Any]:
        return self.task.to_json()

    def get_task_id(self) -> str:
        return self.task.id

    def get_start_url(self) -> str:
        return self.task.start_url

    def get_goal(self) -> str:
        return self.task.goal

    def is_valid_config(self) -> bool:
        return bool(self.config_json.get("id"))
