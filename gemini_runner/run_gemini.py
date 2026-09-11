#!/usr/bin/env python3
"""Run benchmark tasks with Gemini Computer Use."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

if __package__ in (None, ""):
    # batch.py invokes this file by path, so it starts life as a top-level
    # module with no parent package. Its siblings (runner, browser_agent,
    # playwright_computer) import each other relatively, which needs one —
    # without this they raise "attempted relative import with no known parent
    # package". Adopting the package makes both styles resolve to the same
    # modules instead of loading a second, package-less copy.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "gemini_runner"

from .runner import GeminiRunConfig, run_tasks
from .task_selection import (
    apply_task_range,
    filter_tasks_by_application,
    filter_tasks_by_id,
    list_tasks,
    select_random_tasks,
)

load_dotenv()

THIS_FILE_PATH = Path(__file__).parent
TASKS_PATH = THIS_FILE_PATH / "tasks" / "eval"


def _read_optional_file(text: str, file_path: str, label: str) -> str:
    if text and file_path:
        raise ValueError(f"Provide either --{label} or --{label}_file, not both.")
    if file_path:
        return Path(file_path).expanduser().resolve().read_text(encoding="utf-8")
    return text or ""


def run(
    model: str = "gemini-2.5-computer-use-preview-10-2025",
    task: str = "",
    run_all: bool = False,
    application: str = "",
    task_file: str = "",
    task_range: str = "",
    run_random: bool = False,
    max_tasks: int = 0,
    iterations: int = 1,
    headless: bool = True,
    concurrent: bool = False,
    workers: int = 0,
    max_steps: int = 50,
    initial_delay: float = 0.0,
    results_dir: str = "results/results_gemini",
    url: str = "",
    post_run_url: str = "",
    js_snippet_file: str = "",
    system_prompt: str = "",
    system_prompt_file: str = "",
    prefix_prompt: str = "",
    prefix_prompt_file: str = "",
    viewport_width: int = 1280,
    viewport_height: int = 800,
    inject_proxy_select: bool = False,
    use_vertexai: bool = False,
    vertexai_project: str = "",
    vertexai_location: str = "",
    continue_run: bool = False,
    verbose: bool = False,
) -> dict:
    """Run Gemini Computer Use tasks and save evaluator-compatible artifacts."""
    if use_vertexai:
        os.environ["USE_VERTEXAI"] = "true"
    if vertexai_project:
        os.environ["VERTEXAI_PROJECT"] = vertexai_project
    if vertexai_location:
        os.environ["VERTEXAI_LOCATION"] = vertexai_location

    if not os.environ.get("GEMINI_API_KEY") and os.environ.get("USE_VERTEXAI", "").lower() not in {"true", "1"}:
        raise ValueError("Set GEMINI_API_KEY, or set USE_VERTEXAI=true with VERTEXAI_PROJECT and VERTEXAI_LOCATION.")

    task_source = Path(task_file).expanduser().resolve() if task_file else TASKS_PATH
    if not task_source.exists():
        raise ValueError(f"Task source not found: {task_source}")

    tasks = list_tasks(task_source)
    if not tasks:
        raise ValueError(f"No tasks found under: {task_source}")

    if application:
        tasks = filter_tasks_by_application(tasks, application)
    if task:
        tasks = filter_tasks_by_id(tasks, task)
    if task_range:
        tasks = apply_task_range(tasks, task_range)

    if run_random:
        tasks = select_random_tasks(tasks, max_tasks if max_tasks > 0 else None)
    elif max_tasks > 0:
        tasks = tasks[:max_tasks]

    if not run_all and not task and not run_random and max_tasks == 0:
        tasks = tasks[:1]

    if not tasks:
        raise ValueError("No tasks selected after applying filters.")

    js_snippet_path = Path(js_snippet_file).expanduser().resolve() if js_snippet_file else None
    js_snippet_source = js_snippet_path.read_text(encoding="utf-8") if js_snippet_path else None
    system_prompt_append = _read_optional_file(system_prompt, system_prompt_file, "system_prompt")
    prefix_prompt_text = _read_optional_file(prefix_prompt, prefix_prompt_file, "prefix_prompt")

    assets_dir = THIS_FILE_PATH / "assets"
    proxy_js = (assets_dir / "proxy-select.js").read_text(encoding="utf-8") if inject_proxy_select else ""
    proxy_css = (assets_dir / "proxy-select.css").read_text(encoding="utf-8") if inject_proxy_select else ""

    config = GeminiRunConfig(
        model=model,
        headless=headless,
        max_steps=max_steps,
        concurrent=concurrent,
        workers=workers,
        results_dir=results_dir,
        post_run_url=post_run_url or None,
        post_run_js_snippet=js_snippet_source,
        post_run_js_snippet_path=str(js_snippet_path) if js_snippet_path else None,
        system_prompt_append=system_prompt_append or None,
        prefix_prompt=prefix_prompt_text or None,
        initial_delay=initial_delay,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        inject_proxy_select=inject_proxy_select,
        proxy_select_js=proxy_js,
        proxy_select_css=proxy_css,
        verbose=verbose,
        task_file=str(task_source),
        url=url or None,
        js_snippet_file=str(js_snippet_path) if js_snippet_path else None,
        iterations=iterations,
        continue_run=continue_run,
    )
    entries, manifest_path, skipped_existing = run_tasks(tasks, config)
    completed = sum(1 for entry in entries if entry["status"] == "completed")
    truncated = sum(1 for entry in entries if entry["status"] == "truncated")
    errored = sum(1 for entry in entries if entry["status"] == "error")

    return {
        "results_dir": str(Path(results_dir).expanduser().resolve()),
        "manifest_path": str(manifest_path),
        "runs": len(entries),
        "skipped_existing": skipped_existing,
        "completed": completed,
        "truncated": truncated,
        "errors": errored,
        "selected_tasks": [task_obj["id"] for task_obj in tasks],
    }


if __name__ == "__main__":
    fire.Fire(run)
