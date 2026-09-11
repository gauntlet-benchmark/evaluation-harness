"""Execution runner for Gemini Computer Use benchmark tasks."""

from __future__ import annotations

import json
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

from .browser_agent import GeminiBrowserAgent
from .playwright_computer import PlaywrightComputer


@dataclass
class GeminiRunConfig:
    model: str = "gemini-2.5-computer-use-preview-10-2025"
    headless: bool = True
    max_steps: int = 50
    results_dir: str = "results"
    use_screenshot: bool = True
    concurrent: bool = False
    workers: int = 1
    post_run_url: str | None = None
    post_run_js_snippet: str | None = None
    post_run_js_snippet_path: str | None = None
    system_prompt_append: str | None = None
    prefix_prompt: str | None = None
    initial_delay: float = 0
    viewport_width: int = 1280
    viewport_height: int = 800
    inject_proxy_select: bool = False
    proxy_select_js: str = ""
    proxy_select_css: str = ""
    verbose: bool = False
    num_workers: int = 1
    task_file: str | None = None
    url: str | None = None
    js_snippet_file: str | None = None
    iterations: int = 1
    continue_run: bool = False


def _wait_until_settled(page, budget: float, poll: float = 2.0, stable_polls: int = 3) -> float:
    """Wait for the app to finish loading, up to ``budget`` seconds.

    ``initial_delay`` used to be read as a boolean — any positive value slept
    exactly 5s — which is far too short for apps that download a dataset before
    they are usable. But a fixed sleep is wrong too: the Flight Analyser has
    been measured at 270s on a cold fetch and 1.6s once the dataset is cached
    upstream, so any constant is simultaneously too short and too long.

    Instead treat ``initial_delay`` as a *maximum* and watch the page: while an
    app is loading, its body text keeps changing (spinners, "downloading — 37%"
    counters). Once that text holds steady across a few polls the app has
    settled and the agent can start. Apps that are ready immediately cost only
    ``poll * stable_polls`` seconds rather than the whole budget.

    Returns the seconds actually waited.
    """
    start = time.time()

    # Text alone is not enough: a stalled progress bar ("downloading — 0%")
    # holds steady and reads as settled. A dataset fetch is still an in-flight
    # request though, so require the network to be quiet as well.
    inflight = {"n": 0}

    def _started(_):
        inflight["n"] += 1

    def _ended(_):
        inflight["n"] = max(0, inflight["n"] - 1)

    for event, handler in (("request", _started), ("requestfinished", _ended), ("requestfailed", _ended)):
        try:
            page.on(event, handler)
        except Exception:
            pass

    previous = None
    unchanged = 0
    try:
        while time.time() - start < budget:
            try:
                current = page.inner_text("body")
            except Exception:
                current = None
            quiet = inflight["n"] == 0
            if quiet and current is not None and current == previous:
                unchanged += 1
                if unchanged >= stable_polls:
                    break
            else:
                unchanged = 0
            previous = current
            time.sleep(poll)
    finally:
        for event, handler in (("request", _started), ("requestfinished", _ended), ("requestfailed", _ended)):
            try:
                page.remove_listener(event, handler)
            except Exception:
                pass
    return time.time() - start


def _settle_export_page(page) -> None:
    """Wait for an export endpoint to actually render before scraping it.

    The benchmark apps are SPAs: navigating to /finish or /export-clear returns
    an empty shell that only fills in once the client bundle hydrates. Reading
    the body straight after navigation is a race — fast deployments win it,
    slow ones return "" and the run becomes unscorable even though the agent
    did the work.

    These endpoints render their JSON into a <pre>, so wait for that; fall back
    to network idle for any endpoint that does not, and never let this step
    fail the capture.
    """
    try:
        page.wait_for_selector("pre", timeout=30_000)
        return
    except Exception:
        pass
    try:
        page.wait_for_load_state("networkidle", timeout=15_000)
    except Exception:
        pass


def _extract_page_axtree(page) -> str | None:
    try:
        snapshot = page.accessibility.snapshot(interesting_only=False)
    except Exception:
        return None
    try:
        return json.dumps(snapshot)
    except Exception:
        return str(snapshot)


def _build_query(task: dict[str, Any], prefix_prompt: str | None) -> str:
    base = task.get("prompt") or task.get("goal") or ""
    if prefix_prompt:
        return f"{prefix_prompt.strip()}\n\n{base}"
    return base


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _job_key(task: dict[str, Any], iteration: int) -> tuple[str, int]:
    return str(task["id"]), iteration + 1


def _completed_run_keys(config: GeminiRunConfig) -> set[tuple[str, int]]:
    results_dir = Path(config.results_dir).expanduser().resolve()
    if not results_dir.exists():
        return set()

    completed: set[tuple[str, int]] = set()
    for summary_path in results_dir.glob("*/summary_info.json"):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if summary.get("experiment_status") != "completed":
            continue
        if summary.get("model_name") not in {None, config.model}:
            continue

        task_id = summary.get("task_id")
        if not task_id and isinstance(summary.get("task_name"), str):
            task_id = summary["task_name"].split(".", 1)[-1]
        if not task_id:
            continue

        iteration = summary.get("iteration", 1)
        if not isinstance(iteration, int):
            continue

        completed.add((str(task_id), iteration))
    return completed


def _run_dir_name(task_name: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{timestamp}_{task_name}_{uuid.uuid4().hex}"


def run_single_task(task: dict[str, Any], config: GeminiRunConfig, iteration: int = 0) -> dict[str, Any]:
    run_started_at = datetime.now().isoformat()
    run_started_perf = time.perf_counter()
    results_dir = Path(config.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    task_name = f"eval.{task['id']}"
    exp_dir = results_dir / _run_dir_name(task_name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    summary_info_path = exp_dir / "summary_info.json"
    run_uuid = uuid.uuid4().hex
    initial_summary = {
        "task_name": task_name,
        "task_id": task.get("id"),
        "agent_type": "GeminiComputerUseAgent",
        "model_name": config.model,
        "max_steps": config.max_steps,
        "cache_key": f"{task_name}_GeminiComputerUseAgent_{config.model}_{config.max_steps}",
        "experiment_status": "started",
        "run_uuid": run_uuid,
        "iteration": iteration + 1,
        "started_at": run_started_at,
        "completed_at": None,
        "duration_seconds": None,
    }
    _write_json(summary_info_path, initial_summary)

    error: str | None = None
    stack_trace: str | None = None
    agent_result: dict[str, Any] = {
        "steps": 0,
        "action_history": [],
        "final_reasoning": None,
        "final_answer_json": None,
        "token_usage": {},
        "token_usage_by_call": [],
        "model_call_count": 0,
        "total_model_duration_seconds": 0.0,
    }
    post_run_page_url: str | None = None
    post_run_page_content: str | None = None
    post_run_page_html: str | None = None
    post_run_page_axtree: str | None = None
    post_run_page_error: str | None = None
    post_run_js_result: Any = None
    post_run_js_error: str | None = None

    task_url = task.get("website", {}).get("url") or config.url
    if not task_url:
        raise ValueError(f"Task {task_name} has no website URL. Provide --url override.")

    with PlaywrightComputer(
        headless=config.headless,
        viewport_width=config.viewport_width,
        viewport_height=config.viewport_height,
        initial_url=task_url,
        inject_proxy_select=config.inject_proxy_select,
        proxy_select_js=config.proxy_select_js,
        proxy_select_css=config.proxy_select_css,
        exp_dir=exp_dir,
    ) as computer:
        try:
            if config.initial_delay > 0:
                _wait_until_settled(computer.page, budget=config.initial_delay)
            agent = GeminiBrowserAgent(
                computer=computer,
                query=_build_query(task, config.prefix_prompt),
                model_name=config.model,
                max_steps=config.max_steps,
                system_prompt_append=config.system_prompt_append or "",
                verbose=config.verbose,
            )
            agent_result = agent.agent_loop()
            if agent_result.get("error"):
                error = agent_result["error"]

            if config.post_run_url:
                try:
                    target_url = config.post_run_url
                    if not urlparse(target_url).scheme:
                        current_url = computer.page.url if computer.page else ""
                        base_url = current_url or task_url
                        target_url = urljoin(base_url, target_url)
                    state = computer.navigate(target_url)
                    post_run_page_url = state.url
                    _settle_export_page(computer.page)
                    post_run_page_content = computer.page.inner_text("body")
                    post_run_page_html = computer.page.content()
                    post_run_page_axtree = _extract_page_axtree(computer.page)
                except Exception as exc:
                    post_run_page_error = f"{type(exc).__name__}: {exc}"

            if config.post_run_js_snippet:
                try:
                    post_run_js_result = computer.page.evaluate(config.post_run_js_snippet)
                except Exception as exc:
                    post_run_js_error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # pragma: no cover - runtime integration path
            error = f"{type(exc).__name__}: {exc}"
            stack_trace = traceback.format_exc()

    final_response = agent_result.get("final_reasoning") or ""
    if agent_result.get("final_answer_json") is not None:
        final_response = json.dumps({"answer": agent_result["final_answer_json"]}, ensure_ascii=True)

    agent_outputs = {
        "raw_agent_response": final_response,
        "agent_response": final_response,
        "primary_output": final_response,
        "action_history": agent_result.get("action_history", []),
    }
    _write_json(exp_dir / "agent_outputs.json", agent_outputs)
    (exp_dir / "agent_output.txt").write_text(final_response or "", encoding="utf-8")

    step_state_files = sorted(str(path.resolve()) for path in exp_dir.glob("step_*.pkl.gz"))
    screenshot_files = sorted(str(path.resolve()) for path in exp_dir.glob("screenshot_step_*.png"))
    truncated = bool(agent_result.get("steps", 0) >= config.max_steps)
    run_completed_at = datetime.now().isoformat()
    duration_seconds = time.perf_counter() - run_started_perf
    summary_payload = {
        **initial_summary,
        "experiment_status": "completed",
        "completed_at": run_completed_at,
        "duration_seconds": duration_seconds,
        "completed": error is None,
        "success": error is None,
        "error": bool(error),
        "err_msg": error,
        "stack_trace": stack_trace,
        "n_steps": agent_result.get("steps", 0),
        "terminated": error is None and not truncated,
        "truncated": truncated,
        "score": 0.0,
        "agent_response": final_response,
        "finish_state": {},
        "post_run_js_snippet_path": config.post_run_js_snippet_path,
        "post_run_js_result": post_run_js_result,
        "post_run_js_error": post_run_js_error,
        "post_run_page_url": post_run_page_url,
        "post_run_page_content": post_run_page_content,
        "post_run_page_html": post_run_page_html,
        "post_run_page_axtree": post_run_page_axtree,
        "post_run_page_error": post_run_page_error,
        "step_state_count": len(step_state_files),
        "screenshot_count": len(screenshot_files),
        "token_usage": agent_result.get("token_usage", {}),
        "token_usage_by_call": agent_result.get("token_usage_by_call", []),
        "model_call_count": agent_result.get("model_call_count", 0),
        "total_model_duration_seconds": agent_result.get("total_model_duration_seconds", 0.0),
    }
    _write_json(summary_info_path, summary_payload)

    return {
        "iteration": iteration + 1,
        "task_name": task_name,
        "task_id": task["id"],
        "status": "error" if error else ("truncated" if truncated else "completed"),
        "task_steps": agent_result.get("steps", 0),
        "started_at": run_started_at,
        "completed_at": run_completed_at,
        "duration_seconds": duration_seconds,
        "exp_dir": str(exp_dir.resolve()),
        "summary_info_path": str(summary_info_path.resolve()),
        "experiment_log_path": None,
        "agent_outputs_path": str((exp_dir / "agent_outputs.json").resolve()),
        "agent_output_text_path": str((exp_dir / "agent_output.txt").resolve()),
        "step_state_files": step_state_files,
        "step_state_count": len(step_state_files),
        "screenshot_files": screenshot_files,
        "screenshot_count": len(screenshot_files),
        "agent_response": final_response,
        "finish_page_content": None,
        "finish_page_html": None,
        "finish_page_axtree": None,
        "post_run_js_snippet_path": config.post_run_js_snippet_path,
        "post_run_js_result": post_run_js_result,
        "post_run_js_error": post_run_js_error,
        "post_run_page_url": post_run_page_url,
        "post_run_page_content": post_run_page_content,
        "post_run_page_html": post_run_page_html,
        "post_run_page_axtree": post_run_page_axtree,
        "post_run_page_error": post_run_page_error,
        "terminated": summary_payload["terminated"],
        "truncated": summary_payload["truncated"],
        "error": bool(error),
        "token_usage": agent_result.get("token_usage", {}),
        "model_call_count": agent_result.get("model_call_count", 0),
        "total_model_duration_seconds": agent_result.get("total_model_duration_seconds", 0.0),
    }


def save_run_manifest(
    config: GeminiRunConfig,
    task_names: list[str],
    manifest_entries: list[dict[str, Any]],
    skipped_existing_runs: list[dict[str, Any]] | None = None,
) -> Path:
    results_dir = Path(config.results_dir).expanduser().resolve()
    manifests_dir = results_dir / "run_manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    manifest_path = manifests_dir / f"run_{timestamp}.json"
    payload = {
        "created_at": datetime.now().isoformat(),
        "model": config.model,
        "headless": config.headless,
        "use_screenshot": config.use_screenshot,
        "concurrent": config.concurrent,
        "num_workers": config.num_workers,
        "max_steps": config.max_steps,
        "results_dir": str(results_dir),
        "task_file": str(Path(config.task_file).resolve()) if config.task_file else None,
        "url": config.url or None,
        "js_snippet_file": str(Path(config.js_snippet_file).resolve()) if config.js_snippet_file else None,
        "post_run_url": config.post_run_url or None,
        "tasks": task_names,
        "iterations": config.iterations,
        "continue_run": config.continue_run,
        "skipped_existing_runs": skipped_existing_runs or [],
        "runs": manifest_entries,
    }
    _write_json(manifest_path, payload)
    return manifest_path


def run_tasks(tasks: list[dict[str, Any]], config: GeminiRunConfig) -> tuple[list[dict[str, Any]], Path, int]:
    task_names = [f"eval.{task['id']}" for task in tasks]
    manifest_entries: list[dict[str, Any]] = []
    skipped_existing_runs: list[dict[str, Any]] = []

    jobs: list[tuple[dict[str, Any], int]] = []
    completed_run_keys = _completed_run_keys(config) if config.continue_run else set()
    for iteration in range(config.iterations):
        for task in tasks:
            if _job_key(task, iteration) in completed_run_keys:
                skipped_existing_runs.append(
                    {
                        "iteration": iteration + 1,
                        "task_name": f"eval.{task['id']}",
                        "task_id": task["id"],
                        "status": "skipped_existing",
                    }
                )
                continue
            jobs.append((task, iteration))

    if skipped_existing_runs:
        print(f"Continuing run: skipping {len(skipped_existing_runs)} existing completed run(s).")
    if not jobs:
        print("Continuing run: no remaining task runs to start.")

    if config.concurrent:
        workers = config.workers if config.workers > 0 else min(6, len(tasks))
        config.num_workers = workers
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_single_task, task, config, iteration) for task, iteration in jobs]
            for future in as_completed(futures):
                manifest_entries.append(future.result())
    else:
        config.num_workers = 1
        for task, iteration in jobs:
            manifest_entries.append(run_single_task(task, config, iteration))

    manifest_entries.sort(key=lambda entry: (entry["iteration"], entry["task_name"]))
    manifest_path = save_run_manifest(config, task_names, manifest_entries, skipped_existing_runs)
    return manifest_entries, manifest_path, len(skipped_existing_runs)
