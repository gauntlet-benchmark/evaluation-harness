"""Run gauntlet-bench tasks through the OpenAI CUA sample app's JS agent.

The sample app's core loop is untouched: this driver starts the app's runner
server through ``javascript-app/src/bench-index.ts`` (an additive entrypoint
that injects a benchmark executor via the app's own extension seam), submits
each task as a normal ``POST /api/runs`` request, and converts the finished
run's artifacts into the same per-experiment directory layout the REAL and
Gemini runners write under ``seed_runs/`` (``summary_info.json``,
``agent_outputs.json``, ``screenshot_step_N.png``).

The CLI mirrors ``gemini_runner/run_gemini.py`` so ``run_experiments.py`` can
drive it from a run config via the ``entrypoint`` key.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

import fire

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cua_runner.task_selection import (  # type: ignore[no-redef]
        apply_task_range,
        filter_tasks_by_application,
        filter_tasks_by_id,
        list_tasks,
    )
else:
    from .task_selection import (
        apply_task_range,
        filter_tasks_by_application,
        filter_tasks_by_id,
        list_tasks,
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
CUA_APP_DIR = REPO_ROOT / "openai-cua-sample-app"
TASKS_PATH = REPO_ROOT / "tasks"

# The sample app is vendored from upstream. The entrypoint sits inside the app's
# own src/ because it imports the app's modules relatively.
BENCH_ENTRYPOINT = CUA_APP_DIR / "javascript-app" / "src" / "bench-index.ts"

# Any catalog scenario satisfies the server's start-run validation; the bench
# executor ignores the scenario entirely (its lab template is copied into the
# run workspace and left unused). Kanban has the smallest template.
SCENARIO_ID = "kanban-reprioritize-sprint"

# The shared API contract caps maxResponseTurns at 50. One turn is one model
# response, which usually executes a whole exec_js block of many browser
# actions, so 50 turns is a materially larger budget than 50 single-action
# steps in the REAL/Gemini runners.
MAX_RESPONSE_TURNS_CAP = 50

AGENT_TYPE = "OpenAICUAAgent"


def _http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urlrequest.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class BenchServerError(RuntimeError):
    pass


class BenchServer:
    """One bench-index.ts instance: one benchmark app, one run at a time."""

    def __init__(
        self,
        url: str,
        post_run_url: str | None,
        post_run_js_file: str | None,
        initial_delay: float,
    ) -> None:
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.data_root = Path(tempfile.mkdtemp(prefix="cua-bench-"))
        self.log_path = self.data_root / "server.log"
        env = {
            **os.environ,
            "HOST": "127.0.0.1",
            "PORT": str(self.port),
            "BENCH_DATA_ROOT": str(self.data_root),
            "BENCH_URL": url,
            "BENCH_POST_RUN_URL": post_run_url or "",
            "BENCH_POST_RUN_JS_FILE": post_run_js_file or "",
            "BENCH_INITIAL_DELAY": str(initial_delay),
        }
        self._log_handle = self.log_path.open("wb")
        self.process = subprocess.Popen(
            ["node", "--import", "tsx", str(BENCH_ENTRYPOINT)],
            cwd=CUA_APP_DIR,
            env=env,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
        )

    def wait_ready(self, timeout: float = 120.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise BenchServerError(
                    f"Bench server exited with code {self.process.returncode}. Log tail:\n{self._log_tail()}"
                )
            try:
                health = _http_json("GET", f"{self.base_url}/health", timeout=5)
                if health.get("status") == "ok":
                    return
            except (urlerror.URLError, OSError, TimeoutError):
                pass
            time.sleep(1.0)
        raise BenchServerError(f"Bench server did not become healthy in {timeout:.0f}s. Log tail:\n{self._log_tail()}")

    def _log_tail(self, limit: int = 4000) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")[-limit:]
        except OSError:
            return "<no log>"

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=15)
        self._log_handle.close()

    def cleanup(self) -> None:
        shutil.rmtree(self.data_root, ignore_errors=True)


def check_sample_app() -> None:
    """Fail fast if the vendored sample app has not been installed yet."""
    if not BENCH_ENTRYPOINT.is_file():
        raise SystemExit(f"Missing bench entrypoint: {BENCH_ENTRYPOINT}")
    if not (CUA_APP_DIR / "node_modules").is_dir():
        raise SystemExit(
            f"{CUA_APP_DIR} is not installed. Run 'pnpm install --frozen-lockfile' "
            "and 'pnpm playwright:install' there first."
        )


def _read_optional_file(inline: str, file_path: str, label: str) -> str:
    if inline and file_path:
        raise ValueError(f"Provide either {label} or {label}_file, not both.")
    if file_path:
        return Path(file_path).expanduser().read_text(encoding="utf-8")
    return inline


def _build_prompt(task: dict[str, Any], prefix_prompt: str | None) -> str:
    base = task.get("prompt") or task.get("goal") or ""
    if prefix_prompt:
        return f"{prefix_prompt.strip()}\n\n{base}"
    return base


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_dir_name(task_name: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{timestamp}_{task_name}_{uuid.uuid4().hex}"


def _completed_run_keys(results_dir: Path, model: str) -> set[tuple[str, int]]:
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
        if summary.get("err_msg"):
            continue
        if summary.get("model_name") not in {None, model}:
            continue
        task_id = summary.get("task_id")
        if not task_id and isinstance(summary.get("task_name"), str):
            task_id = summary["task_name"].split(".", 1)[-1]
        iteration = summary.get("iteration", 1)
        if task_id and isinstance(iteration, int):
            completed.add((str(task_id), iteration))
    return completed


def _aggregate_token_usage(token_usage_by_call: list[dict[str, Any]]) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
    for call in token_usage_by_call:
        usage = call.get("usage") or {}
        totals["input_tokens"] += int(usage.get("input_tokens") or 0)
        totals["output_tokens"] += int(usage.get("output_tokens") or 0)
        totals["total_tokens"] += int(usage.get("total_tokens") or 0)
        details = usage.get("output_tokens_details") or {}
        totals["reasoning_tokens"] += int(details.get("reasoning_tokens") or 0)
    return totals


def _action_history_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") != "function_call_requested":
            continue
        detail = event.get("detail") or ""
        name, _, raw_args = detail.partition(" ")
        entry: dict[str, Any] = {"type": name or "function_call"}
        try:
            args = json.loads(raw_args) if raw_args else {}
            if isinstance(args, dict) and isinstance(args.get("code"), str):
                entry["code"] = args["code"]
        except json.JSONDecodeError:
            entry["raw_arguments"] = raw_args
        history.append(entry)
    return history


def _copy_screenshots(replay: dict[str, Any], exp_dir: Path) -> int:
    screenshots = ((replay.get("browser") or {}).get("screenshots")) or []
    count = 0
    for index, shot in enumerate(screenshots):
        source = Path(shot.get("path", ""))
        if not source.exists():
            continue
        shutil.copyfile(source, exp_dir / f"screenshot_step_{index}.png")
        count += 1
    return count


def _poll_run(base_url: str, run_id: str, max_task_seconds: float) -> dict[str, Any]:
    deadline = time.time() + max_task_seconds
    while True:
        detail = _http_json("GET", f"{base_url}/api/runs/{run_id}", timeout=30)
        status = (detail.get("run") or {}).get("status")
        if status in {"completed", "failed", "cancelled"}:
            return detail
        if time.time() > deadline:
            try:
                _http_json("POST", f"{base_url}/api/runs/{run_id}/stop", timeout=60)
            except (urlerror.URLError, OSError, TimeoutError):
                pass
            detail = _http_json("GET", f"{base_url}/api/runs/{run_id}", timeout=30)
            detail["_bench_timeout"] = True
            return detail
        time.sleep(3.0)


def run_single_task(
    task: dict[str, Any],
    server: BenchServer,
    *,
    model: str,
    results_dir: Path,
    prefix_prompt: str | None,
    headless: bool,
    max_steps: int,
    max_task_seconds: float,
    post_run_js_snippet_path: str | None,
    iteration: int = 0,
) -> dict[str, Any]:
    run_started_at = datetime.now().isoformat()
    run_started_perf = time.perf_counter()
    results_dir.mkdir(parents=True, exist_ok=True)
    task_name = f"eval.{task['id']}"
    exp_dir = results_dir / _run_dir_name(task_name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    summary_info_path = exp_dir / "summary_info.json"
    max_turns = min(max_steps, MAX_RESPONSE_TURNS_CAP)
    initial_summary = {
        "task_name": task_name,
        "task_id": task.get("id"),
        "agent_type": AGENT_TYPE,
        "model_name": model,
        "max_steps": max_turns,
        "cache_key": f"{task_name}_{AGENT_TYPE}_{model}_{max_turns}",
        "experiment_status": "started",
        "run_uuid": uuid.uuid4().hex,
        "iteration": iteration + 1,
        "started_at": run_started_at,
        "completed_at": None,
        "duration_seconds": None,
    }
    _write_json(summary_info_path, initial_summary)

    error: str | None = None
    timed_out = False
    bench_result: dict[str, Any] = {}
    replay: dict[str, Any] = {}
    run_id: str | None = None

    try:
        start_response = _http_json(
            "POST",
            f"{server.base_url}/api/runs",
            {
                "scenarioId": SCENARIO_ID,
                "browserMode": "headless" if headless else "headful",
                "maxResponseTurns": max_turns,
                "prompt": _build_prompt(task, prefix_prompt),
                "model": model,
            },
            timeout=120,
        )
        run_id = start_response["runId"]
        detail = _poll_run(server.base_url, run_id, max_task_seconds)
        timed_out = bool(detail.get("_bench_timeout"))
        run_record = detail.get("run") or {}

        bench_result_path = server.data_root / "workspaces" / run_id / "bench_result.json"
        if bench_result_path.exists():
            bench_result = json.loads(bench_result_path.read_text(encoding="utf-8"))
        replay_path = server.data_root / "runs" / run_id / "replay.json"
        if replay_path.exists():
            replay = json.loads(replay_path.read_text(encoding="utf-8"))

        if timed_out:
            error = f"TimeoutError: run exceeded max_task_seconds={max_task_seconds:.0f}"
        elif run_record.get("status") == "failed" and not bench_result:
            notes = (run_record.get("summary") or {}).get("notes") or []
            error = f"RunFailed: {'; '.join(notes) or 'unknown failure'}"
        elif bench_result.get("loop_error"):
            error = str(bench_result["loop_error"])
    except (urlerror.URLError, OSError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        error = f"{type(exc).__name__}: {exc}"

    final_response = bench_result.get("final_message") or ""
    events = replay.get("events") or []
    action_history = _action_history_from_events(events)
    agent_outputs = {
        "raw_agent_response": final_response,
        "agent_response": final_response,
        "primary_output": final_response,
        "action_history": action_history,
    }
    _write_json(exp_dir / "agent_outputs.json", agent_outputs)
    (exp_dir / "agent_output.txt").write_text(final_response or "", encoding="utf-8")

    screenshot_count = _copy_screenshots(replay, exp_dir)
    token_usage_by_call = bench_result.get("token_usage_by_call") or []
    n_steps = int(bench_result.get("agent_exec_js_calls") or 0)
    loop_error = bench_result.get("loop_error") or ""
    truncated = timed_out or "turn budget" in loop_error or "exhausted the configured" in loop_error
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
        "stack_trace": None,
        "n_steps": n_steps,
        "terminated": error is None and not truncated,
        "truncated": truncated,
        "score": 0.0,
        "agent_response": final_response,
        "finish_state": {},
        "post_run_js_snippet_path": post_run_js_snippet_path,
        "post_run_js_result": bench_result.get("post_run_js_result"),
        "post_run_js_error": bench_result.get("post_run_js_error"),
        "post_run_page_url": bench_result.get("post_run_page_url"),
        "post_run_page_content": bench_result.get("post_run_page_content"),
        "post_run_page_html": bench_result.get("post_run_page_html"),
        "post_run_page_axtree": bench_result.get("post_run_page_axtree"),
        "post_run_page_error": bench_result.get("post_run_page_error"),
        "step_state_count": 0,
        "screenshot_count": screenshot_count,
        "token_usage": _aggregate_token_usage(token_usage_by_call),
        "token_usage_by_call": token_usage_by_call,
        "model_call_count": int(bench_result.get("model_call_count") or 0),
        "total_model_duration_seconds": sum(
            float(call.get("duration_seconds") or 0) for call in token_usage_by_call
        ),
        "cua_run_id": run_id,
        "settle_seconds": bench_result.get("settle_seconds"),
    }
    _write_json(summary_info_path, summary_payload)

    return {
        "iteration": iteration + 1,
        "task_name": task_name,
        "task_id": task["id"],
        "status": "error" if error else ("truncated" if truncated else "completed"),
        "task_steps": n_steps,
        "duration_seconds": duration_seconds,
        "exp_dir": str(exp_dir.resolve()),
        "err_msg": error,
    }


def run(
    model: str = "gpt-6-astra",
    task: str = "",
    task_file: str = "",
    application: str = "",
    run_all: bool = False,
    headless: bool = True,
    max_steps: int = 50,
    results_dir: str = "results",
    url: str = "",
    post_run_url: str = "",
    js_snippet_file: str = "",
    prefix_prompt: str = "",
    prefix_prompt_file: str = "",
    initial_delay: float = 0.0,
    task_range: str = "",
    iterations: int = 1,
    continue_run: bool = False,
    max_task_seconds: float = 3600.0,
    keep_server_data: bool = False,
    # Accepted for run-config compatibility with the other runners; the CUA
    # loop has no equivalents (screenshots are model-driven, no seed control,
    # and the server serializes runs).
    use_screenshot: bool = True,
    seed: int | None = None,
    verbose: bool = False,
    concurrent: bool = False,
    workers: int = 0,
) -> None:
    del use_screenshot, seed, verbose, workers
    if concurrent:
        print("Note: the CUA runner server executes one run at a time; running sequentially.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (source set_env.sh first).")
    check_sample_app()
    if not url:
        raise SystemExit("--url is required: the CUA runner opens one benchmark app per invocation.")
    if max_steps > MAX_RESPONSE_TURNS_CAP:
        print(f"Note: capping max_steps={max_steps} to {MAX_RESPONSE_TURNS_CAP} (Responses-turn contract limit).")

    task_source = Path(task_file).expanduser().resolve() if task_file else TASKS_PATH
    tasks = list_tasks(task_source)
    if application:
        tasks = filter_tasks_by_application(tasks, application)
    if task:
        tasks = filter_tasks_by_id(tasks, task)
    if task_range:
        tasks = apply_task_range(tasks, task_range)
    if not tasks:
        raise SystemExit("No tasks selected.")
    if not (run_all or task or task_range):
        tasks = tasks[:1]

    prefix_prompt_text = _read_optional_file(prefix_prompt, prefix_prompt_file, "prefix_prompt") or None
    js_snippet_path = str(Path(js_snippet_file).expanduser().resolve()) if js_snippet_file else None
    resolved_results_dir = Path(results_dir).expanduser().resolve()

    jobs: list[tuple[dict[str, Any], int]] = []
    skipped = 0
    completed_keys = _completed_run_keys(resolved_results_dir, model) if continue_run else set()
    for iteration in range(iterations):
        for candidate in tasks:
            if (str(candidate["id"]), iteration + 1) in completed_keys:
                skipped += 1
                continue
            jobs.append((candidate, iteration))
    if skipped:
        print(f"Continuing run: skipping {skipped} existing completed run(s).")
    if not jobs:
        print("Nothing to run.")
        return

    server = BenchServer(
        url=url,
        post_run_url=post_run_url or None,
        post_run_js_file=js_snippet_path,
        initial_delay=initial_delay,
    )
    manifest_entries: list[dict[str, Any]] = []
    try:
        server.wait_ready()
        print(f"Bench server ready at {server.base_url} for {url} ({len(jobs)} job(s)).")
        for candidate, iteration in jobs:
            entry = run_single_task(
                candidate,
                server,
                model=model,
                results_dir=resolved_results_dir,
                prefix_prompt=prefix_prompt_text,
                headless=headless,
                max_steps=max_steps,
                max_task_seconds=max_task_seconds,
                post_run_js_snippet_path=js_snippet_path,
                iteration=iteration,
            )
            manifest_entries.append(entry)
            print(
                f"[{entry['status']:>9}] {entry['task_name']} iter {entry['iteration']} "
                f"steps={entry['task_steps']} {entry['duration_seconds']:.0f}s"
                + (f" err={entry['err_msg']}" if entry["err_msg"] else "")
            )
    finally:
        server.stop()
        if not keep_server_data:
            server.cleanup()
        else:
            print(f"Server data kept at {server.data_root}")

    manifests_dir = resolved_results_dir / "run_manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    _write_json(
        manifest_path,
        {
            "created_at": datetime.now().isoformat(),
            "agent_type": AGENT_TYPE,
            "model": model,
            "url": url,
            "headless": headless,
            "max_steps": min(max_steps, MAX_RESPONSE_TURNS_CAP),
            "results_dir": str(resolved_results_dir),
            "task_file": str(task_source),
            "post_run_url": post_run_url or None,
            "post_run_js_snippet_path": js_snippet_path,
            "iterations": iterations,
            "continue_run": continue_run,
            "skipped_existing_runs": skipped,
            "runs": manifest_entries,
        },
    )
    errored = sum(1 for entry in manifest_entries if entry["status"] == "error")
    print(f"Done: {len(manifest_entries)} run(s), {errored} error(s). Manifest: {manifest_path}")


if __name__ == "__main__":
    fire.Fire(run)
