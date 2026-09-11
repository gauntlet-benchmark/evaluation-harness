#!/usr/bin/env python3
"""Batch parallel objective evaluation across benchmark applications.

Discovers model result directories under a given root (e.g. clean_results/open_source)
and runs the appropriate evaluator for each benchmark app (circuit, flightradar, voidcut)
in parallel using a process pool.

Usage:
    python -m evaluation.objective.batch_evaluate clean_results/open_source
    python -m evaluation.objective.batch_evaluate clean_results/closed_source --workers 4
    python -m evaluation.objective.batch_evaluate clean_results/open_source clean_results/closed_source
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


TASKS_DIR = Path("tasks")  # overridden by --tasks-dir
GT_VOIDCUT = Path("assets/video_ground_truth")
GT_3D = Path("assets/3d_ground_truth")
GT_GRAPH = Path("assets/graph_ground_truth")

APP_CIRCUIT = "circuit"
APP_FRAD = "frad"
APP_VIDEO = "video"
APP_3D = "3d"
APP_GRAPH = "graph"

CIRCUIT_PATTERNS = ("_circuit",)
FRAD_PATTERNS = ("_frad",)
VIDEO_PATTERNS = ("_video", "_voidcut")
THREE_D_PATTERNS = ("_3d", "_clone3d")
GRAPH_PATTERNS = ("_graph",)

ROOT_APP_ALIASES = {
    "circuit": APP_CIRCUIT,
    "flightradar": APP_FRAD,
    "frad": APP_FRAD,
    "video": APP_VIDEO,
    "voidcut": APP_VIDEO,
    "3d": APP_3D,
    "graph": APP_GRAPH,
}


@dataclass
class EvalJob:
    app: str
    model: str
    results_dir: Path

    @property
    def label(self) -> str:
        return f"{self.model}/{self.app}"


@dataclass
class EvalOutcome:
    job: EvalJob
    success: bool
    objective_scores: dict[str, int | float] | None = None
    error: str | None = None


def classify_app(dirname: str) -> str | None:
    """Classify a results directory name into an app type."""
    lower = dirname.lower()
    for pattern in CIRCUIT_PATTERNS:
        if pattern in lower:
            return APP_CIRCUIT
    for pattern in FRAD_PATTERNS:
        if pattern in lower:
            return APP_FRAD
    for pattern in VIDEO_PATTERNS:
        if pattern in lower:
            return APP_VIDEO
    for pattern in GRAPH_PATTERNS:
        if pattern in lower:
            return APP_GRAPH
    for pattern in THREE_D_PATTERNS:
        if pattern in lower:
            return APP_3D
    return None


def _discover_seed_run_jobs(root: Path) -> list[EvalJob]:
    """Discover jobs in the seed-runs layout: [run*/]{app}/{model}/{tasks}.

    Accepts either a single run (``seed_runs/run1``) or the collection root
    (``seed_runs``).  Symlinked entries are skipped so the ``results_*`` shims
    do not yield each results dir a second time.
    """
    run_dirs = [p for p in sorted(root.glob("run*")) if p.is_dir() and not p.is_symlink()]
    bases = run_dirs or [root]

    jobs: list[EvalJob] = []
    for base in bases:
        for app_name, app_code in ROOT_APP_ALIASES.items():
            app_dir = base / app_name
            if not app_dir.is_dir() or app_dir.is_symlink():
                continue
            for model_dir in sorted(app_dir.iterdir()):
                if not model_dir.is_dir() or model_dir.is_symlink():
                    continue
                label = f"{base.name}/{model_dir.name}" if run_dirs else model_dir.name
                jobs.append(EvalJob(app=app_code, model=label, results_dir=model_dir))
    return jobs


def discover_jobs(root_dirs: list[Path]) -> list[EvalJob]:
    """Walk root directories and discover all evaluation jobs."""
    jobs: list[EvalJob] = []

    for root in root_dirs:
        if not root.is_dir():
            print(f"WARNING: skipping non-directory {root}", file=sys.stderr)
            continue

        app_from_root = ROOT_APP_ALIASES.get(root.name.lower())
        if app_from_root is not None:
            # {app}/{model}/{tasks} layout — model dir IS the results dir.
            for model_dir in sorted(root.iterdir()):
                if not model_dir.is_dir():
                    continue
                jobs.append(EvalJob(app=app_from_root, model=model_dir.name, results_dir=model_dir))
            continue

        # {run}/{app}/{model}/{tasks} layout, e.g. seed_runs/run1 or seed_runs itself.
        # results_{model}_{app} symlink shims alias the same dirs, so they are skipped.
        seed_run_jobs = _discover_seed_run_jobs(root)
        if seed_run_jobs:
            jobs.extend(seed_run_jobs)
            continue

        for model_dir in sorted(root.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name

            for results_dir in sorted(model_dir.iterdir()):
                if not results_dir.is_dir():
                    continue
                app = classify_app(results_dir.name)
                if app is None:
                    continue
                jobs.append(EvalJob(app=app, model=model_name, results_dir=results_dir))

    return jobs


def _build_command(job: EvalJob) -> list[str]:
    """Build the subprocess command for an evaluation job."""
    if job.app == APP_CIRCUIT:
        return [
            sys.executable, "-m", "evaluation.objective.evaluate_circuit_scheme",
            "--tasks", str(TASKS_DIR / "circuit.yaml"),
            "--responses", str(job.results_dir),
        ]

    if job.app == APP_FRAD:
        return [
            sys.executable, "-m", "evaluation.objective.eval_flightradar",
            str(job.results_dir),
            "--tasks-yaml", str(TASKS_DIR / "flightradar.yaml"),
        ]

    if job.app == APP_VIDEO:
        return [
            sys.executable, "-m", "evaluation.objective.eval_voidcut",
            str(job.results_dir),
            str(GT_VOIDCUT),
        ]

    if job.app == APP_3D:
        return [
            sys.executable, "-m", "evaluation.objective.eval_3d_editor",
            str(job.results_dir),
            str(GT_3D),
        ]

    if job.app == APP_GRAPH:
        return [
            sys.executable, "-m", "evaluation.objective.eval_graph",
            str(job.results_dir),
            str(GT_GRAPH),
        ]

    raise ValueError(f"Unknown app type: {job.app}")


def _find_objective_file(job: EvalJob) -> Path | None:
    """Locate the objective_evaluation.json produced by the evaluator."""
    candidate = job.results_dir / "objective_evaluation.json"
    if candidate.exists():
        return candidate
    return None


def run_job(job: EvalJob) -> EvalOutcome:
    """Execute a single evaluation job in a subprocess.

    The evaluators exit 1 whenever *any* task fails, so the return code cannot
    distinguish "the evaluator crashed" from "the model scored badly".  Instead
    the objective file must be freshly written: its mtime is recorded first and
    a result is only accepted if the evaluator rewrote it.  Otherwise a stale
    file from an earlier run would be silently reported as a fresh result.
    """
    cmd = _build_command(job)

    existing = _find_objective_file(job)
    mtime_before = existing.stat().st_mtime_ns if existing is not None else None

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return EvalOutcome(job=job, success=False, error="timeout after 300s")
    except Exception as exc:
        return EvalOutcome(job=job, success=False, error=str(exc))

    stdout_tail = result.stdout[-1000:] if result.stdout else ""
    stderr_tail = result.stderr[-1000:] if result.stderr else ""

    obj_path = _find_objective_file(job)
    if obj_path is None:
        return EvalOutcome(
            job=job, success=False,
            error=f"no objective_evaluation.json produced (exit={result.returncode})\n"
                  f"stdout: {stdout_tail}\nstderr: {stderr_tail}",
        )

    if mtime_before is not None and obj_path.stat().st_mtime_ns == mtime_before:
        return EvalOutcome(
            job=job, success=False,
            error=f"evaluator did not rewrite {obj_path.name}; refusing to report the stale file "
                  f"(exit={result.returncode})\nstdout: {stdout_tail}\nstderr: {stderr_tail}",
        )

    try:
        scores = json.loads(obj_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return EvalOutcome(
            job=job, success=False,
            error=f"failed to read {obj_path}: {exc}\nstdout: {stdout_tail}\nstderr: {stderr_tail}",
        )
    return EvalOutcome(job=job, success=True, objective_scores=scores)


def _aggregate_results(outcomes: list[EvalOutcome]) -> dict:
    """Build a summary dict from all outcomes."""
    per_model: dict[str, dict[str, dict]] = {}

    for outcome in outcomes:
        model = outcome.job.model
        app = outcome.job.app
        if model not in per_model:
            per_model[model] = {}

        if outcome.success and outcome.objective_scores is not None:
            scores = outcome.objective_scores
            total = len(scores)
            passed = sum(1 for v in scores.values() if v == 1)
            per_model[model][app] = {
                "passed": passed,
                "total": total,
                "accuracy": round(passed / total, 4) if total else 0.0,
                "scores": scores,
            }
        else:
            per_model[model][app] = {
                "passed": 0,
                "total": 0,
                "accuracy": 0.0,
                "error": outcome.error,
            }

    return per_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run batch parallel objective evaluation across benchmark applications.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "root_dirs", nargs="+", type=Path,
        help="Root directories containing model subdirectories (e.g. clean_results/open_source)",
    )
    parser.add_argument(
        "--tasks-dir", type=Path, default=None,
        help="Override the tasks directory (default: tasks/)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Max parallel workers (default: number of jobs)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save aggregated JSON results (default: print to stdout)",
    )
    args = parser.parse_args()

    if args.tasks_dir is not None:
        global TASKS_DIR
        TASKS_DIR = args.tasks_dir

    jobs = discover_jobs(args.root_dirs)
    if not jobs:
        print("No evaluation jobs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Discovered {len(jobs)} evaluation job(s):")
    for job in jobs:
        print(f"  {job.label:30s} {job.results_dir}")

    max_workers = args.workers or len(jobs)
    outcomes: list[EvalOutcome] = []

    print(f"\nRunning evaluations with {max_workers} workers...\n")

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_to_job = {pool.submit(run_job, job): job for job in jobs}
        for future in as_completed(future_to_job):
            outcome = future.result()
            outcomes.append(outcome)

            if outcome.success and outcome.objective_scores is not None:
                scores = outcome.objective_scores
                passed = sum(1 for v in scores.values() if v == 1)
                total = len(scores)
                print(f"  [DONE] {outcome.job.label:30s} {passed}/{total}")
            else:
                print(f"  [FAIL] {outcome.job.label:30s} {outcome.error}")

    aggregated = _aggregate_results(outcomes)

    print("\n" + "=" * 82)
    print(f"{'Model':<20s} {'Circuit':>12s} {'FlightRadar':>12s} {'VoidCut':>12s} {'3D Editor':>12s} {'Graph':>12s}")
    print("-" * 94)
    for model in sorted(aggregated):
        cells: list[str] = []
        for app in (APP_CIRCUIT, APP_FRAD, APP_VIDEO, APP_3D, APP_GRAPH):
            info = aggregated[model].get(app)
            if info is None:
                cells.append("—")
            elif "error" in info:
                cells.append("ERROR")
            else:
                cells.append(f"{info['passed']}/{info['total']}")
        print(f"{model:<20s} {cells[0]:>12s} {cells[1]:>12s} {cells[2]:>12s} {cells[3]:>12s} {cells[4]:>12s}")
    print("=" * 94)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")
        print(f"\nAggregated results saved to: {output_path}")
    else:
        print("\nAggregated JSON:")
        print(json.dumps(aggregated, indent=2))


if __name__ == "__main__":
    main()
