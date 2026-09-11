# Harness

## Overview

`harness.py` orchestrates agent runs against the browser tasks: it builds the agent,
configures the browser environment, executes tasks (sequentially or in parallel via Ray),
and writes each run's artifacts to disk.

It does **not** score anything. The benchmark apps carry no in-environment success
criteria, so every run ends with `reward = 0` and is scored afterwards by the evaluators
under `evaluation/` against the app state exported by the post-run capture.

## Architecture and Flow

### 1. Initialization (`__init__`)

**Agent configuration**
- If `model` is provided (e.g. `"gpt-5.4"`), builds a `DemoAgentArgs` for it.
- If `agentargs` is provided, uses that custom agent instead.
- Handles system-message placement (`"separate"` vs `"combined"`).

**Environment configuration**
- Browser settings (headless, viewport, extensions, user data dir).
- Task parameters (`max_steps`, which observation types to include).
- Creates the results directory.

### 2. Execution (`run` → `_run_tasks`)

`run(tasks=[...])` takes an explicit list of gym task ids (`eval.tc_circuit_001`, …),
falling back to the single `task_name` the harness was constructed with. Every task in
the list is executed; there is no result cache, so a re-run always re-runs.

With `num_workers == 1` tasks run sequentially in-process via `_run_single_task`.
With `num_workers > 1` each task is submitted to Ray as a `run_task_ray` future. Ray is
initialised with `resources={"memory_gb": num_workers}` and each task declares
`memory_gb: 3`, so that resource is what actually caps concurrency.

Ray workers are separate processes, so `run_task_ray` re-registers the task YAMLs from
`registration_paths` before running.

### 3. Single task execution

1. Register tasks from `registration_paths` if given.
2. Build `EnvArgs` / `ExpArgs` and call `exp_args.prepare(results_dir)`, which creates
   the run directory.
3. Write `summary_info.json` with the run metadata (`task_name`, `agent_type`,
   `model_name`, `max_steps`, `experiment_status: "started"`) **before** running, so a
   crashed run still identifies itself.
4. `exp_args.run()` drives the episode loop, which overwrites `summary_info.json` with
   the full record on completion and writes `agent_outputs.json`, `agent_output.txt`,
   per-step screenshots and (optionally) per-step pickles.
5. Load the record back via `get_exp_result` and return it with `elapsed_time` and
   `exp_dir` attached.

### 4. Run artifacts

Each run directory holds:

| File | Contents |
|------|----------|
| `summary_info.json` | Run metadata, the agent's final response, the post-run page capture, aggregate step stats |
| `agent_outputs.json` | Per-step actions and model responses, plus the post-run capture the evaluators read |
| `agent_output.txt` | Just the primary output, for eyeballing |
| `screenshot_step_*.png` | Per-step screenshots (Stage 1 of the LLM judge reads these) |
| `step_*.pkl.gz` | Per-step observation/state pickles, when `save_step_info` is on |
| `experiment.log` | Run log |

## Configuration Example

```python
from agisdk import REAL

harness = REAL.harness(
    model="gpt-5.4",
    headless=True,
    max_steps=40,
    num_workers=6,
    results_dir="results/results_gpt54_circuit",
    registration_paths=["tasks/circuit.yaml"],
    post_run_url="https://circuit.example.com/export",
    save_step_screenshots=True,
)

results = harness.run(tasks=["eval.tc_circuit_001"])
```

In practice the harness is driven through `main.py` (single model) or
`run_experiments.py` (batch of models from a JSON config) rather than directly.
