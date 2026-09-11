# Gemini Runner

Runs GauntletBench tasks with Google's
[Gemini Computer Use](https://ai.google.dev/gemini-api/docs/computer-use) model, which operates
a browser from screenshots. The runner wraps a Playwright browser, sends each screenshot and
action through the Gemini API using Google's `google-genai` SDK, and writes the same run
artifacts as the main harness, so the objective evaluators and the LLM judge work on its results
unchanged.

## Setup

From the repository root:

```bash
poetry install
poetry run playwright install
poetry run pip install google-genai   # Google's Gemini SDK; not in the project's dependency list
```

Copy the task files into `tasks/` as described in the main README's
[Tasks and ground truth](../README.md#tasks-and-ground-truth) section.

Then set either a Gemini API key:

```bash
export GEMINI_API_KEY=your_key_here
```

or Vertex AI credentials: pass `--use_vertexai --vertexai_project <project> --vertexai_location <region>`,
or set `USE_VERTEXAI=true`, `VERTEXAI_PROJECT` and `VERTEXAI_LOCATION`.

## Single Runs

Use the Fire script from the repository root:

```bash
poetry run python gemini_runner/run_gemini.py --task_file tasks/flightradar.yaml --run_all \
  --url https://gauntletbench-flight-analyser-app.hf.space/ \
  --prefix_prompt_file app_background/flightradar.md \
  --results_dir results/gemini/flightradar
```

- `--task_file` is required: the runner's built-in default task directory is not shipped.
- `--url` is required: the task files carry no URL, so the runner stops without one. The app URLs
  and their final-state capture flags (`--post_run_url` / `--js_snippet_file`) are listed in the
  main README's [Benchmark Applications](../README.md#benchmark-applications) table.
- Without `--run_all`, `--task`, `--run_random` or `--max_tasks`, only the first task runs.
- The browser is headless by default; pass `--noheadless` to watch it.
- `--continue_run` skips tasks that already have a completed run in `--results_dir`.

Other options: `--model` (default `gemini-2.5-computer-use-preview-10-2025`), `--task`,
`--application`, `--task_range`, `--run_random` / `--max_tasks`, `--iterations`,
`--concurrent` / `--workers`, `--max_steps` (default 50), `--initial_delay`,
`--system_prompt` / `--system_prompt_file`, `--prefix_prompt`, and
`--viewport_width` / `--viewport_height` (default 1280×800). Fire accepts dashes as well as
underscores in flag names.

## Batch Runs

[`run_configs/benchmark.gemini.json`](run_configs/benchmark.gemini.json) runs
`gemini-3-flash-preview` on all five applications with the URLs, capture settings and background
prompts from the main README:

```bash
poetry run python gemini_runner/run_gemini_experiments.py --config gemini_runner/run_configs/benchmark.gemini.json
```

Configs use the same format as the main `run_experiments.py`: shared `defaults`, one or more model
blocks, and per-model `testcases`. Relative paths in a config resolve against the repository root.
Launcher flags: `--max_parallel`, `--fail_fast`, `--dry_run`, `--continue_run` and
`--skip_existing_results`.

Because the config names `gemini_runner/run_gemini.py` as its `entrypoint`, the main batch runner
can launch it too:

```bash
poetry run python run_experiments.py --config gemini_runner/run_configs/benchmark.gemini.json
```

## Outputs

Each task run gets a directory named `<timestamp>_<task>_<id>` under `results_dir`, containing
`summary_info.json`, `agent_outputs.json`, screenshots, and the post-run page or JavaScript capture.
Each invocation also writes a manifest to `run_manifests/run_<timestamp>.json`, and batch runs write
`batch_runner.log` inside each experiment's `results_dir`. Score the results with the evaluators
described in the main README's [Evaluation](../README.md#evaluation) section.
