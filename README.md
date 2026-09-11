<p align="center">
  <img src="docs/logo.png" alt="GauntletBench logo" width="96" height="96" />
</p>

# GauntletBench - Agent Evaluation Framework

This repository hosts the evaluation harness for **GauntletBench**, the benchmark introduced in
*"Running the Gauntlet: Re-evaluating the Capabilities of Agents Beyond Familiar Environments"* (2026).

GauntletBench is a web-based benchmark for measuring how well agentic systems **generalise** to complex,
visually grounded professional tasks. To avoid data contamination from popular apps, it is built around five
less-covered but realistic applications and stresses three underexplored capabilities:

- **Capabilities:** dynamic/temporal perception, graphical understanding, 3D reasoning
- **Applications:** Circuit Designer, Flight Analyser, Video Editor, 3D Modeller, Workflow Builder
- **Tasks:** 135 challenging vision-intensive tasks (27 per app), each feasible for non-expert humans

The benchmark pairs a modular pipeline — an environment compatible with open- and closed-source agent
frameworks, controlled web apps, an extensible task suite, and an automated evaluation engine — with
domain-specific scoring.

**Key findings.** Across 135 human-feasible tasks, the strongest evaluated agent reaches only **28.2% success**,
while non-expert human annotators reach **80.1%** — current agents remain far from reliable performance in
complex real-world environments. Agentic frameworks beat raw models, and video editing is the most tractable
domain while circuit design is near-impossible.

> **Acknowledgement.** Built on top of the **REAL** paper repository
> ([AGI SDK](https://github.com/agi-inc/agisdk) — *paper:* [arxiv.org/abs/2504.11543](https://arxiv.org/abs/2504.11543)).
> The agent run mechanics (browser harness, task loop, action/observation interface) come directly from REAL,
> and the upstream README is preserved as [`REAL-README.md`](./REAL-README.md).
> The evaluation framework, batch runner, objective and LLM-as-a-judge evaluators, and the new task suites
> in this repo are my own contribution on top of those mechanics.

## Setup

Requires Python 3.11+.

```bash
poetry install
poetry run playwright install

# API keys for the model providers you want to use (main.py loads .env automatically)
echo "OPENAI_API_KEY=your_key_here" > .env
echo "OPENROUTER_API_KEY=your_key_here" >> .env   # optional
```

Or with plain pip — note that the harness must be installed from *this* repo, since a
PyPI package of the same name would shadow it:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
playwright install chromium
```

Supported model prefixes are `gpt-*`/`o1*`/`o3*` (OpenAI), `claude-*`/`sonnet-*`
(Anthropic), `openrouter/*`, `litellm/*`, `bedrock/*` (AWS Bedrock runtime, bearer-token
auth) and `gemini/*` (Google's native API). Each reads its own key from the environment.

### Tasks and ground truth

The task suites and their ground truth are published separately as the GauntletBench
dataset, [gauntlet-benchmark/tasks](https://github.com/gauntlet-benchmark/tasks)
(CC BY 4.0), and are not included in this repo. Copy them into `tasks/` and `assets/`
at the repo root. The dataset uses longer task file names than the harness, so copy
each suite under the short name that the runners, batch configs and evaluators expect:

```bash
git clone --depth 1 https://github.com/gauntlet-benchmark/tasks.git ../gauntlet-tasks
mkdir -p tasks
cp ../gauntlet-tasks/tasks/3d_modeller.yaml      tasks/3d.yaml
cp ../gauntlet-tasks/tasks/circuit_designer.yaml tasks/circuit.yaml
cp ../gauntlet-tasks/tasks/flight_analyser.yaml  tasks/flightradar.yaml
cp ../gauntlet-tasks/tasks/video_editor.yaml     tasks/video.yaml
cp ../gauntlet-tasks/tasks/workflow_builder.yaml tasks/graph.yaml
cp -R ../gauntlet-tasks/assets/. assets/   # 3d, graph and video ground truth
```

Both locations are gitignored, so the copied files are not committed back to this repo.
`assets/` tracks only `circuit_txt_export_snippet.js`, the Circuit Designer export script
used after each run.

## Usage

```bash
# Run one task (comma-separate ids to run several)
poetry run python main.py --task-file tasks/flightradar.yaml --task tc_frad_001 \
  --url https://gauntletbench-flight-analyser-app.hf.space/ \
  --prefix-prompt-file app_background/flightradar.md

# Run a whole suite headless, 4 tasks at a time
poetry run python main.py --task-file tasks/graph.yaml --run-all --headless \
  --concurrent --workers 4 \
  --url https://gauntletbench-graph.vercel.app/workflows --post-run-url /export \
  --prefix-prompt-file app_background/graph.md

# Run 5 random tasks, 3 iterations each
poetry run python main.py --task-file tasks/video.yaml --run-random -n 5 --iterations 3 \
  --url https://voidcut.vercel.app/ --post-run-url /finish

# Run a batch of experiments from JSON
poetry run python run_experiments.py --config run_configs/benchmark.json

# Print the commands a batch would run, without starting anything
poetry run python run_experiments.py --config run_configs/benchmark.json --dry-run

# Skip experiments whose results already exist
poetry run python run_experiments.py --config run_configs/benchmark.json --skip-existing-results
```

### Key Options
- `--task-file`: YAML/JSON task file (e.g. `tasks/circuit.yaml`), or a directory of them. Always pass it: the built-in default directory is not shipped.
- `--url`: Start URL of the app (or set `WEBCLONE_URL`). Required, because the task files do not carry one.
- `--model` / `-m`: Model to use (default: `o3`)
- `--task`: Task id, or a comma-separated list of ids
- `--application`: Filter by app when `--task-file` is a directory. The app id is the task file's name, e.g. `flightradar`.
- `--run-all`, `--run-random`, `-n` / `--max-tasks`: Task selection
- `--task-range`: Slice of tasks to run, e.g. `2:5`, `3:`, `:5`
- `--iterations`: Number of times to run each task
- `--concurrent` / `--workers`: Run tasks in parallel (up to 6 workers by default)
- `--headless`: Run browser in headless mode
- `--max-steps`: Agent step budget per task (default: `50`)
- `--post-run-url` / `--js-snippet-file`: How the final app state is captured (see the table below)
- `--prefix-prompt-file` / `--system-prompt-file`: Prepend app background to the task / append to the system prompt (inline `--prefix-prompt` / `--system-prompt` also work)
- `--initial-delay`: Seconds to wait after page load before the first action
- `--seed`: Fixed seed for LLM calls
- `--reasoning` / `--no-reasoning`, `--reasoning-effort`, `--thinking-type`: Reasoning controls
- `--no-use-screenshot`: Run without screenshots
- `--results-dir`: Where to save task artifacts (default: `results`)

### Benchmark Applications

Each task file is a YAML list of tasks for one app. All apps are hosted:

| App | Task file | URL (`--url`) | Final-state capture |
|-----|-----------|---------------|---------------------|
| Circuit Designer | `tasks/circuit.yaml` | `https://gauntletbench-circuit.up.railway.app/circuitjs.html?startCircuit=blank.txt` | `--js-snippet-file assets/circuit_txt_export_snippet.js` |
| Flight Analyser | `tasks/flightradar.yaml` | `https://gauntletbench-flight-analyser-app.hf.space/` | none (the answer is in the agent's response) |
| Video Editor | `tasks/video.yaml` | `https://voidcut.vercel.app/` | `--post-run-url /finish` |
| 3D Modeller | `tasks/3d.yaml` | `https://3d-clone-rouge.vercel.app/` | `--post-run-url /export-clear` |
| Workflow Builder | `tasks/graph.yaml` | `https://gauntletbench-graph.vercel.app/workflows` | `--post-run-url /export` |

The objective evaluators score that final-state capture, so include the capture flag for
each app. [`app_background/<app>.md`](app_background/) holds a short description of each
app, passed with `--prefix-prompt-file`. [`run_configs/benchmark.json`](run_configs/benchmark.json)
is the reference batch config, with these settings for all five apps.

### Run artifacts

Each task run gets its own directory under `--results-dir`:

| File | Contents |
|------|----------|
| `summary_info.json` | Run metadata, the agent's final response, the post-run page capture, aggregate step stats |
| `agent_outputs.json` | Per-step actions and model responses, plus the post-run capture the evaluators read |
| `agent_output.txt` | Just the primary output |
| `screenshot_step_*.png` | Per-step screenshots (Stage 1 of the LLM judge reads these) |
| `step_*.pkl.gz` | Per-step observation/state pickles |
| `experiment.log` | Run log |

The harness does **not** score runs. The apps carry no in-environment success criteria, so
every run finishes with reward 0 and is scored afterwards by `evaluation/`.

### Batch Config

`run_experiments.py` launches many runs at once from a JSON config. Each experiment becomes one
`main.py` command (or one run of the config's `entrypoint` script, which is how the Gemini and
CUA runners below are driven). It writes each experiment's combined stdout/stderr to
`<results_dir>/batch_runner.log`, and on shutdown terminates every running experiment's process
group so browser children do not linger. Relative paths in a config resolve against the repo root.

Pass `--skip-existing-results` to skip experiments whose `results_dir` already contains prior run artifacts (`run_manifests/run_*.json` or task `summary_info.json`). The same flag is also available as `"skip_existing_results": true` at the root, per-model, or per-testcase. Precedence: CLI flag → testcase → model → root. Other flags: `--max-parallel` (overrides the config's `max_parallel`), `--fail-fast`, `--dry-run` and `--use-screenshot` / `--no-use-screenshot`.

```json
{
  "max_parallel": 5,
  "defaults": {
    "run_all": true,
    "headless": true,
    "max_steps": 100
  },
  "models": {
    "openrouter/google/gemini-3.1-pro-preview": {
      "skip_existing_results": true,
      "testcases": [
        {
          "name": "graph",
          "task_file": "tasks/graph.yaml",
          "prefix_prompt_file": "app_background/graph.md",
          "url": "https://gauntletbench-graph.vercel.app/workflows",
          "post_run_url": "/export",
          "results_dir": "results/gemini31pro/graph"
        }
      ]
    }
  }
}
```

Each model key can define shared `defaults` plus a `testcases` list, and each testcase can override any supported `main.py` option (`results_dir`, `max_steps`, `post_run_url`, `post_run_js_snippet_path` — or the legacy `js_snippet_file` alias — `system_prompt`, `prefix_prompt_file`, `extra_args`, …).

## Evaluation

Two complementary methods are available once you have run experiments. Both need the
task files and ground truth from [Tasks and ground truth](#tasks-and-ground-truth).

### Objective Evaluation

Automated pass/fail scoring against ground-truth answers. Each app has a dedicated evaluator:

- **Circuit Designer** — compares truth tables of the GT and predicted circuits (requires `networkx`)
- **Flight Analyser** — extracts JSON from the agent response and compares it field-by-field
- **Video Editor** — validates timeline exports block-by-block against per-scenario rules
- **3D Modeller** — compares scene exports against ground-truth scenes with numeric tolerances and rotation symmetry handling
- **Workflow Builder** — matches nodes by content (greedy cost-minimising pairing) and verifies edges via the resulting ID mapping

```bash
# Evaluate every results directory under a root
poetry run python -m evaluation.objective.batch_evaluate results --workers 8

# Several roots, with the aggregated scores saved to a file
poetry run python -m evaluation.objective.batch_evaluate \
  results/open_source results/closed_source --workers 8 --output objective_summary.json
```

Three directory layouts are auto-discovered:

| Layout | Example |
|--------|---------|
| `{model}/{results_dir}`, app taken from a name suffix (`_circuit`, `_frad`, `_video` / `_voidcut`, `_3d` / `_clone3d`, `_graph`) | `results/open_source/opus46/results_opus46_graph` |
| `{app}/{model}` when the root is named after an app (`circuit`, `flightradar` / `frad`, `video` / `voidcut`, `3d`, `graph`) | `results/graph/opus46` |
| `run*/{app}/{model}` | `seed_runs/run1/graph/opus46` |

Each results directory gets an `objective_evaluation.json` file with `{test_id: 0|1}` scores.
Pass `--tasks-dir` if your task files are not in `tasks/`.

The evaluators exit non-zero whenever *any* task fails, so the batch runner cannot use the return code to detect a crashed evaluator. It instead requires the objective file to have been freshly rewritten — a stale file from an earlier run is reported as a failure rather than silently passed off as a new result.

You can also run individual evaluators directly:

```bash
# Circuit Designer
poetry run python -m evaluation.objective.evaluate_circuit_scheme \
  --tasks tasks/circuit.yaml --responses results/results_gpt54_circuit

# Flight Analyser (the task file argument defaults to tasks/flightradar.yaml)
poetry run python -m evaluation.objective.eval_flightradar \
  results/results_gpt54_frad tasks/flightradar.yaml --verbose

# Video Editor
poetry run python -m evaluation.objective.eval_voidcut \
  results/results_gpt54_video assets/video_ground_truth --tolerance_ms 1000 --verbose

# 3D Modeller
poetry run python -m evaluation.objective.eval_3d_editor \
  results/results_opus46_3d assets/3d_ground_truth --tolerance 0.15 --verbose

# Workflow Builder
poetry run python -m evaluation.objective.eval_graph \
  results/results_opus46_graph assets/graph_ground_truth --verbose
```

### LLM-as-a-Judge Evaluation

A 3-stage pipeline that uses LLMs to assess agent behaviour from screenshots and action traces:

1. **Stage 1** — Screenshot diff filtering. Consecutive screenshots are compared with ImageMagick to flag visually significant changes via normalised RMSE and changed-pixel fraction.
2. **Stage 2** — Per-change judgment. Flagged screenshot pairs are sent to an LLM (default `gpt-4o-mini`) which classifies the visible change, task relevance, and progress.
3. **Stage 3** — Final outcome. The full trajectory and the final screenshot are sent to an LLM (default `gpt-5.1`) which produces a 1–5 score with reasoning.

Requirements:
- `OPENAI_API_KEY` (plus `ANTHROPIC_API_KEY` if the Stage 3 model is a Claude model)
- Working `compare` and `magick` (ImageMagick 7) on `$PATH`
- An `objective_evaluation.json` for each results directory. Run the objective evaluation first: the judge refuses to run without it, since its Stage 3 rubric uses the objective result as a prior.

```bash
# Basic batch run
poetry run python -m evaluation.llm_judge.batch_run_llm_as_judge results/

# Parallel with task YAML overrides
poetry run python -m evaluation.llm_judge.batch_run_llm_as_judge results/ \
  --parallel 4 \
  --task-yaml circuit=tasks/circuit.yaml \
  --task-yaml frad=tasks/flightradar.yaml

# Generate summary CSVs
poetry run python -m evaluation.llm_judge.batch_run_llm_as_judge results/ \
  --summary-csv llm_summary.csv --runs-csv llm_runs.csv

# Skip already-evaluated directories
poetry run python -m evaluation.llm_judge.batch_run_llm_as_judge results/ \
  --skip-existing --match "closed_source"

# Custom models and reasoning
poetry run python -m evaluation.llm_judge.batch_run_llm_as_judge results/ \
  --stage2-model gpt-4o-mini --stage3-model gpt-5.1 \
  --stage3-reasoning-effort high

# Use one objective results file for every results directory
poetry run python -m evaluation.llm_judge.batch_run_llm_as_judge results/ \
  --objective-evaluation-json results/results_gpt54_circuit/objective_evaluation.json

# Dry run (no API calls) — writes dryrun_-prefixed files, never touching real results
poetry run python -m evaluation.llm_judge.batch_run_llm_as_judge results/ --dry-run
```

Each result directory gets an `llm_judgments.json`. The optional summary CSVs aggregate scores across models and apps.

Key options:
- `--parallel N` — number of concurrent worker threads
- `--rmse-threshold`, `--changed-fraction-threshold` — Stage 1 sensitivity knobs
- `--stage2-image-detail` / `--stage3-image-detail` — image detail level (`low` / `auto` / `high`)
- `--max-judgments` — cap on Stage 2 API calls per directory
- `--base-url` — custom OpenAI-compatible API endpoint

## Other Agent Runners

Besides the REAL harness in `main.py`, the repo has three runners for computer-use agents. The
Gemini and OpenAI CUA runners write the same run artifacts as `main.py`, so the evaluators work
on their results unchanged.

### OpenAI CUA (`openai-cua-sample-app/` + `cua_runner/`)

[`openai-cua-sample-app/`](openai-cua-sample-app/) is a vendored copy of OpenAI's
[CUA sample app](https://github.com/openai/openai-cua-sample-app) at upstream commit
[`f2a3dc5`](https://github.com/openai/openai-cua-sample-app/commit/f2a3dc5). Only its
JavaScript/Playwright agent is used. It carries two local changes:

- `javascript-app/src/bench-index.ts` — an additive benchmark entrypoint that plugs into the app's
  `RunnerManager({ executorFactory })` + `createServer({ manager })` extension seam, so a run
  targets a benchmark app instead of a bundled lab. The app's own agent loop is unmodified.
- `javascript-app/src/browser/javascript-process.ts` — Chromium launches with SwiftShader flags
  so WebGL apps (the 3D Modeller) render in headless mode instead of showing a blank viewport.

[`cua_runner/`](cua_runner/) drives it: it starts one app server per application, submits each
task as a run, and converts the results into the harness's artifact layout.

Needs Node.js 22.20.0 (Corepack provides the pinned pnpm 10.26.0) and `OPENAI_API_KEY`:

```bash
cd openai-cua-sample-app
corepack enable
pnpm install --frozen-lockfile
pnpm playwright:install
cd ..

# All five applications in parallel
poetry run python run_experiments.py --config cua_runner/run_configs/benchmark.cua.json
```

The app's contract caps a run at 50 model turns (`max_steps`). See
[`cua_runner/README.md`](cua_runner/README.md) for single-app runs and other details.

### Gemini Computer Use (`gemini_runner/`)

[`gemini_runner/`](gemini_runner/) runs tasks with Google's
[Gemini Computer Use](https://ai.google.dev/gemini-api/docs/computer-use) model. It needs Google's
`google-genai` SDK, which is not in the project's dependency list
(`poetry run pip install google-genai`). Set `GEMINI_API_KEY` (or use Vertex AI credentials), then:

```bash
# One application
poetry run python gemini_runner/run_gemini.py --task_file tasks/flightradar.yaml --run_all --headless \
  --url https://gauntletbench-flight-analyser-app.hf.space/ --results_dir results/gemini_flight

# All five applications in parallel
poetry run python run_experiments.py --config gemini_runner/run_configs/benchmark.gemini.json
```

See [`gemini_runner/README.md`](gemini_runner/README.md) for the other options.

### Claude Computer Use (`computer_use/`)

[`computer_use/`](computer_use/) is a self-contained Poetry project with its own dependencies.
It drives a Playwright browser via Anthropic's
[Computer Use API](https://docs.anthropic.com/en/docs/agents-and-tools/computer-use), with
provider adapters for Anthropic, Bedrock, OpenAI, and LiteLLM. Its `run-tasks` command runs a
task suite from `tasks/`:

```bash
cd computer_use
poetry install
poetry run run-tasks --tasks ../tasks/graph.yaml \
  --start-url https://gauntletbench-graph.vercel.app/workflows --post-run-url /export \
  --results-dir ../results/computer_use/graph
```

See [`computer_use/README.md`](computer_use/README.md) for setup and usage.

## Adding New Tasks

Create a YAML file in `tasks/` (or add a task to an existing one):

```yaml
tasks:
- id: "tc_myapp_001"
  difficulty_level: easy        # optional
  prompt: |-
    # Task Title

    ## GOAL
    Description of what the agent must accomplish.

    ## STEPS
    1. Step one
    2. Step two

    # RESULT FORMAT

    ```json
    {"answer": "<value>"}
    ```
  gt: {answer: '{"answer": "expected_value"}'}
  website: {url: "https://my-app.example.com/"}   # optional; otherwise pass --url
```

Then reference the file with `--task-file tasks/myapp.yaml` when running `main.py`, or add it to a batch config under `task_file`.
`tasks/` is gitignored here, so share new suites through the
[dataset repo](https://github.com/gauntlet-benchmark/tasks).

## License

This project is dual-licensed (see [`LICENSE`](./LICENSE)):

- **Upstream REAL / AGI SDK code** — Apache License 2.0 ([`LICENSE-APACHE`](./LICENSE-APACHE)), © 2025 AGI, Inc.
- **New contributions in this fork** — CC BY 4.0 ([`LICENSE-CC-BY-4.0`](./LICENSE-CC-BY-4.0)), © 2026 Gauntlet Bench.

The vendored [`openai-cua-sample-app/`](openai-cua-sample-app/) keeps its upstream MIT License
([`openai-cua-sample-app/LICENSE`](openai-cua-sample-app/LICENSE)), © 2025 OpenAI.
