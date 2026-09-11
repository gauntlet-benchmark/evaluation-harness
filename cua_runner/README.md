# CUA runner

Runs gauntlet-bench tasks through the [OpenAI CUA sample app](https://github.com/openai/openai-cua-sample-app)'s
JavaScript/Playwright agent, and writes results in the same layout as the REAL
and Gemini runners (`summary_info.json`, `agent_outputs.json`,
`screenshot_step_N.png` per run directory).

The sample app's own loop is **not modified**. The integration is one additive
entrypoint, `bench-index.ts`, which uses the app's documented extension seam
(`RunnerManager({ executorFactory })` + `createServer({ manager })`) to point a
run at an external benchmark application instead of a bundled lab workspace.

## Setup

The sample app is vendored in `openai-cua-sample-app/` (upstream commit
[`f2a3dc5`](https://github.com/openai/openai-cua-sample-app/commit/f2a3dc5)).
Install it once:

```bash
cd openai-cua-sample-app
corepack enable
pnpm install --frozen-lockfile
pnpm playwright:install
```

Local changes to the vendored copy:

- `javascript-app/src/bench-index.ts` — the benchmark entrypoint (additive; it
  lives in the app's `src/` to resolve the app's relative imports).
- `javascript-app/src/browser/javascript-process.ts` — launches Chromium with
  SwiftShader flags so headless WebGL apps (e.g. the 3d SuperSplat app) render
  instead of showing a blank viewport.

Set `OPENAI_API_KEY` (e.g. `source set_env.sh`) before running.

## Usage

All five applications, 27 tasks each, in parallel:

```bash
python run_experiments.py --config cua_runner/run_configs/benchmark.cua.json
```

One application directly:

```bash
python cua_runner/run_cua.py \
    --model gpt-6-astra \
    --run-all --continue-run \
    --task-file tasks/video.yaml \
    --url https://voidcut.vercel.app/ \
    --post-run-url /finish \
    --prefix-prompt-file app_background/video.md \
    --results-dir seed_runs/run1/video/gpt6astra \
    --initial-delay 120
```

`--task` accepts a comma-separated list (`--task tc_vid_003,tc_vid_009`) for
resuming an arbitrary set, and `--continue-run` skips tasks that already have a
completed, error-free run in `--results-dir`.

## Notes

- **One run at a time per server.** The sample app's `RunnerManager` admits a
  single active run, so tasks within an application are sequential; run the
  applications in parallel instead (each gets its own server on its own port).
- **`max_steps` is capped at 50** by the app's shared contract
  (`responseTurnBudgetSchema`). One "step" here is one model turn, which
  typically executes a whole block of Playwright actions, so this is a larger
  budget than 50 single actions in the other runners.
- **No seed control.** The Responses API loop takes no seed, so repeated runs
  are independent samples rather than reproducible replays.
- The agent receives no automatic observation: it calls `display(...)` for
  screenshots and reads the DOM through locators when it chooses to. The
  `screenshot_step_N.png` artifacts are captured by the harness for replay and
  judging, and are not what the model saw.
