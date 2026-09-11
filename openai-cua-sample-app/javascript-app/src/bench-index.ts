// Gauntlet-bench entrypoint: the same runner server and Responses code loop,
// pointed at an external benchmark app instead of a bundled lab workspace.
//
// This file is additive — it uses the documented extension seam
// (RunnerManager({ executorFactory }) + createServer({ manager })) and changes
// nothing in the sample app's own modules, so upstream pulls stay clean.
//
// The stock index.ts is not reused because it acquires the port-4050 backend
// lease, which limits the machine to one runner instance; the benchmark runs
// one instance per application in parallel, so this entrypoint skips the lease.
//
// Per-instance configuration comes from the environment (one server serves one
// benchmark app, so nothing here varies per task except the prompt):
//   PORT / HOST              where to listen (default 127.0.0.1:4101)
//   BENCH_URL                the benchmark application to open (required)
//   BENCH_DATA_ROOT          where run artifacts are written (required)
//   BENCH_POST_RUN_URL       optional export endpoint to scrape after the run
//   BENCH_POST_RUN_JS_FILE   optional JS file to page.evaluate after the run
//   BENCH_INITIAL_DELAY      max seconds to wait for the app to settle

import { readFileSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { type RunDetail } from "@cua-sample/contracts";

import { launchJavaScriptSession, type JavaScriptSession } from "./browser/javascript-process.js";
import { createDefaultResponsesClient, runResponsesCodeLoop } from "./responses-loop.js";
import { RunnerManager } from "./runner-manager.js";
import { type RunExecutionContext, type RunExecutor } from "./scenario-runtime.js";
import { createServer } from "./server.js";

type BenchEnv = {
  url: string;
  postRunUrl: string | undefined;
  postRunJs: string | undefined;
  postRunJsFile: string | undefined;
  initialDelaySeconds: number;
};

function readBenchEnv(): BenchEnv {
  const url = process.env.BENCH_URL?.trim();
  if (!url) throw new Error("BENCH_URL is required.");
  const postRunJsFile = process.env.BENCH_POST_RUN_JS_FILE?.trim() || undefined;
  return {
    url,
    postRunUrl: process.env.BENCH_POST_RUN_URL?.trim() || undefined,
    postRunJs: postRunJsFile ? readFileSync(postRunJsFile, "utf8") : undefined,
    postRunJsFile,
    initialDelaySeconds: Number(process.env.BENCH_INITIAL_DELAY ?? 0) || 0,
  };
}

function joinTextOutputs(outputs: Array<{ type: string; text?: string }>): string {
  return outputs
    .filter((item) => item.type === "input_text" && typeof item.text === "string")
    .map((item) => item.text as string)
    .join("");
}

// The worker appends "exec_js completed with no console output." when nothing
// was logged, and swallows thrown errors into a text item. Marker-framed JSON
// keeps our own reads unambiguous either way.
const resultMarker = "@@BENCH_RESULT@@";

async function execJson(
  session: JavaScriptSession,
  context: RunExecutionContext,
  body: string,
): Promise<unknown> {
  const code = [
    "const __benchRun = async () => {",
    body,
    "};",
    `console.log(${JSON.stringify(resultMarker)} + JSON.stringify({ value: await __benchRun() }) + ${JSON.stringify(resultMarker)});`,
  ].join("\n");
  const raw = joinTextOutputs(await session.execute(code, context.signal));
  const start = raw.indexOf(resultMarker);
  const end = raw.lastIndexOf(resultMarker);
  if (start === -1 || end <= start) {
    throw new Error(`Bench exec produced no result: ${raw.slice(0, 500)}`);
  }
  const parsed = JSON.parse(raw.slice(start + resultMarker.length, end)) as { value: unknown };
  return parsed.value;
}

async function settlePage(
  session: JavaScriptSession,
  context: RunExecutionContext,
  budgetSeconds: number,
): Promise<number> {
  // Port of the Python runner's _wait_until_settled: treat the delay as a
  // maximum and start as soon as the page holds still. In-flight requests are
  // not observable from outside the worker, so "still" here is body text plus
  // the finished-resource count holding across three polls.
  const start = Date.now();
  let previous = "";
  let unchanged = 0;
  while ((Date.now() - start) / 1000 < budgetSeconds) {
    let current = "";
    try {
      current = String(await execJson(session, context, `
        await new Promise((resolve) => setTimeout(resolve, 2000));
        const text = await page.evaluate(() => (document.body ? document.body.innerText : ""));
        const resources = await page.evaluate(() => performance.getEntriesByType("resource").length);
        return text.length + ":" + text.slice(0, 4000) + ":" + resources;
      `));
    } catch {
      current = "";
    }
    if (current && current === previous) {
      unchanged += 1;
      if (unchanged >= 3) break;
    } else {
      unchanged = 0;
    }
    previous = current;
  }
  return (Date.now() - start) / 1000;
}

type PostRunCapture = {
  post_run_page_url: string | null;
  post_run_page_content: string | null;
  post_run_page_html: string | null;
  post_run_page_axtree: string | null;
  post_run_page_error: string | null;
  post_run_js_result: unknown;
  post_run_js_error: string | null;
};

async function capturePostRun(
  session: JavaScriptSession,
  context: RunExecutionContext,
  bench: BenchEnv,
): Promise<PostRunCapture> {
  const capture: PostRunCapture = {
    post_run_page_url: null,
    post_run_page_content: null,
    post_run_page_html: null,
    post_run_page_axtree: null,
    post_run_page_error: null,
    post_run_js_result: null,
    post_run_js_error: null,
  };

  if (bench.postRunUrl) {
    try {
      const page = (await execJson(session, context, `
        // The worker's vm context has no URL global; resolve inside the page.
        const target = await page.evaluate(
          (raw) => new URL(raw, location.href).toString(),
          ${JSON.stringify(bench.postRunUrl)},
        );
        context.setDefaultNavigationTimeout(60000);
        await page.goto(target, { waitUntil: "domcontentloaded" });
        // Export pages are SPA shells that render their payload into a <pre>
        // only after hydration; reading the body immediately is a race.
        try { await page.waitForSelector("pre", { timeout: 30000 }); } catch {}
        try { await page.waitForLoadState("networkidle", { timeout: 15000 }); } catch {}
        return {
          url: page.url(),
          text: await page.evaluate(() => (document.body ? document.body.innerText : "")),
          html: await page.content(),
          // page.accessibility is deprecated and absent from some Playwright
          // versions; the axtree is best-effort, never fail the capture on it.
          axtree: await (async () => {
            try { return await page.accessibility?.snapshot({ interestingOnly: false }) ?? null; }
            catch { return null; }
          })(),
        };
      `)) as { url: string; text: string; html: string; axtree: unknown };
      capture.post_run_page_url = page.url;
      capture.post_run_page_content = page.text;
      capture.post_run_page_html = page.html;
      capture.post_run_page_axtree = page.axtree === null ? null : JSON.stringify(page.axtree);
    } catch (error) {
      capture.post_run_page_error = error instanceof Error ? error.message : String(error);
    }
  }

  if (bench.postRunJs) {
    try {
      capture.post_run_js_result = await execJson(session, context, `
        return await page.evaluate(${JSON.stringify(bench.postRunJs)});
      `);
    } catch (error) {
      capture.post_run_js_error = error instanceof Error ? error.message : String(error);
    }
  }

  return capture;
}

function createBenchExecutor(bench: BenchEnv) {
  return (detail: RunDetail): RunExecutor => ({
    async execute(context) {
      const startedAt = new Date().toISOString();
      const tokenUsageByCall: Array<Record<string, unknown>> = [];
      const baseClient = createDefaultResponsesClient();
      const client = {
        create: async (request: Record<string, unknown>, signal: AbortSignal) => {
          const callStart = Date.now();
          const response = await baseClient.create(request, signal);
          tokenUsageByCall.push({
            response_id: response.id,
            duration_seconds: (Date.now() - callStart) / 1000,
            usage: response.usage ?? null,
          });
          return response;
        },
      };

      let execJsCalls = 0;
      let session: JavaScriptSession | undefined;
      try {
        // Open about:blank first: the session's initial goto runs on stock
        // Playwright timeouts, which cold-starting benchmark deployments
        // (hf.space spin-up, dataset fetches) routinely exceed. Navigation
        // happens below under our own generous timeout instead.
        session = await launchJavaScriptSession({
          browserMode: detail.run.browserMode,
          screenshotDir: context.screenshotDirectory,
          signal: context.signal,
          targetLabel: "gauntlet-bench app",
          url: "about:blank",
          workerPath: fileURLToPath(new URL(
            import.meta.url.endsWith(".ts") ? "./javascript-worker.ts" : "./javascript-worker.js",
            import.meta.url,
          )),
        });
        const rawExecute = session.execute.bind(session);
        const countingSession: JavaScriptSession = {
          ...session,
          execute: (code, signal) => {
            execJsCalls += 1;
            return rawExecute(code, signal);
          },
        };

        await execJson(session, context, `
          context.setDefaultNavigationTimeout(60000);
          await page.goto(${JSON.stringify(bench.url)}, { waitUntil: "domcontentloaded" });
          return page.url();
        `);
        await context.emitEvent({
          detail: bench.url,
          level: "ok",
          message: "Browser navigated to the benchmark application.",
          type: "browser_navigated",
        });

        let settleSeconds = 0;
        if (bench.initialDelaySeconds > 0) {
          settleSeconds = await settlePage(session, context, bench.initialDelaySeconds);
          await context.emitEvent({
            detail: `${settleSeconds.toFixed(1)}s`,
            level: "ok",
            message: "Benchmark application settled.",
            type: "run_progress",
          });
        }

        await context.syncBrowserState(session);
        await context.captureScreenshot(session, "bench-loaded");

        const instructions = [
          "You are operating a persistent Playwright browser session.",
          "You must use the exec_js tool before you answer.",
          `The application under test is already open at ${bench.url}.`,
          "Observe the interface with display((await page.screenshot()).toString('base64')), then use Playwright locators, page.mouse, and page.keyboard to operate the visible controls.",
          "Mouse coordinates refer to the page screenshot.",
          "Stay on the application under test; do not navigate to other sites.",
          "When the task is done, reply with a final answer in the exact RESULT FORMAT the task specifies.",
        ].join("\n");

        // A thrown loop (turn budget exhausted, API error, refusal) must not
        // skip the export capture: the app state the agent produced is still
        // scorable, exactly as in the Python runners.
        const setupExecCalls = execJsCalls;
        let loopError: string | null = null;
        let finalMessage: string | undefined;
        try {
          const result = await runResponsesCodeLoop({
            context,
            instructions,
            maxResponseTurns: detail.run.maxResponseTurns,
            prompt: detail.run.prompt.trim(),
            session: countingSession,
          }, client);
          finalMessage = result.finalAssistantMessage;
        } catch (error) {
          if (context.signal.aborted) throw error;
          loopError = error instanceof Error ? error.message : String(error);
        }
        const agentExecCalls = execJsCalls - setupExecCalls;

        const capture = await capturePostRun(session, context, bench);
        await context.captureScreenshot(session, "bench-final");

        await writeFile(
          join(detail.workspacePath, "bench_result.json"),
          JSON.stringify({
            bench_url: bench.url,
            initial_delay_budget_seconds: bench.initialDelaySeconds,
            settle_seconds: settleSeconds,
            started_at: startedAt,
            completed_at: new Date().toISOString(),
            model: detail.run.model,
            max_response_turns: detail.run.maxResponseTurns,
            model_call_count: tokenUsageByCall.length,
            agent_exec_js_calls: agentExecCalls,
            loop_error: loopError,
            final_message: finalMessage ?? null,
            token_usage_by_call: tokenUsageByCall,
            post_run_js_snippet_path: bench.postRunJsFile ?? null,
            ...capture,
          }, null, 2),
          "utf8",
        );

        await context.completeRun({
          notes: [
            loopError
              ? `Responses loop ended with error: ${loopError}`
              : `Model final response: ${finalMessage ?? ""}`,
            `exec_js calls: ${agentExecCalls}, model turns: ${tokenUsageByCall.length}`,
          ],
        });
      } finally {
        await session?.close();
      }
    },
  });
}

const bench = readBenchEnv();
const dataRoot = process.env.BENCH_DATA_ROOT?.trim();
if (!dataRoot) throw new Error("BENCH_DATA_ROOT is required.");
const port = Number(process.env.PORT ?? 4101);
const host = process.env.HOST ?? "127.0.0.1";

const manager = new RunnerManager({ dataRoot, executorFactory: createBenchExecutor(bench) });
const server = createServer({ dataRoot, manager });

let shuttingDown = false;
function beginShutdown(exitCode: number) {
  if (shuttingDown) return;
  shuttingDown = true;
  void server.close().then(
    () => process.exit(exitCode),
    (error: unknown) => {
      console.error("Bench runner shutdown failed:", error);
      process.exit(1);
    },
  );
}
process.on("SIGINT", () => beginShutdown(130));
process.on("SIGTERM", () => beginShutdown(143));

try {
  await server.listen({ port, host });
  console.log(`Bench runner for ${bench.url} listening on http://${host}:${port}`);
} catch (error) {
  console.error("Bench runner failed to start:", error);
  process.exit(1);
}
