"""Gemini Computer Use browser agent wrapper."""

from __future__ import annotations

import json
import os
import re
import time
from copy import deepcopy
from typing import Any, Literal

from .computer import EnvState
from .playwright_computer import PlaywrightComputer

MAX_RECENT_TURN_WITH_SCREENSHOTS = 3
PREDEFINED_COMPUTER_USE_FUNCTIONS = [
    "open_web_browser",
    "click_at",
    "hover_at",
    "type_text_at",
    "scroll_document",
    "scroll_at",
    "wait_5_seconds",
    "go_back",
    "go_forward",
    "search",
    "navigate",
    "key_combination",
    "drag_and_drop",
]
ANSWER_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class GeminiBrowserAgent:
    def __init__(
        self,
        *,
        computer: PlaywrightComputer,
        query: str,
        model_name: str,
        max_steps: int,
        system_prompt_append: str = "",
        auto_confirm_safety: bool = True,
        verbose: bool = True,
        thinking_budget: int = 8192,
    ) -> None:
        try:
            from google import genai as _genai
            from google.genai import types as _types
            from google.genai.types import Content as _Content
        except ImportError as exc:
            raise ImportError(
                "google-genai is required. Install dependencies (e.g. `poetry install`)."
            ) from exc

        self._types = _types
        self._genai = _genai
        self._Content = _Content
        self._computer = computer
        self._query = query
        self._model_name = model_name
        self._max_steps = max_steps
        self._auto_confirm_safety = auto_confirm_safety
        self._verbose = verbose
        self._step_count = 0
        # Without an explicit timeout a stalled response never returns: one run
        # sat on a single step for 35 hours, blocking every task queued behind
        # it. Cap the request so a hung call surfaces as an error the batch can
        # move past. GEMINI_TIMEOUT_MS overrides it.
        timeout_ms = int(os.environ.get("GEMINI_TIMEOUT_MS", 10 * 60 * 1000))
        self._client = self._genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
            vertexai=os.environ.get("USE_VERTEXAI", "0").lower() in {"true", "1"},
            project=os.environ.get("VERTEXAI_PROJECT"),
            location=os.environ.get("VERTEXAI_LOCATION"),
            http_options=self._types.HttpOptions(timeout=timeout_ms),
        )

        prompt = query.strip()
        if system_prompt_append:
            prompt = f"{system_prompt_append.strip()}\n\n{prompt}"

        self._contents: list[Any] = [self._Content(role="user", parts=[self._types.Part(text=prompt)])]
        self.final_reasoning: str | None = None
        self.final_answer_json: Any = None
        self.action_history: list[str] = []
        self.model_call_count = 0
        self.total_model_duration_seconds = 0.0
        self.token_usage: dict[str, int | float] = {}
        self.token_usage_by_call: list[dict[str, Any]] = []

        self._generate_content_config = self._types.GenerateContentConfig(
            max_output_tokens=thinking_budget,
            temperature=1,
            top_p=0.95,
            top_k=40,
            thinking_config=self._types.ThinkingConfig(include_thoughts=True),
            tools=[
                self._types.Tool(
                    computer_use=self._types.ComputerUse(
                        environment=self._types.Environment.ENVIRONMENT_BROWSER,
                        excluded_predefined_functions=[],
                    )
                )
            ],
        )

    def _extract_function_calls(self, candidate: Any) -> list[Any]:
        if not candidate.content or not candidate.content.parts:
            return []
        return [part.function_call for part in candidate.content.parts if part.function_call]

    def _extract_text(self, candidate: Any) -> str:
        if not candidate.content or not candidate.content.parts:
            return ""
        return " ".join(part.text for part in candidate.content.parts if part.text).strip()

    def _extract_answer_json(self, text: str) -> Any:
        if not text:
            return None
        matches = ANSWER_FENCE_RE.findall(text)
        if not matches:
            return None
        try:
            data = json.loads(matches[-1])
        except json.JSONDecodeError:
            return None
        if isinstance(data, dict) and "answer" in data:
            return data["answer"]
        return None

    def _model_generate(self) -> Any:
        return self._client.models.generate_content(
            model=self._model_name,
            contents=self._contents,
            config=self._generate_content_config,
        )

    def _to_jsonable(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): self._to_jsonable(val) for key, val in value.items() if val is not None}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(item) for item in value if item is not None]

        for method_name in ("to_json_dict", "model_dump", "to_dict"):
            method = getattr(value, method_name, None)
            if not callable(method):
                continue
            try:
                return self._to_jsonable(method())
            except Exception:
                continue

        if hasattr(value, "__dict__"):
            return {
                key: self._to_jsonable(val)
                for key, val in vars(value).items()
                if not key.startswith("_") and val is not None
            }
        return str(value)

    def _extract_usage_metadata(self, response: Any) -> dict[str, Any]:
        usage_metadata = getattr(response, "usage_metadata", None)
        usage = self._to_jsonable(usage_metadata)
        return usage if isinstance(usage, dict) else {}

    def _add_cumulative_usage(self, usage: dict[str, Any], prefix: str = "") -> None:
        for key, value in usage.items():
            usage_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                self.token_usage[usage_key] = self.token_usage.get(usage_key, 0) + value
            elif isinstance(value, dict):
                self._add_cumulative_usage(value, usage_key)

    def _record_model_call(self, response: Any, duration_seconds: float) -> dict[str, Any]:
        self.model_call_count += 1
        self.total_model_duration_seconds += duration_seconds
        usage_metadata = self._extract_usage_metadata(response)
        self._add_cumulative_usage(usage_metadata)
        call_record = {
            "call_index": self.model_call_count,
            "duration_seconds": duration_seconds,
            "usage_metadata": usage_metadata,
            "cumulative_token_usage": deepcopy(self.token_usage),
            "total_model_duration_seconds": self.total_model_duration_seconds,
        }
        self.token_usage_by_call.append(call_record)
        return call_record

    def _denormalize_x(self, x: int) -> int:
        return int(x / 1000 * self._computer.screen_size()[0])

    def _denormalize_y(self, y: int) -> int:
        return int(y / 1000 * self._computer.screen_size()[1])

    @staticmethod
    def _parse_key_combination(keys: Any) -> list[str]:
        if isinstance(keys, list):
            tokens = keys
        elif isinstance(keys, str):
            tokens = keys.split("+")
        else:
            tokens = []
        return [token.strip() for token in tokens if isinstance(token, str) and token.strip()]

    def _handle_action(self, action: Any) -> EnvState:
        if action.name == "open_web_browser":
            return self._computer.open_web_browser()
        if action.name == "click_at":
            return self._computer.click_at(self._denormalize_x(action.args["x"]), self._denormalize_y(action.args["y"]))
        if action.name == "hover_at":
            return self._computer.hover_at(self._denormalize_x(action.args["x"]), self._denormalize_y(action.args["y"]))
        if action.name == "type_text_at":
            return self._computer.type_text_at(
                self._denormalize_x(action.args["x"]),
                self._denormalize_y(action.args["y"]),
                action.args["text"],
                action.args.get("press_enter", False),
                action.args.get("clear_before_typing", True),
            )
        if action.name == "scroll_document":
            return self._computer.scroll_document(action.args["direction"])
        if action.name == "scroll_at":
            direction = action.args["direction"]
            magnitude = action.args.get("magnitude", 800)
            if direction in {"up", "down"}:
                magnitude = self._denormalize_y(magnitude)
            elif direction in {"left", "right"}:
                magnitude = self._denormalize_x(magnitude)
            return self._computer.scroll_at(
                self._denormalize_x(action.args["x"]),
                self._denormalize_y(action.args["y"]),
                direction,
                magnitude,
            )
        if action.name == "wait_5_seconds":
            return self._computer.wait_5_seconds()
        if action.name == "go_back":
            return self._computer.go_back()
        if action.name == "go_forward":
            return self._computer.go_forward()
        if action.name == "search":
            return self._computer.search()
        if action.name == "navigate":
            return self._computer.navigate(action.args["url"])
        if action.name == "key_combination":
            return self._computer.key_combination(self._parse_key_combination(action.args.get("keys")))
        if action.name == "drag_and_drop":
            return self._computer.drag_and_drop(
                self._denormalize_x(action.args["x"]),
                self._denormalize_y(action.args["y"]),
                self._denormalize_x(action.args["destination_x"]),
                self._denormalize_y(action.args["destination_y"]),
            )
        raise ValueError(f"Unsupported function: {action.name}")

    def _update_turn_screenshots(self) -> None:
        turns = 0
        for content in reversed(self._contents):
            if content.role != "user" or not content.parts:
                continue
            has_screenshot = any(
                part.function_response
                and part.function_response.parts
                and part.function_response.name in PREDEFINED_COMPUTER_USE_FUNCTIONS
                for part in content.parts
            )
            if not has_screenshot:
                continue
            turns += 1
            if turns > MAX_RECENT_TURN_WITH_SCREENSHOTS:
                for part in content.parts:
                    if part.function_response and part.function_response.name in PREDEFINED_COMPUTER_USE_FUNCTIONS:
                        part.function_response.parts = None

    def run_one_iteration(self) -> Literal["COMPLETE", "CONTINUE"]:
        model_call_started_at = time.perf_counter()
        response = self._model_generate()
        model_call_duration_seconds = time.perf_counter() - model_call_started_at
        model_call_stats = self._record_model_call(response, model_call_duration_seconds)
        if not response.candidates:
            raise ValueError("Gemini response has no candidates.")

        candidate = response.candidates[0]
        if candidate.content:
            self._contents.append(candidate.content)

        reasoning = self._extract_text(candidate)
        function_calls = self._extract_function_calls(candidate)
        if (
            not function_calls
            and not reasoning
            and candidate.finish_reason == self._types.FinishReason.MALFORMED_FUNCTION_CALL
        ):
            return "CONTINUE"

        if not function_calls:
            self.final_reasoning = reasoning
            self.final_answer_json = self._extract_answer_json(reasoning)
            return "COMPLETE"

        function_responses: list[Any] = []
        for function_call in function_calls:
            if self._step_count >= self._max_steps:
                return "COMPLETE"
            safety = function_call.args.get("safety_decision") if function_call.args else None
            if safety and not self._auto_confirm_safety:
                raise RuntimeError("Safety confirmation required but auto_confirm_safety is disabled.")

            action_string = f"{function_call.name}({json.dumps(function_call.args, ensure_ascii=True)})"
            last_action_error: str | None = None
            try:
                state = self._handle_action(function_call)
            except Exception as exc:
                # Keep the run alive on malformed tool calls and expose the failure in step artifacts.
                last_action_error = f"{type(exc).__name__}: {exc}"
                state = self._computer.current_state()
            self.action_history.append(action_string)
            self._computer.save_step_artifact(
                step_idx=self._step_count,
                action=action_string,
                state=state,
                raw_model_response=reasoning,
                token_usage={
                    "model_call": model_call_stats,
                    "cumulative_token_usage": deepcopy(self.token_usage),
                    "model_call_count": self.model_call_count,
                    "total_model_duration_seconds": self.total_model_duration_seconds,
                },
                last_action_error=last_action_error,
            )
            self._step_count += 1
            function_response_payload = {"url": state.url, "safety_acknowledgement": "true"}
            if last_action_error:
                function_response_payload["error"] = last_action_error
            function_responses.append(
                self._types.FunctionResponse(
                    name=function_call.name,
                    response=function_response_payload,
                    parts=[
                        self._types.FunctionResponsePart(
                            inline_data=self._types.FunctionResponseBlob(mime_type="image/png", data=state.screenshot)
                        )
                    ],
                )
            )
            if self._verbose:
                time.sleep(0.05)

        self._contents.append(
            self._Content(role="user", parts=[self._types.Part(function_response=fr) for fr in function_responses])
        )
        self._update_turn_screenshots()
        return "CONTINUE"

    def agent_loop(self) -> dict[str, Any]:
        status: Literal["COMPLETE", "CONTINUE"] = "CONTINUE"
        error: str | None = None
        try:
            while status == "CONTINUE":
                status = self.run_one_iteration()
        except Exception as exc:  # pragma: no cover - runtime integration path
            error = f"{type(exc).__name__}: {exc}"
        return {
            "error": error,
            "steps": self._step_count,
            "final_reasoning": self.final_reasoning,
            "final_answer_json": self.final_answer_json,
            "action_history": self.action_history,
            "token_usage": self.token_usage,
            "token_usage_by_call": self.token_usage_by_call,
            "model_call_count": self.model_call_count,
            "total_model_duration_seconds": self.total_model_duration_seconds,
        }
