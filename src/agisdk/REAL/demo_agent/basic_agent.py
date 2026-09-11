import base64
import dataclasses
import json
import numpy as np
import io
import logging
import re
import time

from PIL import Image
from typing import Any, Literal, Optional

from agisdk.REAL.browsergym.experiments import Agent, AbstractAgentArgs
from agisdk.REAL.browsergym.core.action.highlevel import HighLevelActionSet
from agisdk.REAL.browsergym.core.action.python import PythonActionSet
from agisdk.REAL.browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
from ..logging import logger as rich_logger

logger = logging.getLogger(__name__)


_CODE_FENCE_RE = re.compile(r"```(?:json|python)?\s*\n?(.*?)\n?```", re.DOTALL)


def _unwrap_code_fence(text: str) -> str:
    """Return the content of the last fenced block if present, else the stripped text.

    The last fence is used because models typically narrate first, then emit the action.
    """
    stripped = text.strip()
    matches = _CODE_FENCE_RE.findall(stripped)
    if matches:
        return matches[-1].strip()
    return stripped


def _extract_action_from_json(text: str) -> Optional[str]:
    """If text is a JSON object with an 'action' field, return the decoded string.

    Returns None if the text isn't JSON-shaped or doesn't contain an action string.
    This also JSON-decodes escape sequences like \\n into real newlines, which the
    downstream high-level action parser requires.
    """
    stripped = text.strip()

    # A JSON array of call objects, e.g. [{"action": "click", "args": ["322"]}, ...].
    # Rendered as one action per line, which the multiaction parser accepts.
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            items = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(items, list):
            return None
        rendered = []
        for item in items:
            if not isinstance(item, dict):
                return None
            call = _render_structured_action(item)
            if call is None:
                return None
            rendered.append(call)
        return "\n".join(rendered) if rendered else None

    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(data, dict):
        value = data.get("action")
        # Only a genuine call string is passed straight through. A bare action
        # name (e.g. {"action": "click", "args": ["322"]}) must be rendered from
        # its arguments, otherwise it parses to nothing.
        if isinstance(value, str) and "(" in value:
            return value
        # A bare name is only an action when there is nothing else in the object
        # that could be an argument. {"action": "click", "bid": "34"} must go to
        # the renderer, or it returns "click" and the bid is lost.
        if isinstance(value, str) and len(data) == 1:
            return value
        return _render_structured_action(data)
    return None


# Some models (notably Gemini) emit the action as a structured function-call
# object instead of call syntax, e.g.
#   {"action_name": "click", "parameters": {"bid": "143"}}
# The high-level parser only understands positional call syntax, so such a
# response parses to nothing and the step is lost to "Received an empty action".
_STRUCTURED_NAME_KEYS = (
    "action_name",
    "action_type",
    "name",
    "function",
    "tool",
    "tool_name",
    "action",
)
_STRUCTURED_ARGS_KEYS = ("parameters", "params", "args", "arguments", "action_args", "input")


def _render_structured_action(data: dict) -> Optional[str]:
    """Render a structured function-call object as positional call syntax.

    Parameters are ordered against the real action signature, because
    `HighLevelActionSet.to_python_code` renders arguments positionally and
    cannot accept keyword syntax. Returns None if the object isn't a
    recognizable call or can't be mapped onto a known action.
    """
    name = None
    for key in _STRUCTURED_NAME_KEYS:
        candidate = data.get(key)
        # Skip call strings — those are handled by the caller, not rendered here.
        if isinstance(candidate, str) and candidate.strip() and "(" not in candidate:
            name = candidate.strip()
            break
    if name is None:
        return None

    # A nested call object, e.g. {"function": {"name": ..., "arguments": {...}}}
    if isinstance(data.get(name), dict):
        data = data[name]

    raw_args: Any = {}
    for key in _STRUCTURED_ARGS_KEYS:
        if key in data:
            raw_args = data[key]
            break
    else:
        # No args container: models often inline the arguments alongside the
        # action name, e.g. {"action": "click", "bid": "34"}. Without this the
        # object renders as a bare "click()" — the argument is dropped silently,
        # the call fails, and the step is lost even though the model was right.
        raw_args = {k: v for k, v in data.items() if k not in _STRUCTURED_NAME_KEYS}

    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except (json.JSONDecodeError, ValueError):
            return None

    try:
        from agisdk.REAL.browsergym.core.action import functions as action_functions
        import inspect

        func = getattr(action_functions, name, None)
        if func is None:
            return None
        parameters = list(inspect.signature(func).parameters.values())
    except Exception:
        return None

    if isinstance(raw_args, dict):
        # Drop keys the action doesn't accept, then fill positionally up to the
        # last supplied parameter, using declared defaults for any gaps.
        param_names = [p.name for p in parameters]
        supplied = {k: v for k, v in raw_args.items() if k in param_names}
        if not supplied and raw_args:
            return None
        last = max((param_names.index(k) for k in supplied), default=-1)
        values = []
        for param in parameters[: last + 1]:
            if param.name in supplied:
                values.append(supplied[param.name])
            elif param.default is not inspect.Parameter.empty:
                values.append(param.default)
            else:
                return None
    elif isinstance(raw_args, (list, tuple)):
        values = list(raw_args)
    else:
        return None

    return f"{name}({', '.join(repr(value) for value in values)})"


_RESULT_FORMAT_RE = re.compile(r"#\s*RESULT\s+FORMAT\s*(.*)", re.S | re.I)


def _declared_result_keys(goal: str) -> frozenset:
    """The output keys a task's own RESULT FORMAT block declares.

    Every task prompt ends with a ``# RESULT FORMAT`` section holding the exact
    JSON shape it wants back — ``{"answer": "done"}`` for most, task-specific
    keys such as ``{"geo_altitude": ..., "vertical_rate": ...}`` for flightradar.
    Returns an empty set when no such block is present, which keeps the caller
    on the strict ``answer``-only rule.
    """
    if not isinstance(goal, str):
        return frozenset()
    section = _RESULT_FORMAT_RE.search(goal)
    if not section:
        return frozenset()
    block = _unwrap_code_fence(section.group(1)).strip()
    if not (block.startswith("{") and block.endswith("}")):
        return frozenset()
    try:
        shape = json.loads(block)
    except (json.JSONDecodeError, ValueError):
        return frozenset()
    return frozenset(shape) if isinstance(shape, dict) else frozenset()


def _extract_answer_from_text(text: str, result_keys: frozenset = frozenset()) -> Optional[Any]:
    """Return the model's final answer if it emitted a completion JSON.

    Test-case prompts instruct the agent to output ```json\n{...}\n``` to signal
    it is done. Most suites use {"answer": ...}; some declare their own result
    keys (flightradar's {"geo_altitude": ..., "vertical_rate": ...}), so a bare
    JSON object counts as a completion ONLY when its keys match what the task
    actually asked for, passed in as `result_keys`.

    That match matters. Treating *any* non-action JSON as a completion ends the
    episode the first time a model emits an intermediate object — a sketched
    plan, a partial result — which chatty models do constantly while still
    working. It cut mistral-large from ~49 steps per task to ~1.6, and the run
    still records as clean because signalling completion is not an error.

    We look at the last fenced block (same convention as _normalize_model_action).
    """
    if not isinstance(text, str):
        return None
    unwrapped = _unwrap_code_fence(text)
    stripped = unwrapped.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    # An action the model happened to encode as JSON is not a completion.
    if "action" in data or _render_structured_action(data) is not None:
        return None
    if "answer" in data:
        return data["answer"]
    # A result-format object, but only in the shape the task declared.
    if result_keys and frozenset(data) == result_keys:
        return data
    return None


def _normalize_model_action(action: Any) -> Optional[str]:
    """Normalize model output into an executable action string or `None`."""
    if action is None:
        return None

    if not isinstance(action, str):
        logger.warning(
            "Model returned a non-string action of type %s; treating it as no action.",
            type(action).__name__,
        )
        return None

    action = action.strip()
    if not action:
        return None

    unwrapped = _unwrap_code_fence(action)
    from_json = _extract_action_from_json(unwrapped)
    if from_json is not None:
        return from_json.strip() or None
    return unwrapped or None


_LITELLM_UNSUPPORTED_RE = re.compile(r"does not support parameters:\s*\[([^\]]*)\]")


def _litellm_unsupported_params(exc: Exception) -> set[str]:
    """Parameter names a LiteLLM ``UnsupportedParamsError`` says to drop.

    The gateway answers with e.g. ``azure_ai does not support parameters:
    ['seed'], for model=claude-opus-4-6``. Returns an empty set for any other
    error, so callers can re-raise anything they cannot fix by dropping a param.
    """
    message = str(exc)
    if "UnsupportedParamsError" not in message:
        return set()
    match = _LITELLM_UNSUPPORTED_RE.search(message)
    if not match:
        return set()
    return {p.strip().strip("'\"") for p in match.group(1).split(",") if p.strip()}


def _openrouter_response_diagnostics(response: Any) -> dict:
    """Extract diagnostic fields from an OpenRouter/OpenAI chat completion response."""
    info: dict = {}
    try:
        choice = response.choices[0]
    except (AttributeError, IndexError, TypeError):
        return {"error": "no choices on response"}

    info["finish_reason"] = getattr(choice, "finish_reason", None)
    info["native_finish_reason"] = getattr(choice, "native_finish_reason", None)

    message = getattr(choice, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        info["content_type"] = type(content).__name__
        info["content_len"] = len(content) if isinstance(content, str) else None
        info["has_reasoning"] = bool(getattr(message, "reasoning", None))
        info["refusal"] = getattr(message, "refusal", None)

    usage = getattr(response, "usage", None)
    if usage is not None:
        info["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
        info["completion_tokens"] = getattr(usage, "completion_tokens", None)
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            info["reasoning_tokens"] = getattr(details, "reasoning_tokens", None)

    provider_error = getattr(response, "error", None)
    if provider_error is not None:
        info["provider_error"] = provider_error

    return info


def _is_empty_content(content: Any) -> bool:
    if content is None:
        return True
    if isinstance(content, str) and not content.strip():
        return True
    return False

# Handling Screenshots
def image_to_jpg_base64_url(image: np.ndarray | Image.Image):
    """Convert a numpy array to a base64 encoded image url."""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")

    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/jpeg;base64,{image_base64}"


class DemoAgent(Agent):
    """A basic agent using OpenAI API, to demonstrate BrowserGym's functionalities."""

    def obs_preprocessor(self, obs: dict) -> dict:

        return {
            "chat_messages": obs["chat_messages"],
            "screenshot": obs["screenshot"],
            "goal_object": obs["goal_object"],
            "last_action": obs["last_action"],
            "last_action_error": obs["last_action_error"],
            "axtree_txt": flatten_axtree_to_str(obs["axtree_object"]),
            "pruned_html": prune_html(flatten_dom_to_str(obs["dom_object"])),
        }
        
    def close(self):
        """Called when the agent is being closed"""
        # Evaluate success if available
        if hasattr(self, 'last_observation') and self.last_observation:
            success = self.last_observation.get('success', None)
            reward = self.last_observation.get('reward', 0)
            time_taken = None
            if hasattr(self, 'session_start_time'):
                time_taken = time.time() - self.session_start_time
            
            if success is not None:
                rich_logger.task_complete(success, reward, time_taken)
            else:
                rich_logger.info(f"🎯 Session completed - {len(self.action_history)} actions taken")
        else:
            rich_logger.info(f"🎯 Session completed - {len(self.action_history)} actions taken")
        
        super().close()
        
    def update_last_observation(self, obs):
        """Store the last observation for metrics"""
        self.last_observation = obs

    def _resolve_reasoning_effort(self) -> Optional[str]:
        if self.reasoning_effort is not None:
            return self.reasoning_effort
        if self.reasoning is True:
            return "high"
        if self.reasoning is False:
            return "none"
        return None

    def _call_openrouter_with_retry(self, chat_kwargs: dict, label: str = "OpenRouter"):
        """Call an OpenAI-compatible endpoint; if content is empty/None, retry once.

        Log diagnostics both times. ``label`` names the upstream in those logs.
        """
        response = self.client.chat.completions.create(**chat_kwargs)
        diag = _openrouter_response_diagnostics(response)
        logger.info("%s response diagnostics: %s", label, diag)

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            content = None

        if _is_empty_content(content):
            logger.warning(
                "%s returned empty content on first attempt (%s). Retrying once.",
                label,
                diag,
            )
            retry_response = self.client.chat.completions.create(**chat_kwargs)
            retry_diag = _openrouter_response_diagnostics(retry_response)
            logger.info("%s retry response diagnostics: %s", label, retry_diag)
            try:
                retry_content = retry_response.choices[0].message.content
            except (AttributeError, IndexError, TypeError):
                retry_content = None
            if _is_empty_content(retry_content):
                logger.warning(
                    "%s retry also returned empty content (%s).", label, retry_diag
                )
            return retry_response

        return response

    def _bedrock_throttle(self) -> None:
        """Block until this process may issue another Bedrock request.

        The requests-per-minute quota is per account+model, but experiments run
        as separate processes, so the pacing state has to be shared. It lives in
        a small lock file holding the timestamp of the last request issued by
        any process using the same model.
        """
        if not self._bedrock_min_interval:
            return

        import fcntl
        import time as _time

        while True:
            with open(self._bedrock_pace_file, "a+") as fh:
                fcntl.flock(fh, fcntl.LOCK_EX)
                try:
                    fh.seek(0)
                    raw = fh.read().strip()
                    last = float(raw) if raw else 0.0
                    now = _time.time()
                    wait = last + self._bedrock_min_interval - now
                    if wait <= 0:
                        fh.seek(0)
                        fh.truncate()
                        fh.write(repr(now))
                        return
                finally:
                    # Never hold the lock while sleeping, or the processes
                    # serialise on the lock instead of on the quota.
                    fcntl.flock(fh, fcntl.LOCK_UN)
            _time.sleep(min(wait, 5.0))

    @staticmethod
    def _bedrock_content_blocks(data: dict) -> list:
        """Return the assistant content blocks from either Bedrock response shape.

        The Anthropic-native invoke API puts them at the top level; the Converse
        API nests them under output.message.
        """
        if "output" in data:
            return ((data.get("output") or {}).get("message") or {}).get("content") or []
        return data.get("content") or []

    def _call_bedrock_with_retry(self, payload: dict) -> dict:
        """POST a payload to the Bedrock runtime and return the parsed response.

        Paces requests against the configured RPM quota, then retries throttles,
        server errors and empty content with exponential backoff. On a tight
        quota (Opus is 5 RPM by default) throttling is routine rather than
        exceptional, so this has to be patient enough to ride it out.
        """
        import time as _time

        base_backoff = max(self._bedrock_min_interval or 0.0, 5.0)
        last_error = None

        for attempt in range(self._bedrock_max_attempts):
            self._bedrock_throttle()
            try:
                response = self.client.post(self._bedrock_url, json=payload)
            except Exception as exc:  # transport-level failure
                last_error = exc
                logger.warning("Bedrock request failed (attempt %d): %s", attempt + 1, exc)
                _time.sleep(min(base_backoff * (2 ** attempt), 120.0))
                continue

            if response.status_code == 200:
                data = response.json()
                if self._bedrock_content_blocks(data):
                    return data
                logger.warning(
                    "Bedrock returned no content blocks (attempt %d): stop_reason=%s usage=%s",
                    attempt + 1,
                    data.get("stop_reason") or data.get("stopReason"),
                    data.get("usage"),
                )
                last_error = ValueError("empty content in Bedrock response")
                delay = min(base_backoff * (2 ** attempt), 120.0)
            elif response.status_code in (429, 500, 502, 503, 504):
                last_error = RuntimeError(
                    f"Bedrock HTTP {response.status_code}: {response.text[:300]}"
                )
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else 0.0
                except ValueError:
                    delay = 0.0
                if not delay:
                    delay = min(base_backoff * (2 ** attempt), 120.0)
                logger.warning(
                    "Bedrock throttled/errored (attempt %d/%d): HTTP %s -- waiting %.0fs. %s",
                    attempt + 1,
                    self._bedrock_max_attempts,
                    response.status_code,
                    delay,
                    response.text[:200],
                )
            else:
                # 400/403 and friends are not worth retrying -- surface immediately.
                raise RuntimeError(
                    f"Bedrock HTTP {response.status_code}: {response.text[:500]}"
                )

            if attempt < self._bedrock_max_attempts - 1:
                _time.sleep(delay)

        raise RuntimeError(
            f"Bedrock call failed after {self._bedrock_max_attempts} attempts: {last_error}"
        )

    @staticmethod
    def _gemini_extract_text(data: dict) -> str:
        """Join the answer text out of a generateContent response.

        Gemini 3 returns its reasoning as ordinary parts flagged ``thought``,
        so those have to be dropped or the agent would try to parse them as
        actions.
        """
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = ((candidates[0].get("content") or {}).get("parts")) or []
        return "".join(
            p.get("text", "") for p in parts if p.get("text") and not p.get("thought")
        )

    @staticmethod
    def _gemini_retry_delay(data: dict) -> float:
        """Read the server-suggested wait out of a Gemini error body.

        Gemini puts it in a RetryInfo detail (``"retryDelay": "27s"``) rather
        than in a Retry-After header.
        """
        details = ((data.get("error") or {}).get("details")) or []
        for detail in details:
            raw = detail.get("retryDelay")
            if raw:
                try:
                    return float(str(raw).rstrip("s"))
                except ValueError:
                    return 0.0
        return 0.0

    def _call_gemini_with_retry(self, payload: dict) -> dict:
        """POST a Gemini-format payload to the generateContent endpoint.

        Retries throttles, server errors and responses that arrive with no
        usable text, backing off exponentially. Preview models throttle
        readily, so a 429 here is routine rather than fatal.
        """
        import time as _time

        last_error = None

        for attempt in range(self._gemini_max_attempts):
            try:
                response = self.client.post(self._gemini_url, json=payload)
            except Exception as exc:  # transport-level failure
                last_error = exc
                logger.warning("Gemini request failed (attempt %d): %s", attempt + 1, exc)
                _time.sleep(min(5.0 * (2 ** attempt), 120.0))
                continue

            if response.status_code == 200:
                data = response.json()
                if self._gemini_extract_text(data):
                    return data
                finish = (data.get("candidates") or [{}])[0].get("finishReason")
                logger.warning(
                    "Gemini returned no usable text (attempt %d): finish_reason=%s usage=%s",
                    attempt + 1,
                    finish,
                    data.get("usageMetadata"),
                )
                last_error = ValueError(f"no text in Gemini response (finish_reason={finish})")
                delay = min(5.0 * (2 ** attempt), 120.0)
            elif response.status_code in (429, 500, 502, 503, 504):
                last_error = RuntimeError(
                    f"Gemini HTTP {response.status_code}: {response.text[:300]}"
                )
                try:
                    delay = self._gemini_retry_delay(response.json())
                except ValueError:
                    delay = 0.0
                if not delay:
                    delay = min(5.0 * (2 ** attempt), 120.0)
                logger.warning(
                    "Gemini throttled/errored (attempt %d/%d): HTTP %s -- waiting %.0fs. %s",
                    attempt + 1,
                    self._gemini_max_attempts,
                    response.status_code,
                    delay,
                    response.text[:200],
                )
            else:
                # 400/403 and friends will fail identically on a retry.
                raise RuntimeError(
                    f"Gemini HTTP {response.status_code}: {response.text[:500]}"
                )

            if attempt < self._gemini_max_attempts - 1:
                _time.sleep(delay)

        raise RuntimeError(
            f"Gemini call failed after {self._gemini_max_attempts} attempts: {last_error}"
        )

    def __init__(
        self,
        model_name: str,
        chat_mode: bool,
        demo_mode: str,
        use_html: bool,
        use_axtree: bool,
        use_screenshot: bool,
        system_message_handling: Literal["separate", "combined"] = "separate",
        system_prompt_append: Optional[str] = None,
        prefix_prompt: Optional[str] = None,
        thinking_type: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        openrouter_site_url: Optional[str] = None,
        openrouter_site_name: Optional[str] = None,
        litellm_api_key: Optional[str] = None,
        litellm_base_url: Optional[str] = None,
        bedrock_api_key: Optional[str] = None,
        bedrock_region: Optional[str] = None,
        bedrock_rpm: Optional[float] = None,
        anthropic_api_key: Optional[str] = None,
        seed: Optional[int] = None,
        reasoning: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._last_token_usage = {}
        self.chat_mode = chat_mode
        self.use_html = use_html
        self.use_axtree = use_axtree
        self.use_screenshot = use_screenshot
        self.system_message_handling = system_message_handling
        self.system_prompt_append = system_prompt_append
        self.prefix_prompt = prefix_prompt
        self.thinking_type = thinking_type
        self.seed = seed
        self.reasoning = reasoning
        self.reasoning_effort = reasoning_effort
        self.provider = provider

        if not (use_html or use_axtree):
            raise ValueError(f"Either use_html or use_axtree must be set to True.")

        from openai import OpenAI
        from anthropic import Anthropic
        import os

        if model_name.startswith("gpt-") or model_name.startswith("o1") or model_name.startswith("o3"):
            # Use provided API key or fall back to environment variable
            self.client = OpenAI(api_key=openai_api_key)
            self.model_name = model_name
            responses_only_model_prefixes = ("o1-pro", "o3-pro")
            reasoning_model_prefixes = ("o1", "o3", "gpt-5.4")

            def is_responses_only_model(name: str) -> bool:
                return any(name.startswith(prefix) for prefix in responses_only_model_prefixes)

            def is_reasoning_model(name: str) -> bool:
                return any(name.startswith(prefix) for prefix in reasoning_model_prefixes)

            def build_responses_content(messages: list[dict]) -> list[dict]:
                content = []
                for msg in messages:
                    if msg["type"] == "text":
                        content.append({"type": "input_text", "text": msg["text"]})
                    elif msg["type"] == "image_url":
                        image_url = msg["image_url"]
                        detail = "auto"
                        if isinstance(image_url, dict):
                            detail = image_url.get("detail", "auto")
                            image_url = image_url["url"]
                        content.append(
                            {
                                "type": "input_image",
                                "image_url": image_url,
                                "detail": detail,
                            }
                        )
                    else:
                        raise ValueError(f"Unsupported message type for Responses API: {msg['type']}")
                return content

            def build_chat_user_content(messages: list[dict]) -> list[dict]:
                content = []
                for msg in messages:
                    if msg["type"] == "text":
                        content.append({"type": "text", "text": msg["text"]})
                    elif msg["type"] == "image_url":
                        content.append({"type": "image_url", "image_url": msg["image_url"]})
                    else:
                        raise ValueError(f"Unsupported message type for Chat Completions API: {msg['type']}")
                return content

            # Define function to query OpenAI models
            def query_model(system_msgs, user_msgs):
                if is_responses_only_model(self.model_name):
                    if self.system_message_handling == "combined":
                        combined_user_msgs = []
                        if system_msgs:
                            combined_user_msgs.append(
                                {
                                    "type": "text",
                                    "text": system_msgs[0]["text"],
                                }
                            )
                        combined_user_msgs.extend(user_msgs)
                        instructions = None
                        input_messages = [
                            {
                                "role": "user",
                                "content": build_responses_content(combined_user_msgs),
                            }
                        ]
                    else:
                        instructions = system_msgs[0]["text"] if system_msgs else None
                        input_messages = [
                            {
                                "role": "user",
                                "content": build_responses_content(user_msgs),
                            }
                        ]

                    responses_kwargs = {
                        "model": self.model_name,
                        "instructions": instructions,
                        "input": input_messages,
                    }
                    effort = self._resolve_reasoning_effort()
                    if effort is not None:
                        responses_kwargs["reasoning"] = {"effort": effort}
                    response = self.client.responses.create(**responses_kwargs)

                    if hasattr(response, "usage") and response.usage:
                        self._last_token_usage = {
                            "input_tokens": getattr(response.usage, "input_tokens", 0) or 0,
                            "output_tokens": getattr(response.usage, "output_tokens", 0) or 0,
                            "total_tokens": getattr(response.usage, "total_tokens", 0) or 0,
                        }

                    if not response.output_text:
                        raise ValueError(f"No text content returned by Responses API for model {self.model_name}")
                    return response.output_text

                if self.system_message_handling == "combined":
                    # Combine system and user text messages into a single user message.
                    # Chat Completions cannot preserve image blocks in this mode.
                    combined_content = ""
                    if system_msgs:
                        combined_content += system_msgs[0]["text"] + "\n\n"
                    for msg in user_msgs:
                        if msg["type"] == "text":
                            combined_content += msg["text"] + "\n"
                    chat_kwargs = {
                        "model": self.model_name,
                        "messages": [
                            {"role": "user", "content": combined_content},
                        ],
                    }
                    if self.seed is not None:
                        chat_kwargs["seed"] = self.seed
                    effort = self._resolve_reasoning_effort()
                    if effort is not None and is_reasoning_model(self.model_name):
                        chat_kwargs["reasoning_effort"] = effort
                    response = self.client.chat.completions.create(**chat_kwargs)
                else:
                    chat_kwargs = {
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": system_msgs[0]["text"] if system_msgs else ""},
                            {"role": "user", "content": build_chat_user_content(user_msgs)},
                        ],
                    }
                    if self.seed is not None:
                        chat_kwargs["seed"] = self.seed
                    effort = self._resolve_reasoning_effort()
                    if effort is not None and is_reasoning_model(self.model_name):
                        chat_kwargs["reasoning_effort"] = effort
                    response = self.client.chat.completions.create(**chat_kwargs)
                if hasattr(response, "usage") and response.usage:
                    self._last_token_usage = {
                        "input_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                        "output_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                        "total_tokens": getattr(response.usage, "total_tokens", 0) or 0,
                    }
                # The OpenRouter/LiteLLM path records this via its retry wrapper;
                # the direct path had no equivalent, which left "was reasoning
                # actually off?" unanswerable after the fact. Log the same two
                # fields here so a finished run can be audited from its log.
                _details = getattr(getattr(response, "usage", None), "completion_tokens_details", None)
                _msg = getattr(response.choices[0], "message", None) if response.choices else None
                logger.info(
                    "OpenAI direct: {'reasoning_effort': %r, 'has_reasoning': %r, 'reasoning_tokens': %r}",
                    chat_kwargs.get("reasoning_effort"),
                    bool(getattr(_msg, "reasoning", None)),
                    getattr(_details, "reasoning_tokens", None),
                )
                return response.choices[0].message.content
            self.query_model = query_model

        elif model_name.startswith("openrouter/"):
            # Extract the actual model name without the openrouter/ prefix
            actual_model_name = model_name.replace("openrouter/", "", 1)
            
            # Initialize OpenRouter client (using OpenAI client with custom base URL)
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_api_key or os.getenv("OPENROUTER_API_KEY"),
            )
            # Store site info for headers
            self.openrouter_site_url = openrouter_site_url or os.getenv("OPENROUTER_SITE_URL", "")
            self.openrouter_site_name = openrouter_site_name or os.getenv("OPENROUTER_SITE_NAME", "")
            self.model_name = actual_model_name
            
            # Define function to query OpenRouter models
            def query_model(system_msgs, user_msgs):
                if self.system_message_handling == "combined":
                    # Combine system and user messages into a single user message
                    combined_content = ""
                    if system_msgs:
                        combined_content += system_msgs[0]["text"] + "\n\n"
                    for msg in user_msgs:
                        if msg["type"] == "text":
                            combined_content += msg["text"] + "\n"
                    chat_kwargs = {
                        "extra_headers": {
                            "HTTP-Referer": self.openrouter_site_url,
                            "X-Title": self.openrouter_site_name,
                        },
                        "model": self.model_name,
                        "messages": [
                            {"role": "user", "content": combined_content},
                        ],
                    }
                    if self.seed is not None:
                        chat_kwargs["seed"] = self.seed
                    effort = self._resolve_reasoning_effort()
                    if effort is not None:
                        chat_kwargs.setdefault("extra_body", {})["reasoning"] = {
                            "effort": effort
                        }
                    if self.provider:
                        chat_kwargs.setdefault("extra_body", {})["provider"] = {
                            "only": [self.provider],
                            "allow_fallbacks": False,
                        }
                    response = self._call_openrouter_with_retry(chat_kwargs)
                else:
                    # Format messages properly - extract text content
                    formatted_messages = []
                    if system_msgs:
                        formatted_messages.append({"role": "system", "content": system_msgs[0]["text"]})

                    # Convert user messages to OpenAI format
                    user_content = []
                    for msg in user_msgs:
                        if msg["type"] == "text":
                            user_content.append({"type": "text", "text": msg["text"]})
                        elif msg["type"] == "image_url":
                            user_content.append({"type": "image_url", "image_url": msg["image_url"]})

                    formatted_messages.append({"role": "user", "content": user_content})

                    chat_kwargs = {
                        "extra_headers": {
                            "HTTP-Referer": self.openrouter_site_url,
                            "X-Title": self.openrouter_site_name,
                        },
                        "model": self.model_name,
                        "messages": formatted_messages,
                    }
                    if self.seed is not None:
                        chat_kwargs["seed"] = self.seed
                    effort = self._resolve_reasoning_effort()
                    if effort is not None:
                        chat_kwargs.setdefault("extra_body", {})["reasoning"] = {
                            "effort": effort
                        }
                    if self.provider:
                        chat_kwargs.setdefault("extra_body", {})["provider"] = {
                            "only": [self.provider],
                            "allow_fallbacks": False,
                        }
                    response = self._call_openrouter_with_retry(chat_kwargs)
                if hasattr(response, "usage") and response.usage:
                    self._last_token_usage = {
                        "input_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                        "output_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                        "total_tokens": getattr(response.usage, "total_tokens", 0) or 0,
                    }
                try:
                    return response.choices[0].message.content
                except (AttributeError, IndexError, TypeError):
                    logger.warning(
                        "OpenRouter response missing choices/message/content: %s",
                        _openrouter_response_diagnostics(response),
                    )
                    return ""
            self.query_model = query_model

        elif model_name.startswith("litellm/"):
            # Any model served by a LiteLLM proxy, addressed by the exact model
            # name the gateway exposes (e.g. litellm/gpt-5.4).
            actual_model_name = model_name.replace("litellm/", "", 1)

            base_url = litellm_base_url or os.getenv("LITELLM_BASE_URL")
            if not base_url:
                raise ValueError(
                    "LITELLM_BASE_URL must be set (or litellm_base_url passed) to use a litellm/ model."
                )
            api_key = litellm_api_key or os.getenv("LITELLM_API_KEY")
            if not api_key:
                raise ValueError(
                    "LITELLM_API_KEY must be set (or litellm_api_key passed) to use a litellm/ model."
                )

            # LiteLLM speaks the OpenAI wire protocol, so the OpenAI client works
            # against it directly.
            self.client = OpenAI(base_url=base_url, api_key=api_key)
            self.model_name = actual_model_name

            # A LiteLLM gateway fans one OpenAI-shaped request out to several
            # upstream providers, and they do not accept the same parameters:
            # azure's GPT route takes `seed`, azure_ai's Claude route rejects it
            # with a 400. The gateway names the offending params in the error, so
            # drop those and retry rather than failing the run — and remember
            # them, to spend one failed call per run instead of one per step.
            unsupported: set[str] = set()

            def query_model(system_msgs, user_msgs):
                if self.system_message_handling == "combined":
                    combined_content = ""
                    if system_msgs:
                        combined_content += system_msgs[0]["text"] + "\n\n"
                    for msg in user_msgs:
                        if msg["type"] == "text":
                            combined_content += msg["text"] + "\n"
                    messages = [{"role": "user", "content": combined_content}]
                else:
                    messages = []
                    if system_msgs:
                        messages.append({"role": "system", "content": system_msgs[0]["text"]})
                    user_content = []
                    for msg in user_msgs:
                        if msg["type"] == "text":
                            user_content.append({"type": "text", "text": msg["text"]})
                        elif msg["type"] == "image_url":
                            user_content.append({"type": "image_url", "image_url": msg["image_url"]})
                    messages.append({"role": "user", "content": user_content})

                chat_kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                }
                if self.seed is not None:
                    chat_kwargs["seed"] = self.seed
                effort = self._resolve_reasoning_effort()
                if effort is not None:
                    # LiteLLM maps reasoning_effort onto whatever the upstream
                    # provider expects.
                    chat_kwargs["reasoning_effort"] = effort
                for param in unsupported:
                    chat_kwargs.pop(param, None)

                try:
                    response = self._call_openrouter_with_retry(chat_kwargs, label="LiteLLM")
                except Exception as exc:
                    dropped = _litellm_unsupported_params(exc) & set(chat_kwargs)
                    if not dropped:
                        raise
                    unsupported.update(dropped)
                    logger.warning(
                        "LiteLLM upstream for %s rejects %s; retrying without %s "
                        "and omitting them for the rest of this run.",
                        self.model_name, sorted(dropped), sorted(dropped),
                    )
                    for param in dropped:
                        chat_kwargs.pop(param, None)
                    response = self._call_openrouter_with_retry(chat_kwargs, label="LiteLLM")

                if hasattr(response, "usage") and response.usage:
                    self._last_token_usage = {
                        "input_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                        "output_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                        "total_tokens": getattr(response.usage, "total_tokens", 0) or 0,
                    }
                try:
                    return response.choices[0].message.content
                except (AttributeError, IndexError, TypeError):
                    logger.warning(
                        "LiteLLM response missing choices/message/content: %s",
                        _openrouter_response_diagnostics(response),
                    )
                    return ""
            self.query_model = query_model

        elif model_name.startswith("bedrock/"):
            # Anthropic models served by AWS Bedrock, addressed by the exact model
            # id Bedrock exposes. Every current Claude model is inference-profile
            # only, so that id normally carries a regional prefix, e.g.
            # bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0
            import httpx

            self.model_name = model_name.replace("bedrock/", "", 1)
            base_model_name = self.model_name.replace(":thinking", "")
            thinking_from_suffix = self.model_name.endswith(":thinking")
            self.model_name = base_model_name

            region = (
                bedrock_region
                or os.getenv("BEDROCK_REGION")
                or os.getenv("AWS_REGION")
                or "us-east-1"
            )
            api_key = (
                bedrock_api_key
                or os.getenv("BEDROCK_API_KEY")
                or os.getenv("AWS_BEARER_TOKEN_BEDROCK")
            )
            if not api_key:
                raise ValueError(
                    "BEDROCK_API_KEY (or AWS_BEARER_TOKEN_BEDROCK) must be set, or "
                    "bedrock_api_key passed, to use a bedrock/ model."
                )

            # Anthropic models are addressed through the native invoke API, whose
            # bodies are the Anthropic message format -- so that path reuses the
            # conversion logic written for the Anthropic API. Everything else
            # (Llama, Nova, Mistral, ...) goes through Converse, the uniform
            # cross-vendor API, which has its own message shape.
            self._bedrock_uses_converse = "anthropic" not in self.model_name.lower()
            endpoint = "converse" if self._bedrock_uses_converse else "invoke"
            # A Bedrock API key is a bearer token, so the runtime endpoint can be
            # called directly: no SigV4 signing, no boto3.
            self._bedrock_url = (
                f"https://bedrock-runtime.{region}.amazonaws.com/model/{self.model_name}/{endpoint}"
            )
            self.client = httpx.Client(
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(300.0, connect=30.0),
            )

            # Bedrock per-model quotas are low by default (Opus is 5 RPM), and
            # one agent step is one request, so pace deliberately rather than
            # discovering the limit through 429s.
            rpm = bedrock_rpm
            if rpm is None and os.getenv("BEDROCK_RPM"):
                rpm = float(os.getenv("BEDROCK_RPM"))
            self._bedrock_min_interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
            self._bedrock_max_attempts = int(os.getenv("BEDROCK_MAX_ATTEMPTS", "6"))
            # Shared by every process driving the same model, so the pace file is
            # keyed on the model id rather than being per-process.
            import re as _re
            import tempfile

            self._bedrock_pace_file = os.path.join(
                tempfile.gettempdir(),
                f"bedrock_pace_{_re.sub(r'[^A-Za-z0-9]+', '_', self.model_name)}.lock",
            )
            if self._bedrock_min_interval:
                logger.info(
                    "Bedrock pacing: %.1f RPM (%.1fs between requests), shared via %s",
                    rpm,
                    self._bedrock_min_interval,
                    self._bedrock_pace_file,
                )

            if self.reasoning is not None:
                thinking_enabled = self.reasoning
            else:
                thinking_enabled = thinking_from_suffix

            if thinking_enabled:
                if self.thinking_type == "adaptive":
                    thinking = {"type": "adaptive"}
                    max_tokens = 8000
                else:
                    budget_tokens = 10000
                    thinking = {"type": "enabled", "budget_tokens": budget_tokens}
                    # max_tokens must exceed the thinking budget, or Bedrock 400s.
                    max_tokens = budget_tokens + 8000
            else:
                thinking = None
                max_tokens = 8000

            def query_model_converse(system_msgs, user_msgs):
                content = []
                for msg in user_msgs:
                    if msg["type"] == "text":
                        content.append({"text": msg["text"]})
                    elif msg["type"] == "image_url":
                        image_url = msg["image_url"]
                        if isinstance(image_url, dict):
                            image_url = image_url["url"]
                        if image_url.startswith("data:image/jpeg;base64,"):
                            # Converse blobs are base64 strings over the REST API
                            # (the AWS SDKs take raw bytes and encode them).
                            content.append({
                                "image": {
                                    "format": "jpeg",
                                    "source": {
                                        "bytes": image_url.replace("data:image/jpeg;base64,", "")
                                    },
                                }
                            })
                        else:
                            content.append({"text": "[Image URL not supported by Bedrock]"})

                system_content = system_msgs[0]["text"] if system_msgs else ""
                if self.system_message_handling == "combined" and system_content:
                    content = [{"text": system_content}] + content
                    system_content = ""

                inference_config = {"maxTokens": max_tokens}
                if self.seed is not None:
                    inference_config["temperature"] = 0.0

                payload = {
                    "messages": [{"role": "user", "content": content}],
                    "inferenceConfig": inference_config,
                }
                if system_content:
                    payload["system"] = [{"text": system_content}]

                data = self._call_bedrock_with_retry(payload)

                usage = data.get("usage") or {}
                self._last_token_usage = {
                    "input_tokens": usage.get("inputTokens", 0) or 0,
                    "output_tokens": usage.get("outputTokens", 0) or 0,
                    "total_tokens": usage.get("totalTokens", 0) or 0,
                }

                blocks = self._bedrock_content_blocks(data)
                for block in blocks:
                    if "text" in block:
                        return block["text"]
                    if "reasoningContent" in block:
                        logger.debug("Skipping reasoningContent block")

                logger.warning(
                    "No text content in Bedrock Converse response (stopReason=%s)",
                    data.get("stopReason"),
                )
                raise ValueError("No text content in Bedrock Converse response")

            def query_model(system_msgs, user_msgs):
                anthropic_content = []
                for msg in user_msgs:
                    if msg["type"] == "text":
                        anthropic_content.append({"type": "text", "text": msg["text"]})
                    elif msg["type"] == "image_url":
                        image_url = msg["image_url"]
                        if isinstance(image_url, dict):
                            image_url = image_url["url"]

                        if image_url.startswith("data:image/jpeg;base64,"):
                            anthropic_content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_url.replace("data:image/jpeg;base64,", ""),
                                },
                            })
                        else:
                            anthropic_content.append(
                                {"type": "text", "text": "[Image URL not supported by Bedrock]"}
                            )

                if self.system_message_handling == "combined" and system_msgs:
                    anthropic_content = [
                        {"type": "text", "text": system_msgs[0]["text"]}
                    ] + anthropic_content
                    system_content = None
                else:
                    system_content = system_msgs[0]["text"] if system_msgs else ""

                payload = {
                    # Bedrock takes the API version in the body rather than a header.
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": anthropic_content}],
                }
                if thinking is not None:
                    payload["thinking"] = thinking
                # No seed parameter exists here either; temperature=0 is the closest
                # equivalent, and is rejected while thinking is enabled.
                if self.seed is not None and (thinking or {}).get("type") != "enabled":
                    payload["temperature"] = 0.0
                if system_content:
                    payload["system"] = system_content

                data = self._call_bedrock_with_retry(payload)

                usage = data.get("usage") or {}
                self._last_token_usage = {
                    "input_tokens": usage.get("input_tokens", 0) or 0,
                    "output_tokens": usage.get("output_tokens", 0) or 0,
                    "total_tokens": (usage.get("input_tokens", 0) or 0)
                        + (usage.get("output_tokens", 0) or 0),
                }

                blocks = data.get("content", [])
                logger.info(
                    "Bedrock response content types: %s", [b.get("type") for b in blocks]
                )
                for block in blocks:
                    if block.get("type") == "text":
                        return block.get("text", "")
                    if block.get("type") == "thinking":
                        logger.debug("Thinking block: %s...", str(block.get("thinking"))[:100])

                logger.warning(
                    "No text content in Bedrock response (stop_reason=%s)",
                    data.get("stop_reason"),
                )
                raise ValueError("No text content in Bedrock response")

            self.query_model = (
                query_model_converse if self._bedrock_uses_converse else query_model
            )

        elif model_name.startswith("gemini/"):
            # Google's own Gemini API, addressed by the id that API exposes,
            # e.g. gemini/gemini-3.1-pro-preview. This is the native
            # generativelanguage endpoint keyed on GEMINI_API_KEY, not the
            # OpenAI-compatible shim, so the bodies are Gemini's own format.
            import httpx

            self.model_name = model_name.replace("gemini/", "", 1)

            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY (or GOOGLE_API_KEY) must be set to use a gemini/ model."
                )

            api_base = os.getenv(
                "GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta"
            )
            self._gemini_url = f"{api_base.rstrip('/')}/models/{self.model_name}:generateContent"
            self.client = httpx.Client(
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                timeout=httpx.Timeout(300.0, connect=30.0),
            )
            self._gemini_max_attempts = int(os.getenv("GEMINI_MAX_ATTEMPTS", "6"))
            # Gemini 3 spends part of this budget on reasoning before it emits
            # any answer, so the ceiling has to be well above the ~1k tokens an
            # action actually needs or responses come back truncated and empty.
            max_output_tokens = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "16000"))

            effort = self._resolve_reasoning_effort()
            if effort is None:
                thinking_config = None
            else:
                # Gemini exposes a coarse level rather than a token budget, and
                # reasoning cannot be switched off on the Pro models.
                thinking_config = {"thinkingLevel": "high" if effort == "high" else "low"}

            def query_model(system_msgs, user_msgs):
                parts = []
                for msg in user_msgs:
                    if msg["type"] == "text":
                        parts.append({"text": msg["text"]})
                    elif msg["type"] == "image_url":
                        image_url = msg["image_url"]
                        if isinstance(image_url, dict):
                            image_url = image_url["url"]

                        if image_url.startswith("data:image/jpeg;base64,"):
                            parts.append({
                                "inlineData": {
                                    "mimeType": "image/jpeg",
                                    "data": image_url.replace("data:image/jpeg;base64,", ""),
                                },
                            })
                        else:
                            parts.append({"text": "[Image URL not supported by Gemini]"})

                payload = {
                    "contents": [{"role": "user", "parts": parts}],
                    "generationConfig": {"maxOutputTokens": max_output_tokens},
                }

                if self.system_message_handling == "combined" and system_msgs:
                    payload["contents"][0]["parts"] = [
                        {"text": system_msgs[0]["text"]}
                    ] + parts
                elif system_msgs:
                    payload["systemInstruction"] = {
                        "parts": [{"text": system_msgs[0]["text"]}]
                    }

                if thinking_config is not None:
                    payload["generationConfig"]["thinkingConfig"] = thinking_config
                if self.seed is not None:
                    # Gemini does honour a seed, but only pins sampling when the
                    # temperature is pinned too.
                    payload["generationConfig"]["seed"] = self.seed
                    payload["generationConfig"]["temperature"] = 0.0

                data = self._call_gemini_with_retry(payload)

                usage = data.get("usageMetadata") or {}
                self._last_token_usage = {
                    "input_tokens": usage.get("promptTokenCount", 0) or 0,
                    # Reasoning tokens are billed as output but reported apart.
                    "output_tokens": (usage.get("candidatesTokenCount", 0) or 0)
                        + (usage.get("thoughtsTokenCount", 0) or 0),
                    "total_tokens": usage.get("totalTokenCount", 0) or 0,
                }

                text = self._gemini_extract_text(data)
                if not text:
                    logger.warning(
                        "No text content in Gemini response (finish_reason=%s)",
                        (data.get("candidates") or [{}])[0].get("finishReason"),
                    )
                    raise ValueError("No text content in Gemini response")
                return text
            self.query_model = query_model

        elif any(model_name.startswith(prefix) for prefix in ["claude-", "sonnet-"]):
            # Comprehensive model mapping for all Claude models
            ANTHROPIC_MODELS = {
                "claude-3-opus": "claude-3-opus-20240229",
                "claude-3-sonnet": "claude-3-sonnet-20240229",
                "claude-3-haiku": "claude-3-haiku-20240307",
                "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
                "claude-opus-4": "claude-opus-4-20250514",
                "claude-sonnet-4": "claude-sonnet-4-20250514",
                "sonnet-3.7": "claude-3-7-sonnet-20250219",
                "claude-opus-4-6": "claude-opus-4-6"
            }
            
            # Parse model name and thinking mode
            base_model_name = model_name.replace(":thinking", "")
            thinking_from_suffix = model_name.endswith(":thinking")

            # Get the actual model ID
            if base_model_name in ANTHROPIC_MODELS:
                self.model_name = ANTHROPIC_MODELS[base_model_name]
            else:
                # If not in mapping, assume it's a direct model ID
                self.model_name = base_model_name

            # Initialize Anthropic client
            self.client = Anthropic(api_key=anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"))

            # Configure thinking: explicit reasoning param overrides :thinking suffix
            if self.reasoning is not None:
                thinking_enabled = self.reasoning
            else:
                thinking_enabled = thinking_from_suffix

            if thinking_enabled:
                if self.thinking_type == "adaptive":
                    thinking = {"type": "adaptive"}
                else:
                    thinking = {"type": "enabled", "budget_tokens": 10000}
            else:
                thinking = {"type": "disabled"}
                
            # Define function to query Anthropic models
            def query_model(system_msgs, user_msgs):
                # Convert OpenAI format messages to Anthropic format
                anthropic_content = []
                for msg in user_msgs:
                    if msg["type"] == "text":
                        anthropic_content.append({"type": "text", "text": msg["text"]})
                    elif msg["type"] == "image_url":
                        # Handle base64 image URLs for Anthropic
                        image_url = msg["image_url"]
                        if isinstance(image_url, dict):
                            image_url = image_url["url"]
                        
                        if image_url.startswith("data:image/jpeg;base64,"):
                            base64_data = image_url.replace("data:image/jpeg;base64,", "")
                            anthropic_content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_data
                                }
                            })
                        else:
                            # Skip external URLs or unsupported image formats
                            anthropic_content.append({"type": "text", "text": "[Image URL not supported by Anthropic API]"})
                
                # Handle system message based on system_message_handling
                if self.system_message_handling == "combined" and system_msgs:
                    # Prepend system message to user content
                    combined_content = [{"type": "text", "text": system_msgs[0]["text"]}]
                    combined_content.extend(anthropic_content)
                    anthropic_content = combined_content
                    system_content = None
                else:
                    # Use separate system message - extract text from the list
                    system_content = system_msgs[0]["text"] if system_msgs else ""
                
                # Simple messages array - no manual thinking block management
                messages = [{
                    "role": "user", 
                    "content": anthropic_content
                }]
                
                # Make API request - conditionally include system parameter
                create_params = {
                    "model": self.model_name,
                    "max_tokens": 8000,
                    "messages": messages,
                    "thinking": thinking,
                }
                # Anthropic has no seed param; temperature=0 is the closest
                # equivalent for determinism (not allowed when thinking is enabled)
                if self.seed is not None and thinking.get("type") != "enabled":
                    create_params["temperature"] = 0.0
                
                # Only add system parameter if we have content
                if system_content is not None:
                    create_params["system"] = system_content
                    
                response = self.client.messages.create(**create_params)

                if hasattr(response, "usage") and response.usage:
                    self._last_token_usage = {
                        "input_tokens": getattr(response.usage, "input_tokens", 0) or 0,
                        "output_tokens": getattr(response.usage, "output_tokens", 0) or 0,
                        "total_tokens": (getattr(response.usage, "input_tokens", 0) or 0)
                            + (getattr(response.usage, "output_tokens", 0) or 0),
                    }

                # Log response content types for debugging
                logger.info(f"Response content types: {[content.type for content in response.content]}")
                
                # Extract text content, handling all block types properly
                text_content = None
                for content_block in response.content:
                    if content_block.type == "text":
                        text_content = content_block.text
                        break
                    elif content_block.type == "thinking":
                        # Log thinking content for debugging but don't return it
                        logger.debug(f"Thinking block: {content_block.thinking[:100]}...")
                    elif content_block.type == "redacted_thinking":
                        # Log that we encountered redacted thinking
                        logger.debug("Encountered redacted thinking block")
                
                if text_content is None:
                    logger.warning("No text content found in response")
                    # This shouldn't happen with a properly formed response
                    raise ValueError("No text content in Anthropic response")
                
                return text_content
            self.query_model = query_model
        else:
            raise ValueError(f"Model {model_name} not supported. Use a model name starting with 'gpt-', 'claude-', 'sonnet-', 'openrouter/' followed by the OpenRouter model ID, 'litellm/' followed by the model name your LiteLLM proxy exposes, 'bedrock/' followed by the AWS Bedrock model id, or 'gemini/' followed by the Gemini API model id.")

        self.action_set = HighLevelActionSet(
            subsets=["chat", "bid", "coord", "infeas"],  # allow the agent to use absolute x,y mouse actions
            strict=False,  # less strict on the parsing of the actions
            multiaction=True,  # allow sequential actions in a single turn
            demo_mode=demo_mode,  # add visual effects
        )
        # use this instead to allow the agent to directly use Python code
        # self.action_set = PythonActionSet())

        self.action_history = []
        self.last_observation = None

    @staticmethod
    def _goal_text(obs: dict) -> str:
        """The task prompt as plain text, from whichever shape the goal arrives in."""
        goal_object = obs.get("goal_object")
        if isinstance(goal_object, list):
            texts = [
                part["text"]
                for part in goal_object
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ]
            if texts:
                return "\n".join(texts)
        return str(goal_object or "")

    def get_action(self, obs: dict) -> tuple[Optional[str], dict]:
        # Print task start information if this is the first action
        if len(self.action_history) == 0:
            goal_str = self._goal_text(obs)
            rich_logger.task_start(goal_str, self.model_name)
            self.session_start_time = time.time()
            
        system_msgs = []
        user_msgs = []

        def append_system_message(default_text: str):
            system_text = default_text
            if self.system_prompt_append:
                system_text = (
                    default_text.rstrip()
                    + "\n\n# Additional App-Specific Instructions\n\n"
                    + self.system_prompt_append.strip()
                )
            system_msgs.append(
                {
                    "type": "text",
                    "text": system_text,
                }
            )

        if self.chat_mode:
            append_system_message(
                f"""\
                            # Instructions

                            You are a UI Assistant, your goal is to help the user perform tasks using a web browser. You can
                            communicate with the user via a chat, to which the user gives you instructions and to which you
                            can send back messages. You have access to a web browser that both you and the user can see,
                            and with which only you can interact via specific commands.

                            Review the instructions from the user, the current state of the page and all other information
                            to find the best possible next action to accomplish your goal. Your answer will be interpreted
                            and executed by a program, make sure to follow the formatting instructions.
                            """
            )
            # append chat messages
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
                            # Chat Messages
                            """,
                }
            )
            for msg in obs["chat_messages"]:
                if msg["role"] in ("user", "assistant", "infeasible"):
                    user_msgs.append(
                        {
                            "type": "text",
                            "text": f"""\
                                    - [{msg['role']}] {msg['message']}
                                    """,
                        }
                    )
                elif msg["role"] == "user_image":
                    user_msgs.append({"type": "image_url", "image_url": msg["message"]})
                else:
                    raise ValueError(f"Unexpected chat message role {repr(msg['role'])}")

        else:
            assert obs["goal_object"], "The goal is missing."
            append_system_message(
                f"""\
                            # Instructions

                            Review the current state of the page and all other information to find the best
                            possible next action to accomplish your goal. Your answer will be interpreted
                            and executed by a program, make sure to follow the formatting instructions.
                            """
            )
            # append goal, with optional prefix prepended directly into the prompt
            goal_object = list(obs["goal_object"])
            if self.prefix_prompt and goal_object and goal_object[0].get("type") == "text":
                goal_object[0] = {
                    **goal_object[0],
                    "text": self.prefix_prompt.strip() + "\n\n" + goal_object[0]["text"],
                }
            elif self.prefix_prompt:
                goal_object.insert(0, {"type": "text", "text": self.prefix_prompt.strip()})

            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
                            # Goal
                            """,
                }
            )
            # goal_object is directly presented as a list of openai-style messages
            user_msgs.extend(goal_object)

        # append page AXTree (if asked)
        if self.use_axtree:
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
                            # Current page Accessibility Tree
                            
                            {obs["axtree_txt"]}
                            
                            """,
                }
            )
        # append page HTML (if asked)
        if self.use_html:
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
                            # Current page DOM
                            
                            {obs["pruned_html"]}
                            
                            """,
                }
            )

        # append page screenshot (if asked)
        if self.use_screenshot:
            user_msgs.append(
                {
                    "type": "text",
                    "text": """\
                            # Current page Screenshot
                            """,
                }
            )
            user_msgs.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_jpg_base64_url(obs["screenshot"]),
                        "detail": "auto",
                    },  # Literal["low", "high", "auto"] = "auto"
                }
            )

        # append action space description
        user_msgs.append(
            {
                "type": "text",
                "text": f"""\
                        # Action Space

                        {self.action_set.describe(with_long_description=False, with_examples=True)}

                        Here are examples of actions with chain-of-thought reasoning:

                        I now need to click on the Submit button to send the form. I will use the click action on the button, which has bid 12.
                        ```click("12")```

                        I found the information requested by the user, I will send it to the chat.
                        ```send_msg_to_user("The price for a 15\\" laptop is 1499 USD.")```

                        """,
                }
            )

        # append past actions (and last error message) if any
        if self.action_history:
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
                            # History of past actions
                            """,
                }
            )
            user_msgs.extend(
                [
                    {
                        "type": "text",
                        "text": f"""\
{action}
""",
                    }
                    for action in self.action_history
                ]
            )

            if obs["last_action_error"]:
                # Log error to console
                rich_logger.error(f"Error: {str(obs['last_action_error'])[:100]}...")
                
                # Add error to message
                user_msgs.append(
                    {
                        "type": "text",
                        "text": f"""\
                                # Error message from last action

                                {obs["last_action_error"]}

                                """,
                    }
                )

        # ask for the next action
        user_msgs.append(
            {
                "type": "text",
                "text": f"""\
                        # Next action

                        You will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action.

                        IMPORTANT: Only return your answer in the RESULT FORMAT after you have fully completed the task. Do not emit the result format prematurely — continue taking actions until the task is fully complete.
                        """,
            }
        )

        prompt_text_strings = []
        for message in system_msgs + user_msgs:
            match message["type"]:
                case "text":
                    prompt_text_strings.append(message["text"])
                case "image_url":
                    image_url = message["image_url"]
                    if isinstance(message["image_url"], dict):
                        image_url = image_url["url"]
                    if image_url.startswith("data:image"):
                        prompt_text_strings.append(
                            "image_url: " + image_url[:30] + "... (truncated)"
                        )
                    else:
                        prompt_text_strings.append("image_url: " + image_url)
                case _:
                    raise ValueError(
                        f"Unknown message type {repr(message['type'])} in the task goal."
                    )
        full_prompt_txt = "\n".join(prompt_text_strings)
        # Don't log the full prompt - too verbose
        # logger.info(full_prompt_txt)

        # Save full prompt text on first step for summary reporting
        is_first_step = len(self.action_history) == 0
        if is_first_step:
            self._first_step_prompt = full_prompt_txt

        # query model using the abstraction function
        self._last_token_usage = {}
        raw_action = self.query_model(system_msgs, user_msgs)
        token_usage = self._last_token_usage

        step_num = len(self.action_history) + 1

        answer = (
            _extract_answer_from_text(raw_action, _declared_result_keys(self._goal_text(obs)))
            if raw_action
            else None
        )
        action_from_answer: Optional[str] = None
        if answer is not None and isinstance(answer, str):
            candidate = answer.strip()
            if candidate:
                try:
                    self.action_set.to_python_code(candidate)
                except Exception:
                    action_from_answer = None
                else:
                    action_from_answer = candidate
                    logger.info(
                        "Answer field parsed as a valid action %r; treating as action instead of completion.",
                        candidate,
                    )
        if answer is not None and action_from_answer is None:
            rich_logger.task_step(step_num, "answer", details=f"Agent signaled completion: {answer!r}")
            logger.info("Agent emitted completion answer %r; ending episode.", answer)
            self.action_history.append(f'answer({answer!r})')
            self.update_last_observation(obs)
            info = {
                "model_response": None,
                "raw_model_response": raw_action,
                "answer": answer,
                "stats": token_usage,
            }
            if is_first_step:
                info["full_prompt"] = full_prompt_txt
            return None, info

        action = action_from_answer or _normalize_model_action(raw_action)

        if action is None:
            rich_logger.warning("Model returned no executable action; falling back to noop(500) and continuing.")
            rich_logger.task_step(step_num, "noop", details="Model response was empty or invalid; substituting noop.")
            logger.warning("Model returned an empty or invalid action response: %r. Substituting noop(500).", raw_action)
            action = "noop(500)"
            if not raw_action:
                raw_action = action

        # Extract action type for a cleaner log message
        action_type = action.split("(")[0] if "(" in action else "unknown"
        action_args = action.split("(", 1)[1].rstrip(")") if "(" in action else ""

        # Log concise action summary to console
        action_summary = f"{action_type}"
        if action_args:
            action_summary += f"({action_args[:50]}{'...' if len(action_args) > 50 else ''})"
        
        rich_logger.task_step(step_num, action_summary)

        self.action_history.append(action)
        
        # Store observation for metrics
        self.update_last_observation(obs)

        info = {
            "model_response": action,
            "raw_model_response": raw_action,
            "stats": token_usage,
        }
        if is_first_step:
            info["full_prompt"] = full_prompt_txt
        return action, info


@dataclasses.dataclass
class DemoAgentArgs(AbstractAgentArgs):
   

    model_name: str = "gpt-4o"
    chat_mode: bool = False
    demo_mode: str = "off"
    use_html: bool = False
    use_axtree: bool = True
    use_screenshot: bool = False
    system_message_handling: Literal["separate", "combined"] = "separate"
    system_prompt_append: Optional[str] = None
    prefix_prompt: Optional[str] = None
    thinking_type: Optional[str] = None

    # API keys and configuration - these can be None and the agent will fall back to environment variables
    openai_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    openrouter_site_url: Optional[str] = None
    openrouter_site_name: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # Seed for reproducibility - passed to OpenAI/OpenRouter as seed param,
    # for Anthropic sets temperature=0 (no native seed support)
    seed: Optional[int] = None

    # Reasoning toggle: True=enable, False=disable, None=provider default
    # Overrides :thinking suffix for Anthropic models
    reasoning: Optional[bool] = None

    # Explicit reasoning effort ("minimal"|"low"|"medium"|"high"|"none").
    # When set, overrides the default high/none derived from `reasoning`.
    reasoning_effort: Optional[str] = None

    # OpenRouter: pin routing to a specific provider (e.g. "fireworks", "together").
    # Sends provider.only=[provider] with allow_fallbacks=false. Ignored for non-OpenRouter models.
    provider: Optional[str] = None

    def make_agent(self):
        return DemoAgent(
            model_name=self.model_name,
            chat_mode=self.chat_mode,
            demo_mode=self.demo_mode,
            use_html=self.use_html,
            use_axtree=self.use_axtree,
            use_screenshot=self.use_screenshot,
            system_message_handling=self.system_message_handling,
            system_prompt_append=self.system_prompt_append,
            prefix_prompt=self.prefix_prompt,
            thinking_type=self.thinking_type,
            # Pass API keys and configuration
            openai_api_key=self.openai_api_key,
            openrouter_api_key=self.openrouter_api_key,
            openrouter_site_url=self.openrouter_site_url,
            openrouter_site_name=self.openrouter_site_name,
            anthropic_api_key=self.anthropic_api_key,
            seed=self.seed,
            reasoning=self.reasoning,
            reasoning_effort=self.reasoning_effort,
            provider=self.provider,
        )
