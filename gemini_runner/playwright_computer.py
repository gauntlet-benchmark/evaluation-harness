"""Playwright-backed Computer implementation with artifact hooks."""

from __future__ import annotations

import gzip
import pickle
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from playwright.sync_api import Browser, BrowserContext, Page, Playwright

from .computer import Computer, EnvState


class PlaywrightComputer(Computer):
    def __init__(
        self,
        *,
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 800,
        initial_url: str | None = None,
        inject_proxy_select: bool = False,
        proxy_select_js: str = "",
        proxy_select_css: str = "",
        exp_dir: Path | None = None,
    ) -> None:
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.initial_url = initial_url
        self.inject_proxy_select = inject_proxy_select
        self.proxy_select_js = proxy_select_js
        self.proxy_select_css = proxy_select_css
        self.exp_dir = exp_dir
        self._playwright: "Playwright | None" = None
        self._browser: "Browser | None" = None
        self._context: "BrowserContext | None" = None
        self.page: "Page | None" = None

    def __enter__(self) -> "PlaywrightComputer":
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context(viewport={"width": self.viewport_width, "height": self.viewport_height})
        self.page = self._context.new_page()
        if self.inject_proxy_select and self.proxy_select_js:
            self.page.add_init_script(self.proxy_select_js)

            if self.proxy_select_css:
                def inject_style(page: "Page") -> None:
                    try:
                        page.add_style_tag(content=self.proxy_select_css)
                    except Exception:
                        pass

                self.page.on("domcontentloaded", inject_style)

        if self.initial_url:
            self._goto_initial_url()
        return self

    def _goto_initial_url(self) -> None:
        """Open the task's start page, tolerating a slow or flaky first load.

        Playwright's default 30s is tight for these deployments — cold starts
        have been measured in the teens — and a single timeout here aborts the
        whole experiment, losing every task queued behind it. So allow longer
        and retry before giving up.
        """
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                self.page.goto(self.initial_url, timeout=90_000)
                return
            except Exception as exc:  # playwright TimeoutError and transport errors
                last_error = exc
                print(
                    f"[playwright] initial navigation to {self.initial_url} failed "
                    f"(attempt {attempt + 1}/3): {type(exc).__name__}",
                    flush=True,
                )
        raise RuntimeError(
            f"Could not open {self.initial_url} after 3 attempts: {last_error}"
        ) from last_error

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._context is not None:
            self._context.close()
        if self._browser is not None:
            self._browser.close()
        if self._playwright is not None:
            self._playwright.stop()

    def _require_page(self) -> "Page":
        if self.page is None:
            raise RuntimeError("PlaywrightComputer is not initialized. Use as context manager.")
        return self.page

    def _capture_state(self) -> EnvState:
        page = self._require_page()
        screenshot = page.screenshot(full_page=False, type="png")
        return EnvState(screenshot=screenshot, url=page.url)

    def screen_size(self) -> tuple[int, int]:
        return self.viewport_width, self.viewport_height

    def open_web_browser(self) -> EnvState:
        page = self._require_page()
        if not page.url or page.url == "about:blank":
            page.goto("https://www.google.com")
        return self._capture_state()

    def click_at(self, x: int, y: int) -> EnvState:
        page = self._require_page()
        page.mouse.click(x, y)
        return self._capture_state()

    def hover_at(self, x: int, y: int) -> EnvState:
        page = self._require_page()
        page.mouse.move(x, y)
        return self._capture_state()

    def type_text_at(
        self,
        x: int,
        y: int,
        text: str,
        press_enter: bool,
        clear_before_typing: bool,
    ) -> EnvState:
        page = self._require_page()
        page.mouse.click(x, y)
        if clear_before_typing:
            page.keyboard.press("Meta+A")
            page.keyboard.press("Backspace")
        page.keyboard.type(text)
        if press_enter:
            page.keyboard.press("Enter")
        return self._capture_state()

    def scroll_document(self, direction: Literal["up", "down", "left", "right"]) -> EnvState:
        page = self._require_page()
        if direction == "down":
            page.mouse.wheel(0, 800)
        elif direction == "up":
            page.mouse.wheel(0, -800)
        elif direction == "right":
            page.mouse.wheel(800, 0)
        else:
            page.mouse.wheel(-800, 0)
        return self._capture_state()

    def scroll_at(self, x: int, y: int, direction: Literal["up", "down", "left", "right"], magnitude: int) -> EnvState:
        page = self._require_page()
        page.mouse.move(x, y)
        if direction == "down":
            page.mouse.wheel(0, magnitude)
        elif direction == "up":
            page.mouse.wheel(0, -magnitude)
        elif direction == "right":
            page.mouse.wheel(magnitude, 0)
        else:
            page.mouse.wheel(-magnitude, 0)
        return self._capture_state()

    def wait_5_seconds(self) -> EnvState:
        time.sleep(5)
        return self._capture_state()

    def go_back(self) -> EnvState:
        page = self._require_page()
        page.go_back()
        return self._capture_state()

    def go_forward(self) -> EnvState:
        page = self._require_page()
        page.go_forward()
        return self._capture_state()

    def search(self) -> EnvState:
        page = self._require_page()
        page.goto("https://www.google.com")
        return self._capture_state()

    def navigate(self, url: str) -> EnvState:
        page = self._require_page()
        page.goto(url)
        return self._capture_state()

    def key_combination(self, keys: list[str]) -> EnvState:
        page = self._require_page()
        normalized_keys = [self._to_playwright_key(key) for key in keys if key and key.strip()]
        combo = "+".join(key for key in normalized_keys if key)
        if not combo:
            raise ValueError("No valid keys provided for key_combination.")
        page.keyboard.press(combo)
        return self._capture_state()

    def drag_and_drop(self, x: int, y: int, destination_x: int, destination_y: int) -> EnvState:
        page = self._require_page()
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(destination_x, destination_y)
        page.mouse.up()
        return self._capture_state()

    def current_state(self) -> EnvState:
        return self._capture_state()

    def save_step_artifact(
        self,
        *,
        step_idx: int,
        action: str,
        state: EnvState,
        raw_model_response: str | None,
        token_usage: dict[str, Any] | None = None,
        last_action_error: str | None = None,
        terminated: bool = False,
        truncated: bool = False,
    ) -> None:
        if self.exp_dir is None:
            return
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / f"screenshot_step_{step_idx}.png").write_bytes(state.screenshot)
        step_payload = {
            "step": step_idx,
            "obs": {
                "url": state.url,
                "last_action": action,
                "last_action_error": last_action_error,
                "screenshot": None,
            },
            "reward": 0.0,
            "raw_reward": 0.0,
            "terminated": terminated,
            "truncated": truncated,
            "action": action,
            "agent_info": {
                "model_response": action,
                "raw_model_response": raw_model_response,
            },
            "stats": token_usage or {},
            "task_info": {},
        }
        with gzip.open(self.exp_dir / f"step_{step_idx}.pkl.gz", "wb") as handle:
            pickle.dump(step_payload, handle)

    @staticmethod
    def _to_playwright_key(value: str) -> str:
        normalized = value.strip()
        lower = normalized.lower()
        mapping = {
            "control": "Control",
            "ctrl": "Control",
            "command": "Meta",
            "cmd": "Meta",
            "option": "Alt",
            "opt": "Alt",
            "alt": "Alt",
            "shift": "Shift",
            "meta": "Meta",
            "super": "Meta",
            "win": "Meta",
            "escape": "Escape",
            "esc": "Escape",
            "return": "Enter",
            "left": "ArrowLeft",
            "right": "ArrowRight",
            "up": "ArrowUp",
            "down": "ArrowDown",
            "arrowleft": "ArrowLeft",
            "arrowright": "ArrowRight",
            "arrowup": "ArrowUp",
            "arrowdown": "ArrowDown",
            "del": "Delete",
            "delete": "Delete",
            "backspace": "Backspace",
            "tab": "Tab",
            "home": "Home",
            "end": "End",
            "insert": "Insert",
            "pageup": "PageUp",
            "pagedown": "PageDown",
            "pgup": "PageUp",
            "pgdn": "PageDown",
            "space": " ",
        }
        return mapping.get(lower, normalized)
