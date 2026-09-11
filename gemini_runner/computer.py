# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import abc
from dataclasses import dataclass
from typing import Literal

@dataclass
class EnvState:
    """Current environment state returned to the model."""

    screenshot: bytes
    url: str


class Computer(abc.ABC):
    """Defines an interface for browser environments."""

    @abc.abstractmethod
    def screen_size(self) -> tuple[int, int]:
        """Returns environment screen size."""

    @abc.abstractmethod
    def open_web_browser(self) -> EnvState:
        """Opens browser and returns current state."""

    @abc.abstractmethod
    def click_at(self, x: int, y: int) -> EnvState:
        """Clicks at absolute x/y coordinates."""

    @abc.abstractmethod
    def hover_at(self, x: int, y: int) -> EnvState:
        """Hovers at absolute x/y coordinates."""

    @abc.abstractmethod
    def type_text_at(
        self,
        x: int,
        y: int,
        text: str,
        press_enter: bool,
        clear_before_typing: bool,
    ) -> EnvState:
        """Types text at coordinates."""

    @abc.abstractmethod
    def scroll_document(self, direction: Literal["up", "down", "left", "right"]) -> EnvState:
        """Scrolls page in given direction."""

    @abc.abstractmethod
    def scroll_at(
        self,
        x: int,
        y: int,
        direction: Literal["up", "down", "left", "right"],
        magnitude: int,
    ) -> EnvState:
        """Scrolls near x/y coordinates."""

    @abc.abstractmethod
    def wait_5_seconds(self) -> EnvState:
        """Pauses for 5 seconds."""

    @abc.abstractmethod
    def go_back(self) -> EnvState:
        """Navigates back in history."""

    @abc.abstractmethod
    def go_forward(self) -> EnvState:
        """Navigates forward in history."""

    @abc.abstractmethod
    def search(self) -> EnvState:
        """Navigates to search engine."""

    @abc.abstractmethod
    def navigate(self, url: str) -> EnvState:
        """Navigates to URL."""

    @abc.abstractmethod
    def key_combination(self, keys: list[str]) -> EnvState:
        """Presses key combination."""

    @abc.abstractmethod
    def drag_and_drop(self, x: int, y: int, destination_x: int, destination_y: int) -> EnvState:
        """Drags from one coordinate to another."""

    @abc.abstractmethod
    def current_state(self) -> EnvState:
        """Returns current environment state."""
