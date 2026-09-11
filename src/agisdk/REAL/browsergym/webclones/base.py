import os
import logging

import playwright.sync_api
from agisdk.REAL.browsergym.core.task import AbstractBrowserTask
from agisdk.REAL.browsergym.webclones.task_config import TaskConfig
from agisdk.REAL.logging import logger as rich_logger

logger = logging.getLogger(__name__)


class AbstractWebCloneTask(AbstractBrowserTask):
    """
    Abstract class for all WebClones tasks
    """

    @classmethod
    def get_task_id(cls):
        return cls.task_id

    def __init__(self, seed: int, task_id: str, task_source: str = None) -> None:
        """
        Args:
            seed: Random seed for the task.
            task_id: ID of the task to load.
            task_source: Optional path to the task file holding `task_id`.
                         The start URL falls back to the WEBCLONE_URL
                         environment variable when the task does not define one.
        """
        super().__init__(seed)

        self.seed = seed
        self.task_id = task_id
        self.task_config = TaskConfig(self.task_id, task_source=task_source)
        if not self.task_config.is_valid_config():
            raise ValueError(f"Invalid task configuration for task ID: {self.task_id}")

        self.goal = self.task_config.get_goal()
        self.url = self.task_config.get_start_url()
        if not self.url:
            if "WEBCLONE_URL" in os.environ:
                self.url = os.environ["WEBCLONE_URL"]
            else:
                raise ValueError("Provide a WebClones base URL or set it up as WEBCLONE_URL env var.")
        rich_logger.info(f"⚙️ Initialized {self.task_id} task.")
        rich_logger.info(f"🎯 Goal: {self.goal}")

    def setup(self, page: playwright.sync_api.Page) -> str:
        self.page = page
        self.page.bring_to_front()  # Ensure main page stays focused
        self.page.goto(self.url)
        return self.goal

    def teardown(self) -> None:
        self.page.close()

    def validate(
        self,
        page: playwright.sync_api.Page,
        chat_messages: list[str],
    ) -> tuple[float, bool]:
        """End the episode once the agent has sent its answer.

        The first assistant message is the harness greeting, so a second one is
        the agent's own reply. Scoring happens offline in ``evaluation/``, so no
        reward is computed here.
        """
        assistant_messages = [m for m in chat_messages if m["role"] == "assistant"]
        done = len(assistant_messages) > 1
        return 0.0, done
