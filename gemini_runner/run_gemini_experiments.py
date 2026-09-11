#!/usr/bin/env python3
"""Run many run_gemini.py experiments from a JSON config."""

from __future__ import annotations

import fire

from batch import run_from_config


def run(
    config: str,
    max_parallel: int | None = None,
    fail_fast: bool = False,
    dry_run: bool = False,
    use_screenshot: bool | None = None,
    continue_run: bool = False,
    skip_existing_results: bool = False,
) -> int:
    return run_from_config(
        config=config,
        max_parallel=max_parallel,
        fail_fast=fail_fast,
        dry_run=dry_run,
        use_screenshot=use_screenshot,
        continue_run=continue_run,
        skip_existing_results=skip_existing_results,
    )


if __name__ == "__main__":
    fire.Fire(run)
