#!/usr/bin/env python3
"""Stage 1 screenshot filtering: decide whether two consecutive frames differ enough to judge.

Two metrics are used, both computed on grayscale frames downscaled to a common
size:

``rmse``
    ImageMagick's normalised root-mean-square error — how *far* the frames
    differ.

``changed_fraction``
    The fraction of pixels that differ at all, computed from a thresholded
    difference composite.

A perceptual-hash metric used to be included as well.  Across the 409,786
recorded stage-1 comparisons it was the sole trigger for 372 of 185,399 flags
(0.2%), while rmse was sole trigger 16,288 times and changed_fraction 41,042
times.  It also reports 0 for plainly different frames on ImageMagick 7, so it
has been dropped rather than repaired.

``changed_fraction`` is deliberately *not* taken from ``-metric AE``.  Under
ImageMagick 6 that returned a count of differing pixels, but under 7 it returns
the summed absolute difference in quantum units — for one sample pair, 54,536
changed pixels versus an AE implying 3,196.  Thresholding the difference
composite restores the original count semantics and is stable across versions.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


PAREN_RE = re.compile(r"\(([^)]+)\)")
SCREENSHOT_RE = re.compile(r"screenshot_step_(\d+)\.png$")
DEFAULT_COMPARE_SIZE = (320, 200)


@dataclass(frozen=True)
class ScreenshotDiffThresholds:
    rmse: float = 0.03
    changed_fraction: float = 0.01


@dataclass(frozen=True)
class ScreenshotDiffResult:
    first_path: str
    second_path: str
    rmse: float
    changed_pixels: int
    total_pixels: int
    changed_fraction: float
    significance_score: float
    is_significant: bool
    triggered_metrics: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def extract_step_number(path: Path) -> int:
    match = SCREENSHOT_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"Unexpected screenshot name: {path}")
    return int(match.group(1))


def sorted_screenshot_paths(folder: Path) -> list[Path]:
    screenshots = [path for path in folder.iterdir() if path.is_file() and SCREENSHOT_RE.fullmatch(path.name)]
    return sorted(screenshots, key=extract_step_number)


def compare_screenshots(
    first_path: str | Path,
    second_path: str | Path,
    thresholds: ScreenshotDiffThresholds | None = None,
    compare_size: tuple[int, int] = DEFAULT_COMPARE_SIZE,
) -> ScreenshotDiffResult:
    first = Path(first_path)
    second = Path(second_path)
    if not first.exists():
        raise FileNotFoundError(first)
    if not second.exists():
        raise FileNotFoundError(second)

    for tool in ("compare", "magick"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"ImageMagick `{tool}` is required but was not found in PATH.")

    thresholds = thresholds or ScreenshotDiffThresholds()
    resize_geometry = f"{compare_size[0]}x{compare_size[1]}!"
    total_pixels = compare_size[0] * compare_size[1]

    rmse = _run_rmse_metric(first, second, resize_geometry)
    changed_fraction = _run_changed_fraction(first, second, resize_geometry)
    changed_pixels = int(round(changed_fraction * total_pixels))

    triggered_metrics = []
    if rmse >= thresholds.rmse:
        triggered_metrics.append("rmse")
    if changed_fraction >= thresholds.changed_fraction:
        triggered_metrics.append("changed_fraction")

    significance_score = max(
        rmse / thresholds.rmse if thresholds.rmse > 0 else 0.0,
        changed_fraction / thresholds.changed_fraction if thresholds.changed_fraction > 0 else 0.0,
    )

    return ScreenshotDiffResult(
        first_path=str(first),
        second_path=str(second),
        rmse=rmse,
        changed_pixels=changed_pixels,
        total_pixels=total_pixels,
        changed_fraction=changed_fraction,
        significance_score=significance_score,
        is_significant=bool(triggered_metrics),
        triggered_metrics=tuple(triggered_metrics),
    )


def _run_rmse_metric(first: Path, second: Path, resize_geometry: str) -> float:
    """Normalised RMSE, taken from the parenthesised value.

    ImageMagick prints ``<absolute> (<normalised>)``; only the normalised value
    is comparable across quantum depths and builds.
    """
    command = [
        "compare",
        "-colorspace", "Gray",
        "-resize", resize_geometry,
        "-metric", "RMSE",
        str(first), str(second),
        "null:",
    ]
    output = _run(command)
    match = PAREN_RE.search(output)
    if not match:
        raise RuntimeError(f"Could not parse RMSE output from ImageMagick compare: {output!r}")
    return float(match.group(1))


def _run_changed_fraction(first: Path, second: Path, resize_geometry: str) -> float:
    """Fraction of pixels that differ at all.

    The two frames are differenced and thresholded at 0, so every pixel is
    either changed or not; the mean of that mask is the changed fraction.
    """
    command = [
        "magick",
        str(first), str(second),
        "-colorspace", "Gray",
        "-resize", resize_geometry,
        "-compose", "difference", "-composite",
        "-threshold", "0",
        "-format", "%[fx:mean]",
        "info:",
    ]
    output = _run(command).strip()
    if not output:
        raise RuntimeError("Could not parse changed-fraction output from ImageMagick magick.")
    try:
        return float(output)
    except ValueError as exc:
        raise RuntimeError(f"Unexpected changed-fraction output from ImageMagick: {output!r}") from exc


def _run(command: list[str]) -> str:
    # `compare` exits 1 when the images differ, which is not an error here.
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode not in (0, 1):
        raise RuntimeError(
            "ImageMagick failed with "
            f"exit code {completed.returncode}: {(completed.stderr or completed.stdout).strip()}"
        )
    return completed.stdout or completed.stderr
