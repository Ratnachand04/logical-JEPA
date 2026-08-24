"""Console logging, metric averaging and CSV/JSON run records."""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict, deque
from typing import Any, Mapping

_LOG_FORMAT = "[%(asctime)s] %(levelname)-7s %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str = "logical_jepa", log_file: str | None = None,
               level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger, optionally tee-ing to ``log_file``."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured in this process
        return logger

    logger.setLevel(level)
    logger.propagate = False

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    logger.addHandler(stream)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        logger.addHandler(file_handler)

    return logger


class AverageMeter:
    """Running mean with a short window, for smooth progress-bar numbers."""

    def __init__(self, window: int = 100):
        self.window = window
        self._values: deque[float] = deque(maxlen=window)
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self._values.append(float(value))
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        """Mean over the whole run."""
        return self.total / self.count if self.count else 0.0

    @property
    def smooth(self) -> float:
        """Mean over the recent window."""
        return sum(self._values) / len(self._values) if self._values else 0.0

    def __format__(self, spec: str) -> str:
        return format(self.smooth, spec or ".4f")


class MetricTracker:
    """Collects several named AverageMeters for one epoch."""

    def __init__(self, window: int = 100):
        self._window = window
        self._meters: dict[str, AverageMeter] = defaultdict(lambda: AverageMeter(window))

    def update(self, **kwargs: float) -> None:
        for key, value in kwargs.items():
            self._meters[key].update(value)

    def __getitem__(self, key: str) -> AverageMeter:
        return self._meters[key]

    def averages(self) -> dict[str, float]:
        return {key: meter.avg for key, meter in self._meters.items()}

    def summary(self, precision: int = 4) -> str:
        return "  ".join(
            f"{key}={meter.smooth:.{precision}f}" for key, meter in self._meters.items()
        )

    def reset(self) -> None:
        self._meters.clear()


class Timer:
    """Context manager reporting wall-clock duration."""

    def __init__(self, label: str = "block", logger: logging.Logger | None = None):
        self.label = label
        self.logger = logger
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self._start
        message = f"{self.label} took {self.elapsed:.2f}s"
        if self.logger:
            self.logger.info(message)
        else:
            print(message)


def save_json(payload: Mapping[str, Any], path: str) -> None:
    """Write a metrics dict as pretty JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, default=str)


def append_csv(row: Mapping[str, Any], path: str) -> None:
    """Append one row to a CSV, writing the header on first use.

    Used by ``run_ablations.py`` so a partially finished sweep still leaves a
    readable results table.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(dict(row))


def format_table(rows: list[Mapping[str, Any]], float_fmt: str = ".4f") -> str:
    """Render a list of dicts as a fixed-width text table for the console."""
    if not rows:
        return "(no rows)"

    headers = list(rows[0].keys())

    def cell(value: Any) -> str:
        if isinstance(value, float):
            return format(value, float_fmt)
        return str(value)

    table = [headers] + [[cell(row.get(h, "")) for h in headers] for row in rows]
    widths = [max(len(r[i]) for r in table) for i in range(len(headers))]

    lines = [
        "  ".join(h.ljust(w) for h, w in zip(table[0], widths)),
        "  ".join("-" * w for w in widths),
    ]
    lines += ["  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in table[1:]]
    return "\n".join(lines)
