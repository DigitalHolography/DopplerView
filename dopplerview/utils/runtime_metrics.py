"""Lightweight runtime instrumentation for performance baselines."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

import psutil


logger = logging.getLogger(__name__)
_MEBIBYTE = 1024 * 1024


def available_cpu_count() -> int:
    """Return CPUs available to this process, respecting affinity when possible."""
    try:
        affinity = psutil.Process().cpu_affinity()
        if affinity:
            return len(affinity)
    except (AttributeError, NotImplementedError, psutil.Error):
        pass
    return os.cpu_count() or 1


def process_snapshot() -> Dict[str, Any]:
    """Take a cheap process-level memory/thread snapshot."""
    try:
        process = psutil.Process()
        memory = process.memory_info()
        return {
            "rss_mb": memory.rss / _MEBIBYTE,
            "vms_mb": memory.vms / _MEBIBYTE,
            "process_threads": process.num_threads(),
        }
    except psutil.Error:
        # Instrumentation must never make a scientific run fail.
        return {"rss_mb": 0.0, "vms_mb": 0.0, "process_threads": 0}


def _format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if value is None:
        return "none"
    return str(value).replace(" ", "_")


def emit_metric(kind: str, **values: Any) -> Dict[str, Any]:
    """Debug-log and return a structured metric record.

    Detailed telemetry is intentionally kept out of the normal user-facing log.
    Human-readable performance summaries are emitted by the components that own
    the corresponding operation.
    """
    record = {"kind": kind, **values}
    fields = " ".join(
        f"{key}={_format_metric_value(value)}" for key, value in record.items()
    )
    logger.debug("[Metrics] %s", fields)
    return record


class RuntimeMetrics:
    """Thread-safe in-memory collection attached to a pipeline Context."""

    def __init__(self) -> None:
        self._records = []
        self._lock = threading.Lock()

    def record(self, kind: str, **values: Any) -> Dict[str, Any]:
        record = emit_metric(kind, **values)
        with self._lock:
            self._records.append(record)
        return record

    def snapshot(self):
        with self._lock:
            return [dict(record) for record in self._records]

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


def record_for_context(ctx, kind: str, **values: Any) -> Dict[str, Any]:
    collector = getattr(ctx, "runtime_metrics", None)
    if collector is not None:
        return collector.record(kind, **values)
    return emit_metric(kind, **values)


@dataclass
class ProcessMemoryMeasurement:
    """Sample process RSS during an operation.

    RSS is process-wide. When DAG steps overlap, the peak cannot be attributed to
    one step alone; the metric deliberately keeps the ``process_*`` prefix.
    """

    interval_seconds: float = 0.05

    def __post_init__(self) -> None:
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.started = None
        self.finished = None
        self.process_peak_rss_mb = 0.0

    def _rss_mb(self) -> float:
        return self._process.memory_info().rss / _MEBIBYTE

    def start(self) -> "ProcessMemoryMeasurement":
        self.started = process_snapshot()
        self.process_peak_rss_mb = self.started["rss_mb"]
        self._thread = threading.Thread(
            target=self._sample,
            name="dopplerview-memory-sampler",
            daemon=True,
        )
        self._thread.start()
        return self

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.process_peak_rss_mb = max(
                    self.process_peak_rss_mb,
                    self._rss_mb(),
                )
            except psutil.Error:
                return

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.2, self.interval_seconds * 2))
        self.finished = process_snapshot()
        self.process_peak_rss_mb = max(
            self.process_peak_rss_mb,
            self.finished["rss_mb"],
        )
        return {
            "process_rss_start_mb": self.started["rss_mb"],
            "process_rss_end_mb": self.finished["rss_mb"],
            "process_rss_delta_mb": (
                self.finished["rss_mb"] - self.started["rss_mb"]
            ),
            "process_peak_rss_mb": self.process_peak_rss_mb,
            "process_threads_start": self.started["process_threads"],
            "process_threads_end": self.finished["process_threads"],
        }
