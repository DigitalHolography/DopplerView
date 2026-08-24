"""One bounded executor shared by every internally parallel pipeline operation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Callable, Iterable

import numpy as np

from dopplerview.utils.parallelization_utils import compute_n_jobs
from dopplerview.utils.runtime_metrics import emit_metric, process_snapshot
from dopplerview.utils.cancellation import CancellationToken


logger = logging.getLogger(__name__)


def _run_indexed_chunk(function, indexed_items, cancellation):
    results = []
    for index, item in indexed_items:
        cancellation.check()
        results.append((index, function(item)))
    return results


def _run_indexed_chunk_into(function, indexed_items, output, cancellation):
    for index, item in indexed_items:
        cancellation.check()
        output[index] = function(item)


class SharedExecutor:
    """
    A single executor shared by all parallel operations in the pipeline.
    This is a workaround for the fact that some parallelization libraries (e.g. OpenCV) do not play well with multiple concurrent executors in the same process.
    The executor is bounded to a maximum number of workers, which can be configured by the user. The executor can be used to run parallel operations in a thread-safe manner, and it will respect the cancellation token if provided.
    """
    def __init__(
        self,
        max_workers: int,
        available_cpus: int,
        cancellation: CancellationToken | None = None,
    ):
        self.max_workers = max(1, int(max_workers))
        self.available_cpus = max(1, int(available_cpus))
        self._executor = (
            ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="dopplerview-compute",
            )
            if self.max_workers > 1
            else None
        )
        self._closed = False
        self.cancellation = cancellation or CancellationToken()

    def resolve_workers(self, requested_workers) -> int:
        requested = compute_n_jobs(
            requested_workers,
            cpu_count=self.available_cpus,
        )
        return max(1, min(requested, self.max_workers))

    def map(
        self,
        function: Callable,
        iterable: Iterable,
        *,
        n_jobs=-1,
        chunking=True,
        task_name=None,
    ):
        if self._closed:
            raise RuntimeError("Shared executor is closed.")
        self.cancellation.check()

        items = list(iterable)
        if not items:
            return np.empty((0,))

        effective_workers = min(self.resolve_workers(n_jobs), len(items))
        snapshot = process_snapshot()
        emit_metric(
            "parallel_operation",
            task=task_name or getattr(function, "__name__", "anonymous"),
            requested_workers=n_jobs,
            effective_workers=effective_workers,
            shared_capacity=self.max_workers,
            available_cpus=self.available_cpus,
            items=len(items),
            chunking=chunking,
            process_threads=snapshot["process_threads"],
        )
        if task_name is not None:
            logger.info(
                "    - Running %s with %d shared worker(s)",
                task_name,
                effective_workers,
            )

        if effective_workers == 1:
            results = []
            for item in items:
                self.cancellation.check()
                results.append(function(item))
            return np.stack(results, axis=0)

        indexed = list(enumerate(items))
        index_chunks = np.array_split(np.arange(len(indexed)), effective_workers)
        chunks = [[indexed[index] for index in indices] for indices in index_chunks]
        futures = [
            self._executor.submit(
                _run_indexed_chunk,
                function,
                chunk,
                self.cancellation,
            )
            for chunk in chunks
        ]
        ordered_results = [None] * len(items)
        for future in futures:
            for index, result in future.result():
                ordered_results[index] = result
        return np.stack(ordered_results, axis=0)

    def map_into(
        self,
        function: Callable,
        iterable: Iterable,
        output: np.ndarray,
        *,
        n_jobs=-1,
        task_name=None,
    ):
        """Map directly into a caller-owned array without a final stack copy."""
        if self._closed:
            raise RuntimeError("Shared executor is closed.")
        self.cancellation.check()

        items = list(iterable)
        if len(items) != len(output):
            raise ValueError("Output length must match the mapped iterable length.")
        if not items:
            return output

        effective_workers = min(self.resolve_workers(n_jobs), len(items))
        snapshot = process_snapshot()
        emit_metric(
            "parallel_operation",
            task=task_name or getattr(function, "__name__", "anonymous"),
            requested_workers=n_jobs,
            effective_workers=effective_workers,
            shared_capacity=self.max_workers,
            available_cpus=self.available_cpus,
            items=len(items),
            chunking=True,
            output_preallocated=True,
            process_threads=snapshot["process_threads"],
        )
        if task_name is not None:
            logger.info(
                "    - Running %s with %d shared worker(s)",
                task_name,
                effective_workers,
            )

        indexed = list(enumerate(items))
        if effective_workers == 1:
            _run_indexed_chunk_into(
                function, indexed, output, self.cancellation
            )
            return output

        index_chunks = np.array_split(np.arange(len(indexed)), effective_workers)
        futures = [
            self._executor.submit(
                _run_indexed_chunk_into,
                function,
                [indexed[index] for index in indices],
                output,
                self.cancellation,
            )
            for indices in index_chunks
        ]
        for future in futures:
            future.result()
        return output

    def shutdown(self, wait=True):
        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=wait, cancel_futures=not wait)
