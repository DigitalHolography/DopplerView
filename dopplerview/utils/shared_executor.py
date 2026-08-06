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


class SharedExecutor:
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

    def shutdown(self, wait=True):
        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=wait, cancel_futures=not wait)
