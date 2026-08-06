from __future__ import annotations

import queue
import pytest

from dopplerview.input_output.output_manager import OutputManager


def bare_output_manager():
    manager = OutputManager.__new__(OutputManager)
    manager.running = False
    manager.output_queue = queue.Queue()
    manager.cache_queue = queue.Queue()
    manager.output_worker = None
    manager.cache_worker = None
    return manager


def test_output_worker_starts_and_stops():
    manager = bare_output_manager()

    manager.start()
    worker = manager.output_worker
    assert worker is not None and worker.is_alive()

    manager.close_workers()
    assert not worker.is_alive()
    assert manager.running is False


@pytest.mark.xfail(
    strict=True,
    reason="close_workers currently leaves stale worker references behind.",
)
def test_shutdown_clears_worker_references():
    manager = bare_output_manager()
    manager.start()

    manager.close_workers()

    assert manager.output_worker is None
    assert manager.cache_worker is None


@pytest.mark.xfail(
    strict=True,
    reason="A stopped cache-worker reference prevents cache_async from restarting it.",
)
def test_cache_worker_can_restart_after_shutdown():
    manager = bare_output_manager()

    # Avoid HDF5 work: this worker only demonstrates the lifecycle contract.
    def cache_worker():
        while manager.running:
            item = manager.cache_queue.get()
            if item == (None, None, None):
                return

    import threading

    manager._cache_worker = cache_worker
    manager.running = True
    manager.cache_worker = threading.Thread(target=cache_worker, daemon=True)
    manager.cache_worker.start()
    manager.close_workers()

    manager.start()
    try:
        assert manager.cache_worker.is_alive()
    finally:
        manager.close_workers()
