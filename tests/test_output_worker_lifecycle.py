from __future__ import annotations

import pytest

def test_output_worker_starts_and_stops(bare_output_manager_factory):
    manager = bare_output_manager_factory()

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
def test_shutdown_clears_worker_references(bare_output_manager_factory):
    manager = bare_output_manager_factory()
    manager.start()

    manager.close_workers()

    assert manager.output_worker is None
    assert manager.cache_worker is None


@pytest.mark.xfail(
    strict=True,
    reason="A stopped cache-worker reference prevents cache_async from restarting it.",
)
def test_cache_worker_can_restart_after_shutdown(bare_output_manager_factory):
    manager = bare_output_manager_factory()

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
