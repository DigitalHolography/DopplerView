from __future__ import annotations

def test_output_worker_starts_and_stops(bare_output_manager_factory):
    manager = bare_output_manager_factory()

    manager.start()
    worker = manager.output_worker
    assert worker is not None and worker.is_alive()

    manager.close_workers()
    assert not worker.is_alive()
    assert manager.running is False


def test_output_worker_can_restart_and_close_is_idempotent(
    bare_output_manager_factory,
):
    manager = bare_output_manager_factory()

    manager.start()
    first_worker = manager.output_worker
    manager.close_workers()
    manager.close_workers()

    manager.start()
    second_worker = manager.output_worker
    try:
        assert second_worker is not first_worker
        assert second_worker.is_alive()
    finally:
        manager.close_workers()


def test_close_flushes_accepted_output_work(bare_output_manager_factory):
    manager = bare_output_manager_factory()
    saved = []
    manager.save = lambda step, key, ctx: saved.append((step, key, ctx))
    manager.start()

    manager.save_async("first", "a", 1)
    manager.save_async("second", "b", 2)
    manager.close_workers()

    assert saved == [("first", "a", 1), ("second", "b", 2)]


def test_shutdown_clears_worker_references(bare_output_manager_factory):
    manager = bare_output_manager_factory()
    manager.start()

    manager.close_workers()

    assert manager.output_worker is None
    assert manager.cache_worker is None


def test_cache_worker_can_restart_after_shutdown(
    bare_output_manager_factory, tmp_path
):
    manager = bare_output_manager_factory()

    # Avoid HDF5 work: this worker only demonstrates the lifecycle contract.
    def cache_worker():
        while True:
            item = manager.cache_queue.get()
            if item == (None, None, None):
                manager.cache_queue.task_done()
                return
            manager.cache_queue.task_done()

    import threading

    manager._cache_worker = cache_worker
    manager.cache_path = tmp_path / "cache.h5"
    manager.start()
    manager.cache_async(object(), "hash", "step")
    manager.close_workers()

    manager.start()
    try:
        manager.cache_async(object(), "hash", "step")
        assert manager.cache_worker.is_alive()
    finally:
        manager.close_workers()
