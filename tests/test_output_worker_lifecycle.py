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
    manager.save = lambda payload: saved.append(payload)
    manager.start()

    class Context:
        def __init__(self, value):
            self.value = value

        def get(self, key):
            return self.value

        def has(self, key):
            return True

    manager.save_async("first", "a", Context(1))
    manager.save_async("second", "b", Context(2))
    manager.close_workers()

    assert [(item.step_name, item.key, item.value) for item in saved] == [
        ("first", "a", 1),
        ("second", "b", 2),
    ]


def test_each_run_gets_a_new_debug_output_folder(
    bare_output_manager_factory, tmp_path
):
    manager = bare_output_manager_factory()
    manager.output_enabled = True
    manager.write_dopplerview_config = lambda: None
    manager.write_version_file = lambda: None

    class RunFolder:
        def __init__(self):
            self.index = -1
            self.measure_name = "measure"

        def get_h5_path(self):
            return tmp_path / "measure_DV.h5"

        def create_output_folder(self):
            self.index += 1
            path = tmp_path / f"output_{self.index}"
            path.mkdir()
            return path

    manager.set_DV_folder(RunFolder())

    manager.begin_run()
    manager.ensure_output_folder()
    first_output = manager.output_dir

    manager.begin_run()
    manager.ensure_output_folder()
    second_output = manager.output_dir

    assert first_output == tmp_path / "output_0"
    assert second_output == tmp_path / "output_1"


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
            if item is None:
                manager.cache_queue.task_done()
                return
            manager.cache_queue.task_done()

    import threading

    manager._cache_worker = cache_worker
    manager.cache_path = tmp_path / "cache.h5"
    class Context:
        def get_produced_values(self):
            return {}

        def cache_values(self, keys):
            pass

    ctx = Context()
    manager.start()
    manager.cache_async(ctx, "hash", "step")
    manager.close_workers()

    manager.start()
    try:
        manager.cache_async(ctx, "hash", "step")
        assert manager.cache_worker.is_alive()
    finally:
        manager.close_workers()
